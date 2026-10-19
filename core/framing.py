"""NDV5 framing, integrity, exact sizes, and incremental recovery."""
from dataclasses import dataclass, replace
from array import array
import math
import struct
import zlib

from .config import MAGIC, EMBEDDING_ID, HEADER_COPIES
from .ecc import (
    BODY_DATA, BODY_SIZE, HEADER_SIZE, INTERLEAVE_DEPTH,
    encode_header, decode_header,
    encode_body_group, decode_body_group,
)

# magic, embedding/ECC/depth/flags, payload length, filename length,
# encoded body length, reserved bytes, CRC32 of the preceding 28 bytes.
BOOTSTRAP = struct.Struct('>4sBBBBQHQ2sI')
CRC = struct.Struct('>I')
ECC_PROFILE = 1
MAX_FILENAME_BYTES = 65535
# Maximum imposed by the unsigned 64-bit encoded body length.
MAX_PAYLOAD_BYTES = ((2**64 - 1) // BODY_SIZE) * BODY_DATA - 1 - CRC.size


@dataclass(frozen=True)
class Header:
    payload_length: int
    filename_length: int
    body_length: int
    corrected_bytes: int = 0
    recovered_copies: int = 1

    @property
    def total_length(self):
        return HEADER_SIZE * HEADER_COPIES + self.body_length


@dataclass(frozen=True)
class Recovery:
    filename: str
    payload_length: int
    header_corrections: int = 0
    header_copies_recovered: int = 1
    confidence_model: str = 'minimum'
    body_correction_total: int = 0
    body_correction_max: int = 0
    body_erasure_total: int = 0
    body_erasure_words: int = 0


def filename_bytes(name):
    if (not name or name in ('.', '..') or any(c in name for c in ('/', '\\', '\0', ':'))):
        raise ValueError('Stored filename is unsafe or damaged')
    encoded = name.encode('utf-8')
    if not 1 <= len(encoded) <= MAX_FILENAME_BYTES:
        raise ValueError('Filename must contain 1–65535 UTF-8 bytes')
    return encoded


def _validate_lengths(payload_length, name_length):
    if not 0 <= payload_length <= MAX_PAYLOAD_BYTES:
        raise ValueError(f'Payload exceeds supported limit of {MAX_PAYLOAD_BYTES} bytes')
    if not 1 <= name_length <= MAX_FILENAME_BYTES:
        raise ValueError('Invalid filename length')


def encoded_body_size(payload_length, name_length):
    _validate_lengths(payload_length, name_length)
    raw_length = name_length + payload_length + CRC.size
    size = BODY_SIZE * ((raw_length + BODY_DATA - 1) // BODY_DATA)
    if size >= 2**64:
        raise ValueError("Encoded body exceeds wire-format length")
    return size


def encoded_size(payload_length, name_length):
    return HEADER_SIZE * HEADER_COPIES + encoded_body_size(payload_length, name_length)


def maximum_payload(capacity, name_length):
    _validate_lengths(0, name_length)
    if capacity < encoded_size(0, name_length):
        return -1
    return min((2**64 - 1) // BODY_SIZE,
               (capacity - HEADER_SIZE * HEADER_COPIES) // BODY_SIZE) * BODY_DATA - name_length - CRC.size


def _pack_header(payload_length, name_length):
    body_length = encoded_body_size(payload_length, name_length)
    raw = BOOTSTRAP.pack(MAGIC, EMBEDDING_ID, ECC_PROFILE,
                         INTERLEAVE_DEPTH, 0, payload_length, name_length,
                         body_length, b'\0\0', 0)
    raw = raw[:-CRC.size] + CRC.pack(zlib.crc32(raw[:-CRC.size]))
    return encode_header(raw) * HEADER_COPIES


class MessageWriter:
    """Sequential wire-byte source with bounded payload reads and RS groups.

    The caller owns the source and keeps it unchanged through verification.
    """
    def __init__(self, name, source, payload_length):
        self.name = filename_bytes(name)
        self.source = source
        self.payload_length = payload_length
        self.length = encoded_size(payload_length, len(self.name))
        self.origin = source.tell()
        self._chunks = self._generate()
        self.pending = b''

    def __len__(self):
        return self.length

    def rewind(self):
        self.source.seek(self.origin)
        self.pending = b''
        self._chunks = self._generate()

    def _generate(self):
        yield _pack_header(self.payload_length, len(self.name))
        size = BODY_DATA * INTERLEAVE_DEPTH
        pending = bytearray()
        checksum = zlib.crc32(self.name)
        for start in range(0, len(self.name), size):
            pending.extend(self.name[start:start + size])
            if len(pending) == size:
                yield encode_body_group(pending)
                pending.clear()
        remaining = self.payload_length
        while remaining:
            chunk = self.source.read(min(size - len(pending), remaining))
            if not chunk:
                raise ValueError('Payload ended before its declared length')
            remaining -= len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            pending.extend(chunk)
            if len(pending) == size:
                yield encode_body_group(pending)
                pending.clear()
        if self.source.read(1):
            raise ValueError('Payload exceeds its declared length')
        pending.extend(CRC.pack(checksum))
        for start in range(0, len(pending), size):
            yield encode_body_group(pending[start:start + size])

    def read(self, size):
        if size < 0:
            raise ValueError('A bounded read size is required')
        result = bytearray()
        while len(result) < size:
            if not self.pending:
                self.pending = next(self._chunks, b'')
                if not self.pending:
                    break
            take = min(size - len(result), len(self.pending))
            result.extend(self.pending[:take])
            self.pending = self.pending[take:]
        return bytes(result)


def parse_header(data):
    raw, corrected = decode_header(data)
    magic, embedding, ecc, depth, flags, length, name_length, body_length, reserved, crc = BOOTSTRAP.unpack(raw)
    if zlib.crc32(raw[:-CRC.size]) != crc:
        raise ValueError('Bootstrap CRC32 mismatch after error correction')
    if (magic, embedding) != (MAGIC, EMBEDDING_ID):
        raise ValueError('Unsupported bootstrap mapping')
    if (ecc, depth, flags, reserved) != (
            ECC_PROFILE, INTERLEAVE_DEPTH, 0, b'\0\0'):
        raise ValueError('Unsupported bootstrap ECC profile, flags, or reserved fields')
    expected = encoded_body_size(length, name_length)
    if body_length != expected:
        raise ValueError('Inconsistent protected body length')
    return Header(length, name_length, body_length, corrected)


def parse_bootstrap(data):
    """Repair independent copies; never choose between conflicting valid metadata."""
    if len(data) != HEADER_SIZE * HEADER_COPIES:
        raise ValueError('Truncated protected bootstrap')
    valid = []
    for start in range(0, len(data), HEADER_SIZE):
        try:
            header = parse_header(data[start:start + HEADER_SIZE])
        except ValueError:
            continue
        valid.append(header)
    if not valid:
        raise ValueError('NDV5 bootstrap: no recoverable header copy')
    identities = [replace(h, corrected_bytes=0, recovered_copies=1) for h in valid]
    if any(h != identities[0] for h in identities[1:]):
        raise ValueError('Conflicting valid bootstrap copies')
    return replace(min(valid, key=lambda h: h.corrected_bytes), recovered_copies=len(valid))


class MessageReader:
    """Decode groups into a caller-owned binary sink.

    Sink bytes are provisional until a Recovery is returned; discard on failure.
    Recovery contains metadata and aggregate repair counts, never payload bytes.
    """
    def __init__(self, sink):
        self.bootstrap_size = HEADER_SIZE * HEADER_COPIES
        self.frames = 0
        self.received = bytearray()
        self.confidence = None
        self.header = None
        self.result = None
        self.bytes_received = 0
        self.body_received = 0
        self.raw_received = 0
        self.sink = sink
        self.name = bytearray()
        self.checksum_bytes = bytearray()
        self.checksum = 0
        self.correction_total = self.correction_max = 0
        self.erasure_total = self.erasure_words = 0

    def _consume(self, raw):
        start = self.raw_received
        name_end = self.header.filename_length
        payload_end = name_end + self.header.payload_length
        self.name.extend(raw[:max(0, name_end - start)])
        payload = raw[max(0, name_end - start):max(0, payload_end - start)]
        if payload and self.sink.write(payload) != len(payload):
            raise OSError('Short payload write')
        self.checksum = zlib.crc32(raw[:max(0, payload_end - start)], self.checksum)
        self.checksum_bytes.extend(raw[max(0, payload_end - start):max(0, payload_end + CRC.size - start)])
        if any(raw[max(0, payload_end + CRC.size - start):]):
            raise ValueError('Nonzero Reed–Solomon data padding')
        self.raw_received += len(raw)

    def feed(self, data, final=False, *, confidence=None):
        if self.result is not None:
            return self.result
        if confidence is not None:
            if (len(confidence) != len(data)
                    or any(not math.isfinite(c) or c < 0 for c in confidence)):
                raise ValueError('Invalid extracted byte confidence')
        offset = 0
        while offset < len(data):
            target = (self.bootstrap_size if self.header is None else
                      min(BODY_SIZE * INTERLEAVE_DEPTH,
                          self.header.body_length - self.body_received))
            take = min(target - len(self.received), len(data) - offset)
            if confidence is not None and self.confidence is None:
                self.confidence = array('d', [float('nan')]) * len(self.received)
            self.received.extend(data[offset:offset + take])
            if self.confidence is not None:
                self.confidence.extend(confidence[offset:offset + take] if confidence is not None
                                       else [float('nan')] * take)
            offset += take
            self.bytes_received += take
            if len(self.received) < target:
                break
            if self.header is None:
                self.header = parse_bootstrap(bytes(self.received))
            else:
                scores = (self.confidence if self.confidence is not None
                          and all(math.isfinite(c) for c in self.confidence) else None)
                raw, corrections, erasures = decode_body_group(
                    bytes(self.received), scores,
                    word_offset=self.body_received // BODY_SIZE)
                self._consume(raw)
                self.body_received += target
                self.correction_total += sum(corrections)
                self.correction_max = max(self.correction_max, max(corrections))
                self.erasure_total += sum(erasures)
                self.erasure_words += sum(n > 0 for n in erasures)
            self.received.clear()
            self.confidence = None
            if self.header is not None and self.body_received == self.header.body_length:
                if self.checksum != CRC.unpack(self.checksum_bytes)[0]:
                    raise ValueError('Filename/payload CRC32 mismatch after error correction')
                name = self.name.decode('utf-8')
                filename_bytes(name)
                self.result = Recovery(
                    filename=name,
                    payload_length=self.header.payload_length,
                    header_corrections=self.header.corrected_bytes,
                    header_copies_recovered=self.header.recovered_copies,
                    body_correction_total=self.correction_total,
                    body_correction_max=self.correction_max,
                    body_erasure_total=self.erasure_total,
                    body_erasure_words=self.erasure_words)
                break
        if final and self.result is None:
            raise ValueError('Video ended before a complete payload was recovered')
        return self.result

    def feed_frame(self, plane, settings, *, confidence_model='minimum'):
        """Discard spare slots independently at the end of each header group."""
        from .video_stego import extract_chunk_with_confidence
        from .layout import frame_chunk
        if self.result is not None:
            return self.result
        chunk, confidence = extract_chunk_with_confidence(plane, settings, model=confidence_model)
        if self.header is None:
            plan = frame_chunk(plane.shape, self.bootstrap_size, self.frames, settings)
            chunk, confidence = chunk[:plan.length], confidence[:plan.length]
        self.frames += 1
        return self.feed(chunk, confidence=confidence)

    def finish(self):
        return self.feed(b'', final=True)
