"""NDV5 framing, integrity, exact sizes, and incremental recovery."""
from dataclasses import dataclass, replace
from array import array
import math
import struct
import zlib

from .config import MAGIC, EMBEDDING_ID, HEADER_COPIES
from .ecc import (
    BODY_DATA, BODY_SIZE, HEADER_SIZE, INTERLEAVE_DEPTH,
    encode_header, decode_header, encode_body, decode_body,
)

# magic, embedding/ECC/depth/flags, payload length, filename length,
# encoded body length, reserved bytes, CRC32 of the preceding 28 bytes.
BOOTSTRAP = struct.Struct('>4sBBBBQHQ2sI')
CRC = struct.Struct('>I')
ECC_PROFILE = 1
MAX_FILENAME_BYTES = 65535
# Bound memory use before trusting metadata. This implementation buffers files.
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024


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
    payload: bytes
    header_corrections: int = 0
    body_corrections: tuple = ()
    header_copies_recovered: int = 1
    body_erasures: tuple = ()


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
    return BODY_SIZE * ((raw_length + BODY_DATA - 1) // BODY_DATA)


def encoded_size(payload_length, name_length):
    return HEADER_SIZE * HEADER_COPIES + encoded_body_size(payload_length, name_length)


def maximum_payload(capacity, name_length):
    _validate_lengths(0, name_length)
    if capacity < encoded_size(0, name_length):
        return -1
    return min(MAX_PAYLOAD_BYTES,
               ((capacity - HEADER_SIZE * HEADER_COPIES) // BODY_SIZE) * BODY_DATA - name_length - CRC.size)


def pack_message(name, payload):
    name_data = filename_bytes(name)
    body_length = encoded_body_size(len(payload), len(name_data))
    raw_header = BOOTSTRAP.pack(MAGIC, EMBEDDING_ID, ECC_PROFILE,
                               INTERLEAVE_DEPTH, 0, len(payload), len(name_data),
                               body_length, b'\0\0', 0)
    raw_header = raw_header[:-CRC.size] + CRC.pack(zlib.crc32(raw_header[:-CRC.size]))
    body = name_data + payload
    body += CRC.pack(zlib.crc32(body))
    return encode_header(raw_header) * HEADER_COPIES + encode_body(body)


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


def unpack_body(header, encoded, confidence=None):
    if len(encoded) != header.body_length:
        raise ValueError('Truncated or oversized protected body')
    raw, corrections, erasures = decode_body(encoded, confidence, return_erasures=True)
    end = header.filename_length + header.payload_length
    if any(raw[end + CRC.size:]):
        raise ValueError('Nonzero Reed–Solomon data padding')
    if zlib.crc32(raw[:end]) != CRC.unpack(raw[end:end + CRC.size])[0]:
        raise ValueError('Filename/payload CRC32 mismatch after error correction')
    name = raw[:header.filename_length].decode('utf-8')
    filename_bytes(name)
    return Recovery(name, raw[header.filename_length:end],
                    header.corrected_bytes, corrections,
                    header.recovered_copies, erasures)


class MessageReader:
    """Incremental NDV5 reader. All declared lengths are bounded."""
    def __init__(self):
        self.bootstrap_size = HEADER_SIZE * HEADER_COPIES
        self.frames = 0
        self.received = bytearray()
        self.confidence = None
        self.header = None
        self.result = None

    def feed(self, data, final=False, *, confidence=None):
        if self.result is not None:
            return self.result
        if confidence is not None:
            confidence = array('d', confidence)
            if (len(confidence) != len(data)
                    or any(not math.isfinite(c) or c < 0 for c in confidence)):
                raise ValueError('Invalid extracted byte confidence')
            if self.confidence is None:
                self.confidence = array('d', [float('nan')]) * len(self.received)
        if self.confidence is not None:
            self.confidence.extend(confidence if confidence is not None
                                   else array('d', [float('nan')]) * len(data))
        self.received.extend(data)
        if self.header is None:
            if len(self.received) < self.bootstrap_size and not final:
                return None
            self.header = parse_bootstrap(bytes(self.received[:self.bootstrap_size]))
        if len(self.received) >= self.header.total_length:
            scores = None
            if self.confidence is not None:
                scores = self.confidence[self.bootstrap_size:self.header.total_length]
                if any(not math.isfinite(c) for c in scores):
                    scores = None  # Byte-only callers retain hard-decision behavior.
            self.result = unpack_body(
                self.header, bytes(self.received[self.bootstrap_size:self.header.total_length]), scores)
        if final and self.result is None:
            raise ValueError('Video ended before a complete payload was recovered')
        return self.result

    def feed_frame(self, plane):
        """Discard spare slots independently at the end of each header group."""
        from .video_stego import extract_chunk_with_confidence
        from .layout import frame_chunk
        if self.result is not None:
            return self.result
        chunk, confidence = extract_chunk_with_confidence(plane)
        if self.header is None:
            plan = frame_chunk(plane.shape, self.bootstrap_size, self.frames)
            chunk, confidence = chunk[:plan.length], confidence[:plan.length]
        self.frames += 1
        return self.feed(chunk, confidence=confidence)

    def finish(self):
        return self.feed(b'', final=True)
