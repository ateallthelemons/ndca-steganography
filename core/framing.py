"""NDV4 framing, integrity checks, exact sizes, and legacy NDV2/NDV3 parsing."""
from dataclasses import dataclass
import struct
import zlib

from . import ndca
from .profiles import LEGACY, get_profile
from .ecc import (
    BODY_DATA, BODY_SIZE, HEADER_SIZE, INTERLEAVE_DEPTH,
    encode_header, decode_header, encode_body, decode_body,
)

MAGIC = b'NDV4'
# magic, embedding/ECC/depth/flags, payload length, filename length,
# encoded body length, reserved bytes, CRC32 of the preceding 28 bytes.
BOOTSTRAP = struct.Struct('>4sBBBBQHQ2sI')
CRC = struct.Struct('>I')
LEGACY_HEADERS = {b'NDV2': struct.Struct('>4sQI'), b'NDV3': struct.Struct('>4sQIH')}
EMBEDDING_PROFILE = 1
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
    profile_id: int = 1

    @property
    def total_length(self):
        return HEADER_SIZE + self.body_length


@dataclass(frozen=True)
class Recovery:
    filename: str | None
    payload: bytes
    version: str
    header_corrections: int = 0
    body_corrections: tuple = ()
    profile_id: int = 1


def require_embedding_profile():
    # The bootstrap and body both use this frozen profile. Future settings need
    # a new profile implementation; changing globals must not silently emit NDV4.
    if (ndca.BLOCK_SIZE, ndca.DELTA, ndca.WAVELET) != (16, 40, 'haar'):
        raise ValueError('NDV4 profile 1 requires 16x16 blocks, DELTA=40, and Haar')


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


def encoded_size(payload_length, name_length):
    _validate_lengths(payload_length, name_length)
    raw_length = name_length + payload_length + CRC.size
    return HEADER_SIZE + BODY_SIZE * ((raw_length + BODY_DATA - 1) // BODY_DATA)


def maximum_payload(capacity, name_length):
    _validate_lengths(0, name_length)
    if capacity < encoded_size(0, name_length):
        return -1
    return min(MAX_PAYLOAD_BYTES,
               ((capacity - HEADER_SIZE) // BODY_SIZE) * BODY_DATA - name_length - CRC.size)


def pack_message(name, payload, profile=LEGACY):
    require_embedding_profile()
    profile = get_profile(profile)
    name_data = filename_bytes(name)
    body_length = encoded_size(len(payload), len(name_data)) - HEADER_SIZE
    raw_header = BOOTSTRAP.pack(MAGIC, profile.id, ECC_PROFILE,
                               INTERLEAVE_DEPTH, 0, len(payload), len(name_data),
                               body_length, b'\0\0', 0)
    raw_header = raw_header[:-CRC.size] + CRC.pack(zlib.crc32(raw_header[:-CRC.size]))
    body = name_data + payload
    body += CRC.pack(zlib.crc32(body))
    return encode_header(raw_header) + encode_body(body)


def parse_header(data):
    require_embedding_profile()
    raw, corrected = decode_header(data)
    magic, embedding, ecc, depth, flags, length, name_length, body_length, reserved, crc = BOOTSTRAP.unpack(raw)
    if zlib.crc32(raw[:-CRC.size]) != crc:
        raise ValueError('Bootstrap CRC32 mismatch after error correction')
    if magic != MAGIC:
        raise ValueError('NDV4 bootstrap magic not found')
    get_profile(embedding)
    if (ecc, depth, flags, reserved) != (
            ECC_PROFILE, INTERLEAVE_DEPTH, 0, b'\0\0'):
        raise ValueError('Unsupported NDV4 profile, flags, or reserved fields')
    expected = encoded_size(length, name_length) - HEADER_SIZE
    if body_length != expected:
        raise ValueError('Inconsistent protected body length')
    return Header(length, name_length, body_length, corrected, embedding)


def unpack_body(header, encoded):
    if len(encoded) != header.body_length:
        raise ValueError('Truncated or oversized protected body')
    raw, corrections = decode_body(encoded)
    end = header.filename_length + header.payload_length
    if any(raw[end + CRC.size:]):
        raise ValueError('Nonzero Reed–Solomon data padding')
    if zlib.crc32(raw[:end]) != CRC.unpack(raw[end:end + CRC.size])[0]:
        raise ValueError('Filename/payload CRC32 mismatch after error correction')
    name = raw[:header.filename_length].decode('utf-8')
    filename_bytes(name)
    return Recovery(name, raw[header.filename_length:end], 'NDV4',
                    header.corrected_bytes, corrections, header.profile_id)


class MessageReader:
    """Incremental frame-byte reader; try repaired NDV4 before legacy magic.

    Legacy messages shorter than the bootstrap are parsed at EOF. No allocation
    is based solely on a header length, and all declared lengths are bounded.
    """
    def __init__(self):
        self.received = bytearray()
        self.header = None
        self.legacy = False
        self.result = None

    def feed(self, data, final=False):
        if self.result is not None:
            return self.result
        self.received.extend(data)
        if self.header is None and not self.legacy:
            if len(self.received) < HEADER_SIZE and not final:
                return None
            try:
                self.header = parse_header(bytes(self.received[:HEADER_SIZE]))
            except ValueError:
                if bytes(self.received[:4]) not in LEGACY_HEADERS:
                    raise
                self.legacy = True
        if self.legacy:
            self.result = self._read_legacy()
        elif len(self.received) >= self.header.total_length:
            self.result = unpack_body(self.header, bytes(self.received[HEADER_SIZE:self.header.total_length]))
        if final and self.result is None:
            raise ValueError('Video ended before a complete payload was recovered')
        return self.result

    def feed_frame(self, plane):
        """Read the fixed bootstrap, then switch profiles at the next frame.

        Profile 1 retains its original continuous header/body packing. New
        profiles discard only the last bootstrap frame's unused byte slots.
        """
        from .video_stego import extract_chunk
        if self.result is not None:
            return self.result
        if self.header is None and not self.legacy:
            chunk = extract_chunk(plane)
            remaining = HEADER_SIZE - len(self.received)
            if len(chunk) >= remaining:
                result = self.feed(chunk[:remaining])
                if self.header is not None and self.header.profile_id != 1:
                    return result
                return self.feed(chunk[remaining:])
            return self.feed(chunk)
        profile = get_profile(self.header.profile_id) if self.header is not None else LEGACY
        return self.feed(extract_chunk(plane, profile))

    def _read_legacy(self):
        magic = bytes(self.received[:4])
        layout = LEGACY_HEADERS[magic]
        if len(self.received) < layout.size:
            return None
        values = layout.unpack(self.received[:layout.size])
        length, expected_crc = values[1:3]
        name_length = values[3] if magic == b'NDV3' else 0
        _validate_lengths(length, name_length if magic == b'NDV3' else 1)
        start = layout.size + name_length
        if len(self.received) < start + length:
            return None
        name = None
        if name_length:
            name = bytes(self.received[layout.size:start]).decode('utf-8')
            filename_bytes(name)
        payload = bytes(self.received[start:start + length])
        if zlib.crc32(payload) != expected_crc:
            raise ValueError('Payload corrupted (CRC32 mismatch)')
        return Recovery(name, payload, magic.decode('ascii'))
