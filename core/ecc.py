"""NDV4's fixed byte-oriented Reed-Solomon codes and interleaving order."""
from functools import lru_cache

from reedsolo import RSCodec, ReedSolomonError

HEADER_DATA = 32
HEADER_SIZE = 96
BODY_DATA = 20
BODY_SIZE = 60
INTERLEAVE_DEPTH = 8


class ECCError(ValueError):
    """A protected block could not be recovered."""


@lru_cache(maxsize=2)
def _codec(n, k):
    # These parameters are part of the wire format, not library defaults.
    return RSCodec(n - k, nsize=n, c_exp=8, prim=0x11d, generator=2, fcr=0)


def encode_header(data):
    if len(data) != HEADER_DATA:
        raise ValueError('Bootstrap must contain exactly 32 data bytes')
    return bytes(_codec(HEADER_SIZE, HEADER_DATA).encode(data))


def decode_header(data):
    if len(data) != HEADER_SIZE:
        raise ECCError('Truncated protected bootstrap')
    try:
        decoded, _, corrected = _codec(HEADER_SIZE, HEADER_DATA).decode(data)
    except ReedSolomonError as error:
        raise ECCError('Bootstrap Reed–Solomon correction failed') from error
    return bytes(decoded), len(corrected)


def interleave(data, depth=INTERLEAVE_DEPTH, reverse=False):
    """Transpose groups of up to depth codewords; the last group is unpadded."""
    if depth != INTERLEAVE_DEPTH or len(data) % BODY_SIZE:
        raise ValueError('Invalid interleaving depth or codeword length')
    output = bytearray()
    for start in range(0, len(data), depth * BODY_SIZE):
        group = data[start:start + depth * BODY_SIZE]
        rows = len(group) // BODY_SIZE
        if reverse:
            output.extend(group[column * rows + row]
                          for row in range(rows) for column in range(BODY_SIZE))
        else:
            output.extend(group[row * BODY_SIZE + column]
                          for column in range(BODY_SIZE) for row in range(rows))
    return bytes(output)


def encode_body(data):
    blocks = bytearray()
    codec = _codec(BODY_SIZE, BODY_DATA)
    for start in range(0, len(data), BODY_DATA):
        blocks.extend(codec.encode(data[start:start + BODY_DATA].ljust(BODY_DATA, b'\0')))
    return interleave(blocks)


def decode_body(data):
    if not data or len(data) % BODY_SIZE:
        raise ECCError('Truncated protected body')
    blocks = interleave(data, reverse=True)
    decoded = bytearray()
    corrections = []
    codec = _codec(BODY_SIZE, BODY_DATA)
    for start in range(0, len(blocks), BODY_SIZE):
        try:
            part, _, corrected = codec.decode(blocks[start:start + BODY_SIZE])
        except ReedSolomonError as error:
            raise ECCError(f'Body Reed–Solomon correction failed at codeword {start // BODY_SIZE + 1}') from error
        decoded.extend(part)
        corrections.append(len(corrected))
    return bytes(decoded), tuple(corrections)
