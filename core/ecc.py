"""NDV5's fixed byte-oriented Reed-Solomon codes and interleaving order."""
from functools import lru_cache

import numpy as np

from reedsolo import RSCodec, ReedSolomonError

HEADER_DATA = 32
HEADER_SIZE = 96
BODY_DATA = 20
BODY_SIZE = 60
INTERLEAVE_DEPTH = 8
# Bound retries per failed codeword; retain at least eight parity constraints
# after accounting for erasures, rather than interpolating with 40 erasures.
MAX_BODY_ERASURES = 32


class ECCError(ValueError):
    """A protected block could not be recovered."""


class UncorrectableBodyError(ECCError):
    """No body repair was found; a different receiver ranking may help.

    Ambiguous repairs deliberately remain ECCError and must not trigger retry.
    """


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


def _repair_with_erasures(codec, word, confidence, number):
    ranked = np.argsort(confidence, kind='stable')
    candidate = None
    for count in range(1, MAX_BODY_ERASURES + 1):
        try:
            part, repaired, _ = codec.decode(word, erase_pos=ranked[:count].tolist())
        except ReedSolomonError:
            continue
        part, repaired = bytes(part), bytes(repaired)
        if candidate is not None and part != candidate[0]:
            raise ECCError(f'Ambiguous body Reed–Solomon repair at codeword {number}')
        if candidate is None:
            candidate = (part, sum(a != b for a, b in zip(word, repaired)), count)
    if candidate is None:
        raise UncorrectableBodyError(f'Body Reed–Solomon correction failed at codeword {number} '
                       '(including confidence-guided erasures)')
    return candidate


def decode_body(data, confidence=None, *, return_erasures=False):
    """Hard decode first; retry failed words with received-byte confidence.

    Confidence follows the interleaved wire bytes. All successful erasure trials
    must agree. Callers must still validate the complete message's integrity.
    """
    if not data or len(data) % BODY_SIZE:
        raise ECCError('Truncated protected body')
    blocks = interleave(data, reverse=True)
    scores = None
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64)
        if (confidence.shape != (len(data),) or not np.all(np.isfinite(confidence))
                or np.any(confidence < 0)):
            raise ValueError('Invalid body byte confidence')
        scores = np.empty_like(confidence)
        for start in range(0, len(data), INTERLEAVE_DEPTH * BODY_SIZE):
            group = confidence[start:start + INTERLEAVE_DEPTH * BODY_SIZE]
            scores[start:start + len(group)] = group.reshape(BODY_SIZE, -1).T.ravel()
    decoded = bytearray()
    corrections = []
    erasures = []
    codec = _codec(BODY_SIZE, BODY_DATA)
    for start in range(0, len(blocks), BODY_SIZE):
        word = blocks[start:start + BODY_SIZE]
        erased = 0
        try:
            part, _, corrected = codec.decode(word)
            changed = len(corrected)
        except ReedSolomonError as error:
            number = start // BODY_SIZE + 1
            if scores is None:
                raise ECCError(f'Body Reed–Solomon correction failed at codeword {number}') from error
            part, changed, erased = _repair_with_erasures(
                codec, word, scores[start:start + BODY_SIZE], number)
        decoded.extend(part)
        corrections.append(changed)
        erasures.append(erased)
    result = bytes(decoded), tuple(corrections)
    return (*result, tuple(erasures)) if return_erasures else result
