import numpy as np
from .profiles import LEGACY, get_profile
from . import stc

from .ndca import (
    BLOCK_SIZE, DELTA, EmbeddingError, get_positions, get_coefficient,
    modify_coefficient, decode_bit, calculate_costs, stc_encode, stc_decode
)


# Encoder-only search limits; extraction and the on-disk format are unchanged.
MAX_COEFFICIENT_SHIFT = 2 * DELTA
MAX_PIXEL_CHANGE = DELTA


def reconstruct_carrier(block, coefficient, bit, profile=LEGACY):
    """Find the nearest bounded target whose bit survives pixel reconstruction.

    Every attempt starts from the original block. Return None when this bit is
    unavailable within the search/distortion limits, so the trellis can avoid it.
    """
    profile = get_profile(profile)
    delta = profile.delta
    low = max(0, int(np.ceil((coefficient - 2 * delta) / delta)))
    high = int(np.floor((coefficient + 2 * delta) / delta))
    targets = [i * delta for i in range(low, high + 1) if i % 2 == bit]
    targets.sort(key=lambda target: (abs(target - coefficient), target))
    for target in targets:
        pixels = np.clip(np.rint(modify_coefficient(block, target)), 0, 255).astype(np.uint8)
        if np.max(np.abs(pixels.astype(np.float64) - block)) > delta:
            continue
        if decode_bit(get_coefficient(pixels.astype(np.float64)), delta) == bit:
            return pixels, abs(target - coefficient)
    return None


def chunk_size(shape, profile=LEGACY):
    profile = get_profile(profile)
    height, width = shape
    blocks = (height // BLOCK_SIZE) * (width // BLOCK_SIZE)
    if profile.stc == 'legacy':
        return max(0, ((blocks - 3) // 3) // 8)
    return max(0, (blocks - 1) // (8 * profile.width))


def embed_chunk(U, data, profile=LEGACY):
    profile = get_profile(profile)
    if len(data) != chunk_size(U.shape, profile) or not data:
        raise ValueError('Invalid chunk size for this frame')

    image = U.astype(np.float64)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    positions = get_positions(image.shape, len(bits), None if profile.stc == 'legacy' else profile.width)
    cols = image.shape[1] // BLOCK_SIZE

    coefficients, factors = calculate_costs(image, profile.delta, profile.cost, return_factor=True)
    coefficients = coefficients.ravel()[positions]
    factors = factors.ravel()[positions]

    cover = np.array([decode_bit(c, profile.delta) for c in coefficients], dtype=np.uint8)
    costs = np.full((len(positions), 2), np.inf)
    candidates = []
    for i, position in enumerate(positions):
        r, c = divmod(int(position), cols)
        y, x = r * BLOCK_SIZE, c * BLOCK_SIZE
        block = image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
        options = []
        for bit in (0, 1):
            candidate = reconstruct_carrier(block, coefficients[i], bit, profile)
            options.append(None if candidate is None else candidate[0])
            if candidate is not None:
                movement = candidate[1]
                costs[i, bit] = (movement * factors[i] if bit == cover[i]
                                else movement / factors[i])
        candidates.append(options)

    if profile.stc == 'legacy':
        indices = np.arange(len(positions))
        stego_bits, cost = stc_encode(cover, bits, costs[indices, cover], costs[indices, 1-cover])
        recovered = stc_decode(stego_bits)
    else:
        stego_bits, cost = stc.embed(bits, costs, profile.width)
        recovered = stc.extract(stego_bits, profile.width)
    if not np.array_equal(recovered, bits):
        raise EmbeddingError('Trellis failed to reproduce the requested message')

    result = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    for i, position in enumerate(positions):
        r, c = divmod(int(position), cols)
        y, x = r * BLOCK_SIZE, c * BLOCK_SIZE
        result[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = candidates[i][int(stego_bits[i])]

    if extract_chunk(result, profile) != data:
        raise EmbeddingError('Chunk failed recovery after pixel reconstruction')
    return result, cost


def extract_chunk(U, profile=LEGACY):
    return _extract_chunk(U, profile, with_confidence=False)


def extract_chunk_with_confidence(U, profile=LEGACY):
    """Return bytes and per-byte minimum quantizer margin (not probability)."""
    return _extract_chunk(U, profile, with_confidence=True)


def _extract_chunk(U, profile, *, with_confidence):
    profile = get_profile(profile)
    size = chunk_size(U.shape, profile)
    if not size:
        raise ValueError('U plane too small for embedding')

    image = U.astype(np.float64)
    positions = get_positions(image.shape, size * 8, None if profile.stc == 'legacy' else profile.width)
    cols = image.shape[1] // BLOCK_SIZE
    carrier = []
    margins = []

    for position in positions:
        r, c = divmod(int(position), cols)
        y, x = r * BLOCK_SIZE, c * BLOCK_SIZE
        coefficient = get_coefficient(image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE])
        carrier.append(decode_bit(coefficient, profile.delta))
        if with_confidence:
            value = coefficient / profile.delta
            margins.append(profile.delta * (0.5 - abs(value - np.rint(value))))

    carrier = np.asarray(carrier, dtype=np.uint8)
    bits = stc_decode(carrier) if profile.stc == 'legacy' else stc.extract(carrier, profile.width)
    data = np.packbits(bits).tobytes()
    if not with_confidence:
        return data
    margins = np.asarray(margins)
    if profile.stc == 'legacy':
        confidence = np.array([margins[3*i:3*i+5].min() for i in range(len(bits))])
    else:
        confidence = np.full(len(bits), np.inf)
        for column, mask in enumerate(stc.columns(profile.width)):
            for offset in range(min(stc.HEIGHT, len(bits))):
                if (mask >> offset) & 1:
                    confidence[offset:] = np.minimum(
                        confidence[offset:],
                        margins[column::profile.width][:len(bits)-offset])
    return data, confidence.reshape(-1, 8).min(axis=1)
