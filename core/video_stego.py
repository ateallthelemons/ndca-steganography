import numpy as np
from .profiles import LEGACY, get_profile
from . import stc

from .ndca import (
    BLOCK_SIZE, DELTA, EmbeddingError, get_positions, get_coefficient,
    modify_coefficient, decode_bit, calculate_costs, stc_encode, stc_decode,
    image_blocks, haar_dwt2, haar_idwt2, get_coefficients, decode_bits
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


# Target indices i (target i * delta) per carrier: at most five fit in +-2 delta.
TARGET_SLOTS = 5


def reconstruct_carriers(blocks, coefficients, delta):
    """Batched reconstruct_carrier for both bits of every block.

    Returns pixels (n, 2, BLOCK_SIZE, BLOCK_SIZE), movement (n, 2) and a
    boolean found (n, 2); entries with found False correspond to None. Every
    target is tried at once, then each bit keeps the same candidate the
    sequential search would: the nearest valid target, ties to the smaller one.
    """
    n = len(blocks)
    LL, details = haar_dwt2(blocks)
    U, S, Vt = np.linalg.svd(LL, full_matrices=False)

    low = np.maximum(0, np.ceil((coefficients - 2 * delta) / delta)).astype(np.int64)
    high = np.floor((coefficients + 2 * delta) / delta).astype(np.int64)
    index = low[:, None] + np.arange(TARGET_SLOTS)
    target = (index * delta).astype(np.float64)
    in_range = index <= high[:, None]

    modified = np.repeat(S[:, None, :], TARGET_SLOTS, axis=1)
    modified[:, :, 0] = target
    new_LL = (U[:, None] * modified[:, :, None, :]) @ Vt[:, None]
    expanded = tuple(np.repeat(d[:, None], TARGET_SLOTS, axis=1) for d in details)
    pixels = np.clip(np.rint(haar_idwt2(new_LL, expanded)), 0, 255).astype(np.uint8)

    as_float = pixels.astype(np.float64)
    within = np.max(np.abs(as_float - blocks[:, None]), axis=(2, 3)) <= delta
    survives = decode_bits(get_coefficients(as_float), delta) == index % 2
    usable = in_range & within & survives
    distance = np.abs(target - coefficients[:, None])

    rows = np.arange(n)
    chosen_pixels = np.zeros((n, 2, BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8)
    movement = np.full((n, 2), np.inf)
    found = np.zeros((n, 2), dtype=bool)
    for bit in (0, 1):
        key = np.where(usable & (index % 2 == bit), distance, np.inf)
        best = np.argmin(key, axis=1)  # slots ascend by target, so ties go low
        found[:, bit] = np.isfinite(key[rows, best])
        chosen_pixels[:, bit] = pixels[rows, best]
        movement[:, bit] = distance[rows, best]
    return chosen_pixels, movement, found


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

    coefficients, factors = calculate_costs(image, profile.delta, profile.cost, return_factor=True)
    coefficients = coefficients.ravel()[positions]
    factors = factors.ravel()[positions]

    cover = decode_bits(coefficients, profile.delta)
    blocks = image_blocks(image).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)[positions]
    candidates, movement, found = reconstruct_carriers(blocks, coefficients, profile.delta)
    keep = np.arange(2) == cover[:, None]
    costs = np.where(keep, movement * factors[:, None], movement / factors[:, None])
    costs[~found] = np.inf

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
    chosen = candidates[np.arange(len(positions)), stego_bits.astype(np.intp)]
    rows, cols = divmod(positions.astype(np.intp), image.shape[1] // BLOCK_SIZE)
    offsets = np.arange(BLOCK_SIZE)
    y = (rows * BLOCK_SIZE)[:, None, None] + offsets[:, None]
    x = (cols * BLOCK_SIZE)[:, None, None] + offsets
    result[y, x] = chosen

    if extract_chunk(result, profile) != data:
        raise EmbeddingError('Chunk failed recovery after pixel reconstruction')
    return result, cost


def extract_chunk(U, profile=LEGACY):
    profile = get_profile(profile)
    size = chunk_size(U.shape, profile)
    if not size:
        raise ValueError('U plane too small for embedding')

    image = U.astype(np.float64)
    positions = get_positions(image.shape, size * 8, None if profile.stc == 'legacy' else profile.width)
    blocks = image_blocks(image).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)[positions]
    carrier = decode_bits(get_coefficients(blocks), profile.delta)
    bits = stc_decode(carrier) if profile.stc == 'legacy' else stc.extract(carrier, profile.width)
    return np.packbits(bits).tobytes()
