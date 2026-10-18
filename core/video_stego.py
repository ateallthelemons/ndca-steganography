import numpy as np
from .config import BLOCK_SIZE
from . import stc
from .ndca import (
    EmbeddingError, get_positions, calculate_costs,
    image_blocks, haar_dwt2, haar_idwt2, get_coefficients, decode_bits,
)


# Target indices i (target i * delta) per carrier: at most five fit in +-2 delta.
TARGET_SLOTS = 5


def reconstruct_carriers(blocks, coefficients, delta):
    """Find bounded, recoverable pixel candidates for both bits of every block.

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


def chunk_size(shape, settings):
    height, width = shape
    blocks = (height // BLOCK_SIZE) * (width // BLOCK_SIZE)
    return max(0, (blocks - 1) // (8 * settings.width))


def embed_chunk(U, data, settings):
    if len(data) != chunk_size(U.shape, settings) or not data:
        raise ValueError('Invalid chunk size for this frame')

    image = U.astype(np.float64)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    positions = get_positions(image.shape, len(bits), settings.width)

    coefficients, factors = calculate_costs(image, settings.cost)
    coefficients = coefficients.ravel()[positions]
    factors = factors.ravel()[positions]

    cover = decode_bits(coefficients, settings.delta)
    blocks = image_blocks(image).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)[positions]
    candidates, movement, found = reconstruct_carriers(blocks, coefficients, settings.delta)
    keep = np.arange(2) == cover[:, None]
    costs = np.where(keep, movement * factors[:, None], movement / factors[:, None])
    costs[~found] = np.inf

    stego_bits, cost = stc.embed(bits, costs, settings.width)
    recovered = stc.extract(stego_bits, settings.width)
    if not np.array_equal(recovered, bits):
        raise EmbeddingError('Trellis failed to reproduce the requested message')

    result = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    chosen = candidates[np.arange(len(positions)), stego_bits.astype(np.intp)]
    rows, cols = divmod(positions.astype(np.intp), image.shape[1] // BLOCK_SIZE)
    offsets = np.arange(BLOCK_SIZE)
    y = (rows * BLOCK_SIZE)[:, None, None] + offsets[:, None]
    x = (cols * BLOCK_SIZE)[:, None, None] + offsets
    result[y, x] = chosen

    if extract_chunk(result, settings) != data:
        raise EmbeddingError('Chunk failed recovery after pixel reconstruction')
    return result, cost


def extract_chunk(U, settings):
    return _extract_chunk(U, settings, with_confidence=False)


def extract_chunk_with_confidence(U, settings, *, model='minimum'):
    """Return unchanged bytes and receiver-only byte reliability scores."""
    if model not in ('minimum', 'likelihood'):
        raise ValueError('Unknown confidence model')
    return _extract_chunk(U, settings, with_confidence=True, model=model)


def likelihood_byte_confidence(coefficients, settings):
    """Approximate byte reliability under a Gaussian quantization-lattice model.

    Use a fixed sigma=delta/2, a uniform lattice prior and nine nearby centers.
    This is a ranking heuristic, not a calibrated channel probability: codec
    noise is correlated and neighboring STC bits share supporting carriers.
    It uses only received coefficients, never payloads or original frames.
    """
    values = np.asarray(coefficients, dtype=np.float64) / settings.delta
    if (values.ndim != 1 or len(values) % (8 * settings.width)
            or not np.all(np.isfinite(values))):
        raise ValueError('Invalid carrier coefficients')
    residual = values - np.rint(values)
    offsets = np.arange(-4, 5)
    weights = np.exp(-2 * (residual[:, None] - offsets) ** 2)
    # Even offsets agree with the hard-decision parity; odd offsets disagree.
    reliability = np.clip(
        (weights * np.where(offsets % 2, -1, 1)).sum(axis=1) / weights.sum(axis=1),
        0, 1)
    length = len(values) // settings.width
    bit_reliability = np.ones(length)
    for column, mask in enumerate(stc.columns(settings.width)):
        for offset in range(min(stc.HEIGHT, length)):
            if (mask >> offset) & 1:
                bit_reliability[offset:] *= reliability[column::settings.width][:length-offset]
    # For independent parity errors, P(correct XOR)=(1+product(1-2p))/2.
    return ((1 + bit_reliability) / 2).reshape(-1, 8).prod(axis=1)


def _extract_chunk(U, settings, *, with_confidence, model='minimum'):
    size = chunk_size(U.shape, settings)
    if not size:
        raise ValueError('U plane too small for embedding')

    image = U.astype(np.float64)
    positions = get_positions(image.shape, size * 8, settings.width)
    blocks = image_blocks(image).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)[positions]
    coefficients = get_coefficients(blocks)
    carrier = decode_bits(coefficients, settings.delta)
    if with_confidence:
        values = coefficients / settings.delta
        margins = settings.delta * (0.5 - np.abs(values - np.rint(values)))
    bits = stc.extract(carrier, settings.width)
    data = np.packbits(bits).tobytes()
    if not with_confidence:
        return data
    if model == 'likelihood':
        return data, likelihood_byte_confidence(coefficients, settings)
    margins = np.asarray(margins)
    confidence = np.full(len(bits), np.inf)
    for column, mask in enumerate(stc.columns(settings.width)):
        for offset in range(min(stc.HEIGHT, len(bits))):
            if (mask >> offset) & 1:
                confidence[offset:] = np.minimum(
                    confidence[offset:],
                    margins[column::settings.width][:len(bits)-offset])
    return data, confidence.reshape(-1, 8).min(axis=1)
