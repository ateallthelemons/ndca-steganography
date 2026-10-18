import numpy as np
from .config import BLOCK_SIZE


class EmbeddingError(ValueError):
    """The requested data could not be embedded and verified reliably."""


def get_positions(shape, message_bits, carriers_per_bit):
    height, width = shape
    rows = height // BLOCK_SIZE
    cols = width // BLOCK_SIZE

    count = rows * cols
    required = carriers_per_bit * message_bits

    if count - 1 < required:
        raise ValueError(f"Image too small: need {required + 1} blocks, have {count}")

    return np.linspace(1, count - 1, required, dtype=int)

HAAR = np.sqrt(0.5)


def image_blocks(image):
    """View an image as a (rows, cols, BLOCK_SIZE, BLOCK_SIZE) grid of blocks."""
    rows = image.shape[0] // BLOCK_SIZE
    cols = image.shape[1] // BLOCK_SIZE
    grid = image[:rows * BLOCK_SIZE, :cols * BLOCK_SIZE]
    return grid.reshape(rows, BLOCK_SIZE, cols, BLOCK_SIZE).swapaxes(1, 2)


def _analysis(x, axis):
    even = np.take(x, np.arange(0, x.shape[axis], 2), axis)
    odd = np.take(x, np.arange(1, x.shape[axis], 2), axis)
    return even * HAAR + odd * HAAR, even * HAAR - odd * HAAR


def _synthesis(approx, detail, axis):
    shape = list(approx.shape)
    shape[axis] *= 2
    out = np.empty(shape)
    even = [slice(None)] * approx.ndim
    odd = list(even)
    even[axis] = slice(0, None, 2)
    odd[axis] = slice(1, None, 2)
    out[tuple(even)] = approx * HAAR + detail * HAAR
    out[tuple(odd)] = approx * HAAR + detail * -HAAR
    return out


def haar_dwt2(blocks):
    """Batched pywt.dwt2(block, 'haar')."""
    low, high = _analysis(blocks, -2)
    LL, cV = _analysis(low, -1)
    cH, cD = _analysis(high, -1)
    return LL, (cH, cV, cD)


def haar_idwt2(LL, details):
    """Batched pywt.idwt2((LL, details), 'haar')."""
    cH, cV, cD = details
    return _synthesis(_synthesis(LL, cV, -1), _synthesis(cH, cD, -1), -2)


def get_coefficients(blocks):
    """Largest LL singular value of each block."""
    LL, _ = haar_dwt2(blocks)
    return np.linalg.svd(LL, compute_uv=False)[..., 0]


def decode_bits(values, delta):
    """Extract quantization parity for a batch of coefficients."""
    return (np.rint(np.asarray(values) / delta).astype(np.int64) % 2).astype(np.uint8)


def cost_factor(normalized, model):
    if model == 'crf':
        return 1 - normalized + 1e-6
    if model == 'cqp':
        return normalized + 1e-6
    raise ValueError('Unknown cost model')


def calculate_costs(image, model):
    """Return LL coefficients and normalized neighbor-cost factors."""
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape
    rows = height // BLOCK_SIZE
    cols = width // BLOCK_SIZE

    blocks = image_blocks(image)
    coefficients = get_coefficients(blocks)

    upper = np.mean(np.abs(blocks[1:] - blocks[:-1]), axis=(2, 3))
    left = np.mean(np.abs(blocks[:, 1:] - blocks[:, :-1]), axis=(2, 3))
    differences = np.full((rows, cols), np.nan)
    differences[1:, 0] = upper[:, 0]
    differences[0, 1:] = left[0, :]
    differences[1:, 1:] = (upper[:, 1:] + left[1:, :]) / 2

    valid = np.isfinite(differences)
    minimum = np.min(differences[valid])
    maximum = np.max(differences[valid])

    normalized = (differences - minimum) / (maximum - minimum + 1e-6)

    return coefficients, cost_factor(normalized, model)
