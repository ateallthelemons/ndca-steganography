
import numpy as np
import pywt

# CONFIG
BLOCK_SIZE = 16
DELTA = 40
WAVELET = "haar"
MESSAGE_BYTES = 3


class EmbeddingError(ValueError):
    """The requested data could not be embedded and verified reliably."""


def get_positions(shape, message_bits, carriers_per_bit=None):
    height, width = shape
    rows = height // BLOCK_SIZE
    cols = width // BLOCK_SIZE

    count = rows * cols
    required = 3 * message_bits + 2 if carriers_per_bit is None else carriers_per_bit * message_bits

    if count - 1 < required:
        raise ValueError(f"Image too small: need {required + 1} blocks, have {count}")

    return np.linspace(1, count - 1, required, dtype=int)



def decode_bit(value, delta=DELTA):
    return int(np.rint(value / delta)) % 2

def encode_bit(value, bit, delta=DELTA):
    index = int(np.rint(value / delta))

    if index % 2 != bit:
        index += 1 if value >= index * delta else -1

    return index * delta

def get_coefficient(block):
    LL, _ = pywt.dwt2(block, WAVELET)
    S = np.linalg.svd(LL, compute_uv=False)
    return S[0]

def modify_coefficient(block, value):
    LL, details = pywt.dwt2(block, WAVELET)
    U, S, Vt = np.linalg.svd(LL, full_matrices=False)

    S[0] = value
    modified_LL = (U * S) @ Vt

    return pywt.idwt2((modified_LL, details), WAVELET)

# Batched Haar/SVD helpers. Each repeats pywt's and LAPACK's per-block arithmetic
# in the same order, so results are bit-identical to the single-block functions
# above. Inputs are stacks of blocks with shape (..., BLOCK_SIZE, BLOCK_SIZE).
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
    """Batched get_coefficient."""
    LL, _ = haar_dwt2(blocks)
    return np.linalg.svd(LL, compute_uv=False)[..., 0]


def decode_bits(values, delta=DELTA):
    """Batched decode_bit."""
    return (np.rint(np.asarray(values) / delta).astype(np.int64) % 2).astype(np.uint8)


def cost_factor(normalized, model):
    if model == 'crf':
        return 1 - normalized + 1e-6
    if model == 'cqp':
        return normalized + 1e-6
    raise ValueError('Unknown cost model')


def calculate_costs(image, delta=DELTA, model='crf', return_factor=False):
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

    nearest = np.rint(coefficients / delta) * delta
    small_move = np.abs(coefficients - nearest)
    large_move = delta - small_move

    factor = cost_factor(normalized, model)

    keep_cost = small_move * factor
    flip_cost = large_move / factor

    if return_factor:
        return coefficients, factor
    return coefficients, keep_cost, flip_cost

# educational four-state syndrome-trellis code
def stc_encode(cover, message, keep_costs, flip_costs):
    k = len(message)
    n = len(cover)

    if n != 3*k + 2:
        raise ValueError('Invalid carrier count for the message')

    dp = np.full((k+1, 4), np.inf)
    previous = np.full((k+1, 4), -1, dtype=int)
    choices = np.full((k+1, 4, 3), -1, dtype=int)

    def cost(i, bit):
        return keep_costs[i] if bit == cover[i] else flip_costs[i]

    for a in [0, 1]:
        for b in [0, 1]:
            state = 2*a + b
            dp[0, state] = cost(0, a) + cost(1, b)

    for i in range(k):
        start = 3*i + 2

        for state in range(4):
            a, b = divmod(state, 2)

            for c in [0, 1]:
                for d in [0, 1]:
                    for e in [0, 1]:
                        if (a ^ b ^ c ^ d ^ e) != message[i]:
                            continue

                        new_state = 2*d + e
                        total = dp[i, state]
                        total += cost(start, c)
                        total += cost(start+1, d)
                        total += cost(start+2, e)

                        if total < dp[i+1, new_state]:
                            dp[i+1, new_state] = total
                            previous[i+1, new_state] = state
                            choices[i+1, new_state] = [c, d, e]

    state = int(np.argmin(dp[k]))
    minimum_cost = dp[k, state]
    if not np.isfinite(minimum_cost):
        raise EmbeddingError('No feasible trellis path for this chunk')
    result = np.zeros(n, dtype=np.uint8)

    for i in range(k, 0, -1):
        result[3*(i-1)+2:3*(i-1)+5] = choices[i, state]
        state = previous[i, state]

    result[:2] = divmod(state, 2)

    return result, minimum_cost

def stc_decode(carrier):
    k = (len(carrier) - 2) // 3
    message = np.zeros(k, dtype=np.uint8)

    for i in range(k):
        message[i] = np.bitwise_xor.reduce(carrier[3*i:3*i+5])

    return message
