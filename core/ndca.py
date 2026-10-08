
import numpy as np
import pywt

# CONFIG
BLOCK_SIZE = 16
DELTA = 40
WAVELET = "haar"
MESSAGE_BYTES = 3


def get_positions(shape, message_bits):
    height, width = shape
    rows = height // BLOCK_SIZE
    cols = width // BLOCK_SIZE

    count = rows * cols
    required = 3 * message_bits + 2

    if count - 1 < required:
        raise ValueError(f"Image too small: need {required + 1} blocks, have {count}")

    return np.linspace(1, count - 1, required, dtype=int)



def decode_bit(value):
    return int(np.rint(value / DELTA)) % 2

def encode_bit(value, bit):
    index = int(np.rint(value / DELTA))

    if index % 2 != bit:
        index += 1 if value >= index * DELTA else -1

    return index * DELTA

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

def calculate_costs(image):
    height, width = image.shape
    rows = height // BLOCK_SIZE
    cols = width // BLOCK_SIZE

    coefficients = np.zeros((rows, cols))
    differences = np.full((rows, cols), np.nan)

    for r in range(rows):
        for c in range(cols):
            y = r * BLOCK_SIZE
            x = c * BLOCK_SIZE
            block = image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            coefficients[r, c] = get_coefficient(block)
            neighbors = []

            if r > 0:
                upper = image[y-BLOCK_SIZE:y, x:x+BLOCK_SIZE]
                neighbors.append(np.mean(np.abs(block - upper)))

            if c > 0:
                left = image[y:y+BLOCK_SIZE, x-BLOCK_SIZE:x]
                neighbors.append(np.mean(np.abs(block - left)))

            if neighbors:
                differences[r, c] = np.mean(neighbors)

    valid = np.isfinite(differences)
    minimum = np.min(differences[valid])
    maximum = np.max(differences[valid])

    normalized = (differences - minimum) / (maximum - minimum + 1e-6)

    nearest = np.rint(coefficients / DELTA) * DELTA
    small_move = np.abs(coefficients - nearest)
    large_move = DELTA - small_move

    factor = 1 - normalized + 1e-6

    keep_cost = small_move * factor
    flip_cost = large_move / factor

    return coefficients, keep_cost, flip_cost

# educational four-state syndrome-trellis code
def stc_encode(cover, message, keep_costs, flip_costs):
    k = len(message)
    n = len(cover)

    assert n == 3*k + 2

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
