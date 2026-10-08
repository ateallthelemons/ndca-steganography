"""Binary syndrome-trellis coding with an explicit height-three banded matrix.

This independent implementation follows the STC parity-check/trellis construction
of Filler, Judas, and Fridrich (2011). Its fixed small matrices are defined here;
they are not claimed to reproduce the NDCA authors' unspecified matrix choices.
Reference: https://dde.binghamton.edu/download/syndrome/
"""
import numpy as np

from .ndca import EmbeddingError

# LSB is the current syndrome row; higher bits affect following rows.
COLUMNS = {2: (7, 5), 3: (7, 5, 3)}
HEIGHT = 3


def columns(width):
    if width not in COLUMNS:
        raise ValueError('STC supports only rates 1/2 and 1/3')
    return COLUMNS[width]


def parity_check(message_bits, width):
    """Explicit matrix, useful as an independent oracle for small test cases."""
    masks = columns(width)
    if message_bits < 0:
        raise ValueError('Negative message length')
    matrix = np.zeros((message_bits, message_bits * width), dtype=np.uint8)
    for row in range(message_bits):
        for column, mask in enumerate(masks):
            for offset in range(min(HEIGHT, message_bits - row)):
                matrix[row + offset, row * width + column] = (mask >> offset) & 1
    return matrix


def extract(carrier, width):
    carrier = np.asarray(carrier)
    masks = columns(width)
    if carrier.ndim != 1 or len(carrier) % width or np.any((carrier != 0) & (carrier != 1)):
        raise ValueError('Invalid STC carrier bits')
    length = len(carrier) // width
    result = np.zeros(length, dtype=np.uint8)
    for row in range(length):
        for column, mask in enumerate(masks):
            if carrier[row * width + column]:
                for offset in range(min(HEIGHT, length - row)):
                    result[row + offset] ^= (mask >> offset) & 1
    return result


def embed(message, costs, width):
    """Minimize sum(costs[i, y_i]) subject to H y = message over GF(2).

    Each candidate bit has its own absolute cost (including keep modulation).
    +inf forbids a bit. Start with zero syndrome state, enforce/shift one row
    after each group, and truncate columns at the last message row. No extra
    tail carriers are needed; the terminal state is zero.
    """
    masks = columns(width)
    message = np.asarray(message)
    costs = np.asarray(costs, dtype=np.float64)
    if message.ndim != 1 or np.any((message != 0) & (message != 1)):
        raise ValueError('Invalid STC message bits')
    if costs.shape != (len(message) * width, 2) or np.any(np.isnan(costs)) or np.any(costs < 0):
        raise ValueError('STC requires nonnegative per-bit costs or +inf')
    states = 1 << HEIGHT
    dp = np.full(states, np.inf)
    dp[0] = 0
    previous = np.full((len(costs), states), -1, dtype=np.int8)
    choices = np.full((len(costs), states), -1, dtype=np.int8)
    for i in range(len(costs)):
        row, column = divmod(i, width)
        mask = masks[column] & ((1 << min(HEIGHT, len(message) - row)) - 1)
        following = np.full(states, np.inf)
        for state in range(states):
            if not np.isfinite(dp[state]):
                continue
            for bit in (0, 1):
                target = state ^ (mask if bit else 0)
                if column == width - 1:
                    if (target & 1) != message[row]:
                        continue
                    target >>= 1
                value = dp[state] + costs[i, bit]
                if value < following[target]:
                    following[target] = value
                    previous[i, target] = state
                    choices[i, target] = bit
        dp = following
    if not np.isfinite(dp[0]):
        raise EmbeddingError('No feasible matrix STC path for this chunk')
    state = 0
    result = np.zeros(len(costs), dtype=np.uint8)
    for i in range(len(costs) - 1, -1, -1):
        result[i] = choices[i, state]
        state = int(previous[i, state])
    return result, float(dp[0])
