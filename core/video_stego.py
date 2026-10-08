import numpy as np

from .ndca import (
    BLOCK_SIZE, get_positions, get_coefficient, modify_coefficient,
    encode_bit, decode_bit, calculate_costs, stc_encode, stc_decode
)


def chunk_size(shape):
    height, width = shape
    blocks = (height // BLOCK_SIZE) * (width // BLOCK_SIZE)
    return max(0, ((blocks - 3) // 3) // 8)


def embed_chunk(U, data):
    if len(data) != chunk_size(U.shape) or not data:
        raise ValueError('Invalid chunk size for this frame')

    image = U.astype(np.float64)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    positions = get_positions(image.shape, len(bits))
    cols = image.shape[1] // BLOCK_SIZE

    coefficients, keep_costs, flip_costs = calculate_costs(image)
    coefficients = coefficients.ravel()[positions]
    keep_costs = keep_costs.ravel()[positions]
    flip_costs = flip_costs.ravel()[positions]

    cover = np.array([decode_bit(c) for c in coefficients], dtype=np.uint8)
    stego_bits, cost = stc_encode(cover, bits, keep_costs, flip_costs)
    assert np.array_equal(stc_decode(stego_bits), bits)

    result = image.copy()
    for i, position in enumerate(positions):
        r, c = divmod(int(position), cols)
        y, x = r * BLOCK_SIZE, c * BLOCK_SIZE
        block = image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
        value = encode_bit(coefficients[i], int(stego_bits[i]))
        result[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = modify_coefficient(block, value)

    return np.clip(np.rint(result), 0, 255).astype(np.uint8), cost


def extract_chunk(U):
    size = chunk_size(U.shape)
    if not size:
        raise ValueError('U plane too small for embedding')

    image = U.astype(np.float64)
    positions = get_positions(image.shape, size * 8)
    cols = image.shape[1] // BLOCK_SIZE
    carrier = []

    for position in positions:
        r, c = divmod(int(position), cols)
        y, x = r * BLOCK_SIZE, c * BLOCK_SIZE
        carrier.append(decode_bit(get_coefficient(image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE])))

    return np.packbits(stc_decode(np.asarray(carrier, dtype=np.uint8))).tobytes()
