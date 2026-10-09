"""Shared allocation of three padded header groups followed by body frames."""
from dataclasses import dataclass

from .config import HEADER_COPIES
from .ecc import HEADER_SIZE
from .video_stego import chunk_size


@dataclass(frozen=True)
class FrameChunk:
    offset: int
    length: int
    size: int


def bootstrap_copy_frames(shape):
    size = chunk_size(shape)
    if not size:
        raise ValueError('Video too small for embedding')
    return (HEADER_SIZE + size - 1) // size


def bootstrap_frames(shape):
    return HEADER_COPIES * bootstrap_copy_frames(shape)


def message_capacity(shape, frames):
    count = bootstrap_frames(shape)
    if frames < count:
        return 0
    return HEADER_SIZE * HEADER_COPIES + (frames - count) * chunk_size(shape)


def frame_chunk(shape, message_length, frame_index):
    if frame_index < 0 or message_length < 0:
        raise ValueError('Invalid frame index or message length')
    count = bootstrap_frames(shape)
    size = chunk_size(shape)
    if frame_index < count:
        copy_index, local_frame = divmod(frame_index, bootstrap_copy_frames(shape))
        offset = copy_index * HEADER_SIZE + local_frame * size
        boundary = min((copy_index + 1) * HEADER_SIZE, message_length)
    else:
        offset = HEADER_SIZE * HEADER_COPIES + (frame_index - count) * size
        boundary = message_length
    return FrameChunk(offset, max(0, min(size, boundary - offset)), size)
