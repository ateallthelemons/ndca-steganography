"""One frame allocation shared by encoding, verification, capacity, and diagnostics."""
from dataclasses import dataclass

from .ecc import HEADER_SIZE
from .profiles import LEGACY, Profile, get_profile
from .video_stego import chunk_size


@dataclass(frozen=True)
class FrameChunk:
    profile: Profile
    offset: int
    length: int
    size: int


def bootstrap_frames(shape):
    size = chunk_size(shape)
    if not size:
        raise ValueError('Video too small for the fixed bootstrap profile')
    return (HEADER_SIZE + size - 1) // size


def message_capacity(shape, frames, profile=LEGACY):
    profile = get_profile(profile)
    count = bootstrap_frames(shape)
    if profile == LEGACY:
        return frames * chunk_size(shape)
    if frames < count:
        return frames * chunk_size(shape)
    return HEADER_SIZE + (frames - count) * chunk_size(shape, profile)


def frame_chunk(shape, message_length, frame_index, profile=LEGACY):
    profile = get_profile(profile)
    if frame_index < 0 or message_length < 0:
        raise ValueError('Invalid frame index or message length')
    if profile == LEGACY:
        size = chunk_size(shape)
        offset = frame_index * size
        boundary = message_length
    else:
        count = bootstrap_frames(shape)
        if frame_index < count:
            profile = LEGACY
            size = chunk_size(shape)
            offset = frame_index * size
            boundary = min(HEADER_SIZE, message_length)
        else:
            size = chunk_size(shape, profile)
            offset = HEADER_SIZE + (frame_index - count) * size
            boundary = message_length
    if not size:
        raise ValueError('Video too small for selected embedding profile')
    return FrameChunk(profile, offset, max(0, min(size, boundary - offset)), size)
