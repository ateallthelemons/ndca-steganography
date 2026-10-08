"""One frame allocation shared by encoding, verification, capacity, and diagnostics."""
from dataclasses import dataclass

from .ecc import HEADER_SIZE
from .profiles import LEGACY, Profile, get_profile, bootstrap_copies
from .video_stego import chunk_size


@dataclass(frozen=True)
class FrameChunk:
    profile: Profile
    offset: int
    length: int
    size: int


def bootstrap_copy_frames(shape, profile=LEGACY):
    profile = get_profile(profile)
    size = chunk_size(shape, profile if profile.version == 'NDV5' else LEGACY)
    if not size:
        raise ValueError('Video too small for the fixed bootstrap profile')
    return (HEADER_SIZE + size - 1) // size


def bootstrap_frames(shape, profile=LEGACY):
    return bootstrap_copies(profile) * bootstrap_copy_frames(shape, profile)


def message_capacity(shape, frames, profile=LEGACY):
    profile = get_profile(profile)
    count = bootstrap_frames(shape, profile)
    if profile == LEGACY:
        return frames * chunk_size(shape)
    if frames < count:
        if profile.version == 'NDV5':
            return 0
        return frames * chunk_size(shape)
    return HEADER_SIZE * bootstrap_copies(profile) + (frames - count) * chunk_size(shape, profile)


def frame_chunk(shape, message_length, frame_index, profile=LEGACY):
    profile = get_profile(profile)
    if frame_index < 0 or message_length < 0:
        raise ValueError('Invalid frame index or message length')
    if profile == LEGACY:
        size = chunk_size(shape)
        offset = frame_index * size
        boundary = message_length
    else:
        count = bootstrap_frames(shape, profile)
        header_bytes = HEADER_SIZE * bootstrap_copies(profile)
        if frame_index < count:
            copy_frames = bootstrap_copy_frames(shape, profile)
            copy_index, local_frame = divmod(frame_index, copy_frames)
            profile = profile if profile.version == 'NDV5' else LEGACY
            size = chunk_size(shape, profile)
            offset = copy_index * HEADER_SIZE + local_frame * size
            boundary = min((copy_index + 1) * HEADER_SIZE, message_length)
        else:
            size = chunk_size(shape, profile)
            offset = header_bytes + (frame_index - count) * size
            boundary = message_length
    if not size:
        raise ValueError('Video too small for selected embedding profile')
    return FrameChunk(profile, offset, max(0, min(size, boundary - offset)), size)
