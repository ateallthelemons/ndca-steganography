import argparse
from pathlib import Path

import av

from core.ecc import HEADER_SIZE
from core.framing import filename_bytes, maximum_payload, MAX_PAYLOAD_BYTES, require_embedding_profile
from core.video_stego import chunk_size
from core.profiles import LEGACY, add_profile_arguments, profile_from_args
from core.layout import message_capacity, bootstrap_frames


def capacity(input_name, profile=LEGACY):
    """Count decoded frames to calculate capacity using the current embedding settings."""
    frames = 0
    dimensions = None
    bytes_per_frame = None

    with av.open(input_name) as source:
        if not source.streams.video:
            raise ValueError('Input has no video stream')
        stream = source.streams.video[0]
        for frame in source.decode(stream):
            current_dimensions = (frame.width, frame.height)
            if frame.width % 2 or frame.height % 2:
                raise ValueError('This 8-bit YUV420p version needs even video dimensions')
            if dimensions is None:
                dimensions = current_dimensions
                bytes_per_frame = chunk_size((frame.height // 2, frame.width // 2), profile)
                bootstrap_frames((frame.height // 2, frame.width // 2))
                if not bytes_per_frame:
                    raise ValueError('Video too small: no available U-plane payload capacity')
            elif current_dimensions != dimensions:
                raise ValueError('Midstream resolution changes are not supported')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Input video frame has no usable timestamp')
            frames += 1

    if not frames:
        raise ValueError('Input has no decodable video frames')
    return dimensions, frames, bytes_per_frame


def main():
    parser = argparse.ArgumentParser(description='Check maximum file size using the current embedding settings.')
    parser.add_argument('input', help='Input video (scans all frames for an accurate count)')
    parser.add_argument('--filename', help='Payload filename or path; the file need not exist')
    add_profile_arguments(parser)
    args = parser.parse_args()

    name = Path(args.filename).name if args.filename is not None else None
    try:
        require_embedding_profile()
        profile = profile_from_args(args)
        name_size = len(filename_bytes(name)) if name is not None else 1
        (width, height), frames, per_frame = capacity(args.input, profile)
    except (ValueError, OSError, av.error.FFmpegError) as error:
        parser.exit(1, f'Error: {error}\n')

    total = message_capacity((height // 2, width // 2), frames, profile)
    maximum = maximum_payload(total, name_size)
    print(f'Video: {width}x{height}, {frames:,} frames')
    print(f'Embedding profile: {profile.name}')
    print(f'Body capacity per frame before RS: {per_frame:,} bytes')
    if profile != LEGACY:
        print(f'Reserved bootstrap frames: {bootstrap_frames((height // 2, width // 2))}')
    print(f'Framed message capacity (excludes bootstrap-frame padding): {total:,} bytes')
    print(f'NDV4: {HEADER_SIZE}-byte protected bootstrap; RS(60,20) body, including filename and CRC32')
    print(f'Payload implementation limit: {MAX_PAYLOAD_BYTES:,} bytes')
    if maximum < 0:
        print('No file fits: insufficient room for the header and filename, even with an empty payload.')
    elif name is not None:
        print(f'Filename: {name_size:,} UTF-8 bytes ({name})')
        print(f'Maximum file size: {maximum:,} bytes ({maximum / 1024:.2f} KiB)')
    else:
        print(f'Maximum file size with a 1-byte filename: {maximum:,} bytes ({maximum / 1024:.2f} KiB)')
        print('Use --filename FILE for the exact limit for that filename.')


if __name__ == '__main__':
    main()
