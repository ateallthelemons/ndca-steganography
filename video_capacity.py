import argparse
from pathlib import Path

import av

from video_encode import HEADER
from core.video_stego import chunk_size


def capacity(input_name):
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
                bytes_per_frame = chunk_size((frame.height // 2, frame.width // 2))
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
    args = parser.parse_args()

    name = Path(args.filename).name if args.filename is not None else None
    name_size = len(name.encode('utf-8')) if name is not None else 1
    if name is not None and (not name or any(c in name for c in ('\\', ':', '\0'))
                             or name in ('.', '..') or name_size > 65535):
        parser.error('Payload filename must be a valid basename of 1–65535 UTF-8 bytes')
    try:
        (width, height), frames, per_frame = capacity(args.input)
    except (ValueError, OSError, av.error.FFmpegError) as error:
        parser.exit(1, f'Error: {error}\n')

    total = frames * per_frame
    maximum = min(total - HEADER.size - name_size, (1 << 64) - 1)
    print(f'Video: {width}x{height}, {frames:,} frames')
    print(f'Capacity per frame: {per_frame:,} bytes')
    print(f'Total embedded capacity: {total:,} bytes')
    if maximum < 0:
        print('No file fits: insufficient room for the header and filename, even with an empty payload.')
    elif name is not None:
        print(f'Header + filename: {HEADER.size + name_size:,} bytes ({name})')
        print(f'Maximum file size: {maximum:,} bytes ({maximum / 1024:.2f} KiB)')
    else:
        print(f'Maximum file size with a 1-byte filename: {maximum:,} bytes ({maximum / 1024:.2f} KiB)')
        print(f'For your filename: {total - HEADER.size:,} minus its UTF-8 byte length.')
        print('Use --filename FILE for the exact limit for that filename.')


if __name__ == '__main__':
    main()
