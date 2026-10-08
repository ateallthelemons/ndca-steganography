import argparse
from pathlib import Path

from core.framing import MessageReader
from core.parallel import default_workers, ordered_map
from core.profiles import LEGACY, get_profile
from core.progress import Progress
from core.video_io import video_frames
from core.video_stego import extract_chunk


def recover(filename, *, progress=False, workers=None):
    """Extract, repair, and validate a file without writing to the filesystem."""
    reader = MessageReader()
    status = Progress('Decoding', enabled=progress)
    planes = (U for _, U, _ in video_frames(filename))
    frames = 0

    # The bootstrap frames say which profile the rest uses, so read them in order.
    for U in planes:
        frames += 1
        result = reader.feed_frame(U)
        status.update(frames, f'{len(reader.received)} encoded bytes read',
                      final=result is not None)
        if result is not None:
            return result
        if reader.header is not None or reader.legacy:
            break

    if reader.header is not None or reader.legacy:
        profile = get_profile(reader.header.profile_id) if reader.header is not None else LEGACY
        tasks = ((extract_chunk, (U, profile), None) for U in planes)
        for _, chunk in ordered_map(tasks, workers):
            frames += 1
            result = reader.feed(chunk)
            status.update(frames, f'{len(reader.received)} encoded bytes read',
                          final=result is not None)
            if result is not None:
                return result

    result = reader.feed(b'', final=True)
    status.update(frames, f'{len(reader.received)} encoded bytes read', final=True)
    return result


def decode(filename, output=None, *, progress=True, workers=None):
    result = recover(filename, progress=progress, workers=workers)
    destination = Path(output) if output is not None else Path.cwd()
    restore_name = output is None or destination.is_dir()
    if restore_name:
        if result.filename is None:
            raise ValueError('NDV2 videos have no stored filename; supply an output path including a filename')
        destination = destination / result.filename
    # Nothing is written until both error correction and integrity checks pass.
    with destination.open('xb' if restore_name else 'wb') as target:
        target.write(result.payload)
    print(f'Recovered {len(result.payload)} bytes into {destination} ({result.version})')
    if result.version == 'NDV4':
        print(f'Embedding profile: {get_profile(result.profile_id).name}')
        print(f'Corrected bytes: bootstrap {result.header_corrections}, '
              f'body {sum(result.body_corrections)} '
              f'(maximum {max(result.body_corrections, default=0)} per codeword)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output', nargs='?',
                        help='Output file or existing directory (default: stored filename in current directory)')
    parser.add_argument('--workers', type=int, default=default_workers(),
                        help='CPU processes for per-frame work (default: all cores; 1 disables)')
    args = parser.parse_args()
    decode(args.input, args.output, workers=args.workers)
