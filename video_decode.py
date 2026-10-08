import argparse
from pathlib import Path

from core.framing import MessageReader
from core.profiles import get_profile
from core.progress import Progress
from core.video_io import video_frames


def recover(filename, *, progress=False):
    """Extract, repair, and validate a file without writing to the filesystem."""
    reader = MessageReader()
    status = Progress('Decoding', enabled=progress)
    frames = 0
    for frames, (_, U, _) in enumerate(video_frames(filename), start=1):
        result = reader.feed_frame(U)
        status.update(frames, f'{len(reader.received)} encoded bytes read',
                      final=result is not None)
        if result is not None:
            return result
    result = reader.feed(b'', final=True)
    status.update(frames, f'{len(reader.received)} encoded bytes read', final=True)
    return result


def decode(filename, output=None, *, progress=True):
    result = recover(filename, progress=progress)
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
    args = parser.parse_args()
    decode(args.input, args.output)
