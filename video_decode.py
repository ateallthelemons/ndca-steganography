import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

from core.ecc import UncorrectableBodyError
from core.framing import MessageReader
from core.config import HEADER_COPIES, PROFILES, add_profile_argument
from core.parallel import default_workers, ordered_map
from core.progress import Progress
from core.video_io import video_frames
from core.video_stego import extract_chunk_with_confidence


def recover(filename, *, settings, progress=False, workers=None):
    """Extract, repair, and validate a file without writing to the filesystem."""
    try:
        return _recover(filename, settings=settings, progress=progress, workers=workers)
    except UncorrectableBodyError:
        # One bounded retry. Do not retry ambiguous repairs, CRC/padding errors,
        # invalid headers, truncated input or I/O errors. Re-read rather than
        # retaining a second score array for every otherwise successful file.
        pass
    result = _recover(filename, settings=settings, progress=progress, workers=workers,
                      confidence_model='likelihood')
    return replace(result, confidence_model='likelihood')


def _recover(filename, *, settings, progress=False, workers=None, confidence_model='minimum'):
    reader = MessageReader()
    label = 'Decoding' if confidence_model == 'minimum' else 'Retrying with likelihood confidence'
    status = Progress(label, enabled=progress)
    planes = (U for _, U, _ in video_frames(filename))
    frames = 0

    # Read all independently padded header groups before processing the body.
    for U in planes:
        frames += 1
        result = reader.feed_frame(U, settings, confidence_model=confidence_model)
        status.update(frames, f'{len(reader.received)} encoded bytes read',
                      final=result is not None)
        if result is not None:
            return result
        if reader.header is not None:
            break

    if reader.header is not None:
        extract = partial(extract_chunk_with_confidence, settings=settings, model=confidence_model)
        tasks = ((extract, (U,), None) for U in planes)
        results = ordered_map(tasks, workers)
        try:
            for _, (chunk, confidence) in results:
                frames += 1
                result = reader.feed(chunk, confidence=confidence)
                status.update(frames, f'{len(reader.received)} encoded bytes read',
                              final=result is not None)
                if result is not None:
                    return result
        finally:
            results.close()

    result = reader.finish()
    status.update(frames, f'{len(reader.received)} encoded bytes read', final=True)
    return result


def decode(filename, output=None, *, settings, progress=True, workers=None):
    result = recover(filename, settings=settings, progress=progress, workers=workers)
    destination = Path(output) if output is not None else Path.cwd()
    restore_name = output is None or destination.is_dir()
    if restore_name:
        destination = destination / result.filename
    # Nothing is written until both error correction and integrity checks pass.
    with destination.open('xb' if restore_name else 'wb') as target:
        target.write(result.payload)
    print(f'Recovered {len(result.payload)} bytes into {destination} (NDV5)')
    print(f'Corrected bytes: bootstrap {result.header_corrections}, '
          f'body {sum(result.body_corrections)} '
          f'(maximum {max(result.body_corrections, default=0)} per codeword)')
    print(f'Recoverable header copies: {result.header_copies_recovered}/{HEADER_COPIES}')
    if result.confidence_model == 'likelihood':
        print('Recovered using likelihood-confidence fallback')
    if any(result.body_erasures):
        print(f'Confidence-guided erasure repair: '
              f'{sum(n > 0 for n in result.body_erasures)} body codewords; '
              f'{sum(result.body_erasures)} byte positions marked unreliable')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output', nargs='?',
                        help='Output file or existing directory (default: stored filename in current directory)')
    parser.add_argument('--workers', type=int, default=default_workers(),
                        help='CPU processes for per-frame work (default: all cores; 1 disables)')
    add_profile_argument(parser)
    args = parser.parse_args()
    decode(args.input, args.output, settings=PROFILES[args.profile], workers=args.workers)
