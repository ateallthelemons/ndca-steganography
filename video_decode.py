import argparse
import struct
import zlib
from pathlib import Path

from core.video_io import video_frames
from core.video_stego import extract_chunk

MAGIC = b'NDV3'
HEADER = struct.Struct('>4sQIH')
LEGACY_MAGIC = b'NDV2'
LEGACY_HEADER = struct.Struct('>4sQI')


def decode(filename, output=None):
    received = bytearray()
    expected_total = None
    expected_crc = None
    payload_start = None
    stored_name = None

    for _, U, _ in video_frames(filename):
        received.extend(extract_chunk(U))

        if expected_total is None and len(received) >= 4:
            magic = bytes(received[:4])
            if magic == LEGACY_MAGIC:
                if output is None:
                    raise ValueError('NDV2 videos have no stored filename; supply an output path')
                if len(received) < LEGACY_HEADER.size:
                    continue
                _, length, expected_crc = LEGACY_HEADER.unpack(received[:LEGACY_HEADER.size])
                payload_start = LEGACY_HEADER.size
            elif magic == MAGIC:
                if len(received) < HEADER.size:
                    continue
                _, length, expected_crc, name_length = HEADER.unpack(received[:HEADER.size])
                payload_start = HEADER.size + name_length
                if len(received) < payload_start:
                    continue
                stored_name = bytes(received[HEADER.size:payload_start]).decode('utf-8')
                if (not stored_name or stored_name in ('.', '..')
                        or any(c in stored_name for c in ('/', '\\', '\0', ':'))):
                    raise ValueError('Stored filename is unsafe or damaged')
            else:
                raise ValueError('Header not found or damaged')
            expected_total = payload_start + length

        if expected_total is not None and len(received) >= expected_total:
            payload = bytes(received[payload_start:expected_total])
            if zlib.crc32(payload) != expected_crc:
                raise ValueError('Payload corrupted (CRC32 mismatch)')
            destination = Path(output) if output is not None else Path(stored_name)
            # Automatic restoration must not overwrite a file in the current directory.
            with destination.open('wb' if output is not None else 'xb') as target:
                target.write(payload)
            print(f'Recovered {len(payload)} bytes into {destination}')
            return

    raise ValueError('Video ended before a complete payload was recovered')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output', nargs='?', help='Output path (default: stored filename in current directory)')
    args = parser.parse_args()
    decode(args.input, args.output)
