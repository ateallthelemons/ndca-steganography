import argparse
import struct
import zlib
from fractions import Fraction
from pathlib import Path

import av

from core.video_io import unpack_video_frame, make_video_frame, rescale
from core.video_stego import chunk_size, embed_chunk

MAGIC = b'NDV3'
HEADER = struct.Struct('>4sQIH')  # magic, uint64 length, CRC32, uint16 filename length


def encode(input_name, output_name, payload):
    payload_path = Path(payload)
    if any(c in payload_path.name for c in ('\\', ':')):
        raise ValueError('Payload filename cannot contain backslashes or colons')
    filename = payload_path.name.encode('utf-8')
    if not filename or len(filename) > (1 << 16) - 1:
        raise ValueError('Filename must contain 1–65535 UTF-8 bytes')
    data = payload_path.read_bytes()
    if len(data) > (1 << 64) - 1:
        raise ValueError('Payload exceeds 64-bit header limit')
    message = HEADER.pack(MAGIC, len(data), zlib.crc32(data), len(filename)) + filename + data

    total = 0
    frames = 0
    first_size = None

    try:
        with av.open(input_name) as source, av.open(output_name, 'w') as target:
            input_video = source.streams.video[0]
            if input_video.width % 2 or input_video.height % 2:
                raise ValueError('This 8-bit YUV420p version needs even video dimensions')

            rate = input_video.average_rate or input_video.guessed_rate or Fraction(30, 1)
            output_video = target.add_stream('ffv1', rate=rate)
            output_video.width = input_video.width
            output_video.height = input_video.height
            output_video.pix_fmt = 'yuv420p'
            output_video.codec_context.time_base = Fraction(1, 90000)

            audio_map = {}
            for stream in source.streams.audio:
                audio_map[stream.index] = target.add_stream_from_template(stream)

            selected = [input_video, *source.streams.audio]
            for packet in source.demux(selected):
                if packet.stream.type == 'audio':
                    if packet.dts is None:
                        continue
                    packet.stream = audio_map[packet.stream.index]
                    target.mux(packet)
                    continue

                for frame in packet.decode():
                    Y, U, V = unpack_video_frame(frame)
                    size = chunk_size(U.shape)
                    if not size:
                        raise ValueError('Video too small: no available U-plane payload capacity')
                    if first_size is None:
                        first_size = size
                        print(f'Frame size: {frame.width}x{frame.height}, payload: {size} bytes/frame')
                    elif first_size != size:
                        raise ValueError('Midstream resolution changes are not supported')

                    if total < len(message):
                        part = message[total:total+size].ljust(size, b'\0')
                        U, _ = embed_chunk(U, part)
                        total += min(size, len(message)-total)

                    if frame.pts is None or frame.time_base is None:
                        raise ValueError('Input video frame has no usable timestamp')
                    pts = rescale(frame.pts, frame.time_base, output_video.codec_context.time_base)
                    encoded = make_video_frame(Y, U, V, pts, output_video.codec_context.time_base)
                    for out_packet in output_video.encode(encoded):
                        target.mux(out_packet)
                    frames += 1

            if total != len(message):
                raise ValueError(f'Video too short: embedded {total} of {len(message)} bytes')
            for out_packet in output_video.encode(None):
                target.mux(out_packet)
    except Exception:
        Path(output_name).unlink(missing_ok=True)
        raise

    print(f'Saved {output_name}: {frames} video frames, {len(data)} payload bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video')
    parser.add_argument('payload', help='File to hide')
    parser.add_argument('output', help='Output MKV file')
    args = parser.parse_args()
    encode(args.input, args.output, args.payload)
