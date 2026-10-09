import argparse
import tempfile
from fractions import Fraction
from pathlib import Path

import av

from core.ndca import EmbeddingError
from core.framing import MAX_PAYLOAD_BYTES, pack_message
from core.layout import frame_chunk
from core.parallel import default_workers, ordered_map
from core.progress import Progress
from core.video_io import unpack_video_frame, make_video_frame, rescale, video_frames
from core.video_stego import chunk_size, embed_chunk, extract_chunk

def embed_frame(U, part, number):
    """Embed one frame's chunk; runs in a worker process."""
    try:
        U, _ = embed_chunk(U, part)
        return U
    except EmbeddingError as error:
        raise EmbeddingError(f'Embedding failed at frame {number}: {error}') from error


def verify_saved_video(filename, message, dimensions, expected_frames,
                       *, progress=False, workers=None):
    """Read the closed file and compare all embedded bytes, including padding."""
    status = Progress('Verifying saved video', enabled=progress)

    def tasks():
        planned = 0
        for index, (Y, U, _) in enumerate(video_frames(filename)):
            if (Y.shape[1], Y.shape[0]) != dimensions:
                raise EmbeddingError(f'Saved video dimensions changed at frame {index + 1}')
            if planned < len(message):
                plan = frame_chunk(U.shape, len(message), index)
                expected = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
                planned += plan.length
                yield extract_chunk, (U,), (plan.length, expected)
            else:
                yield None, (), (0, None)

    total = 0
    frames = 0
    for (length, expected), extracted in ordered_map(tasks(), workers):
        frames += 1
        if expected is not None:
            if extracted != expected:
                raise EmbeddingError(f'Saved FFV1 verification failed at frame {frames}')
            total += length
        status.update(frames, f'{expected_frames} frames expected; {total}/{len(message)} message bytes checked')
    if total != len(message):
        raise EmbeddingError(f'Saved video ended after {total} of {len(message)} message bytes')
    if frames != expected_frames:
        raise EmbeddingError(f'Saved video has {frames} frames; expected {expected_frames}')
    status.update(frames, f'{total}/{len(message)} message bytes checked', final=True)


def check_output_path(destination, *sources):
    for source in sources:
        source = Path(source)
        if (destination.resolve() == source.resolve()
                or (destination.exists() and source.exists() and destination.samefile(source))):
            raise ValueError('Output must not alias the input video or payload')


def encode(input_name, output_name, payload, *, progress=True, workers=None):
    destination = Path(output_name)
    payload_path = Path(payload)
    check_output_path(destination, input_name, payload_path)
    if payload_path.stat().st_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f'Payload exceeds supported limit of {MAX_PAYLOAD_BYTES} bytes')
    data = payload_path.read_bytes()
    message = pack_message(payload_path.name, data)
    status = Progress('Encoding', enabled=progress)

    total = 0
    frames = 0
    dimensions = None

    # Keep the original destination intact until the closed FFV1 file verifies.
    with tempfile.NamedTemporaryFile(prefix=f'.{destination.name}.', suffix='.mkv',
                                     dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with av.open(input_name) as source, av.open(str(temporary_path), 'w') as target:
            if not source.streams.video:
                raise ValueError('Input has no video stream')
            input_video = source.streams.video[0]
            if input_video.width % 2 or input_video.height % 2:
                raise ValueError('This 8-bit YUV420p version needs even video dimensions')

            rate = input_video.average_rate or input_video.guessed_rate or Fraction(30, 1)
            output_video = target.add_stream('ffv1', rate=rate)
            output_video.width = input_video.width
            output_video.height = input_video.height
            dimensions = (input_video.width, input_video.height)
            output_video.pix_fmt = 'yuv420p'
            output_video.codec_context.time_base = Fraction(1, 90000)

            audio_map = {}
            for stream in source.streams.audio:
                audio_map[stream.index] = target.add_stream_from_template(stream)

            selected = [input_video, *source.streams.audio]
            time_base = output_video.codec_context.time_base

            def tasks():
                """Mux audio as it arrives; describe each video frame's embedding work."""
                nonlocal total
                index = 0
                for packet in source.demux(selected):
                    if packet.stream.type == 'audio':
                        if packet.dts is None:
                            continue
                        packet.stream = audio_map[packet.stream.index]
                        target.mux(packet)
                        continue

                    for frame in packet.decode():
                        if (frame.width, frame.height) != dimensions:
                            raise ValueError('Midstream resolution changes are not supported')
                        Y, U, V = unpack_video_frame(frame)
                        size = chunk_size(U.shape)
                        if not size:
                            raise ValueError('Video too small: no available U-plane payload capacity')
                        if index == 0:
                            print(f'Frame size: {frame.width}x{frame.height}, body: {size} bytes/frame')
                        if frame.pts is None or frame.time_base is None:
                            raise ValueError('Input video frame has no usable timestamp')
                        pts = rescale(frame.pts, frame.time_base, time_base)

                        function, args = None, ()
                        if total < len(message):
                            plan = frame_chunk(U.shape, len(message), index)
                            part = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
                            function, args = embed_frame, (U, part, index + 1)
                            total += plan.length
                        yield function, args, (Y, U, V, pts)
                        index += 1

            for (Y, U, V, pts), embedded in ordered_map(tasks(), workers):
                encoded = make_video_frame(Y, U if embedded is None else embedded, V, pts, time_base)
                for out_packet in output_video.encode(encoded):
                    target.mux(out_packet)
                frames += 1
                status.update(frames, f'{frames} frames written')

            if total != len(message):
                raise ValueError(f'Video too short: embedded {total} of {len(message)} bytes')
            for out_packet in output_video.encode(None):
                target.mux(out_packet)
        status.update(frames, f'{total}/{len(message)} message bytes embedded', final=True)
        verify_saved_video(str(temporary_path), message, dimensions, frames,
                           progress=progress, workers=workers)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f'Saved and verified {output_name}: {frames} video frames, {len(data)} payload bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video')
    parser.add_argument('payload', help='File to hide')
    parser.add_argument('output', help='Output MKV file')
    parser.add_argument('--workers', type=int, default=default_workers(),
                        help='CPU processes for per-frame work (default: all cores; 1 disables)')
    args = parser.parse_args()
    encode(args.input, args.output, args.payload, workers=args.workers)
