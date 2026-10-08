import argparse
import tempfile
from fractions import Fraction
from pathlib import Path

import av

from core.ndca import EmbeddingError
from core.framing import MAX_PAYLOAD_BYTES, pack_message, require_embedding_profile
from core.profiles import LEGACY, get_profile, add_profile_arguments, profile_from_args
from core.layout import frame_chunk
from core.progress import Progress
from core.video_io import unpack_video_frame, make_video_frame, rescale, video_frames
from core.video_stego import chunk_size, embed_chunk, extract_chunk

def verify_saved_video(filename, message, dimensions, expected_frames, profile=LEGACY,
                       *, progress=False):
    """Read the closed file and compare all embedded bytes, including padding."""
    total = 0
    frames = 0
    status = Progress('Verifying saved video', enabled=progress)
    for frames, (Y, U, _) in enumerate(video_frames(filename), start=1):
        if (Y.shape[1], Y.shape[0]) != dimensions:
            raise EmbeddingError(f'Saved video dimensions changed at frame {frames}')
        if total < len(message):
            plan = frame_chunk(U.shape, len(message), frames - 1, profile)
            expected = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
            if extract_chunk(U, plan.profile) != expected:
                raise EmbeddingError(f'Saved FFV1 verification failed at frame {frames}')
            total += plan.length
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


def encode(input_name, output_name, payload, profile=LEGACY, *, progress=True):
    require_embedding_profile()
    profile = get_profile(profile)
    destination = Path(output_name)
    payload_path = Path(payload)
    check_output_path(destination, input_name, payload_path)
    if payload_path.stat().st_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f'Payload exceeds supported limit of {MAX_PAYLOAD_BYTES} bytes')
    data = payload_path.read_bytes()
    message = pack_message(payload_path.name, data, profile)
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
                    size = chunk_size(U.shape, profile)
                    if not size:
                        raise ValueError('Video too small: no available U-plane payload capacity')
                    if frames == 0:
                        print(f'Frame size: {frame.width}x{frame.height}, body: {size} bytes/frame, profile: {profile.name}')

                    if total < len(message):
                        plan = frame_chunk(U.shape, len(message), frames, profile)
                        part = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
                        try:
                            U, _ = (embed_chunk(U, part) if plan.profile == LEGACY
                                    else embed_chunk(U, part, plan.profile))
                        except EmbeddingError as error:
                            raise EmbeddingError(f'Embedding failed at frame {frames + 1}: {error}') from error
                        total += plan.length

                    if frame.pts is None or frame.time_base is None:
                        raise ValueError('Input video frame has no usable timestamp')
                    pts = rescale(frame.pts, frame.time_base, output_video.codec_context.time_base)
                    encoded = make_video_frame(Y, U, V, pts, output_video.codec_context.time_base)
                    for out_packet in output_video.encode(encoded):
                        target.mux(out_packet)
                    frames += 1
                    status.update(frames, f'{total}/{len(message)} message bytes embedded')

            if total != len(message):
                raise ValueError(f'Video too short: embedded {total} of {len(message)} bytes')
            for out_packet in output_video.encode(None):
                target.mux(out_packet)
        status.update(frames, f'{total}/{len(message)} message bytes embedded', final=True)
        verify_saved_video(str(temporary_path), message, dimensions, frames, profile,
                           progress=progress)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f'Saved and verified {output_name}: {frames} video frames, {len(data)} payload bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input video')
    parser.add_argument('payload', help='File to hide')
    parser.add_argument('output', help='Output MKV file')
    add_profile_arguments(parser)
    args = parser.parse_args()
    try:
        profile = profile_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    encode(args.input, args.output, args.payload, profile)
