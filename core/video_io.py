import av
import numpy as np
from fractions import Fraction


def read_plane(plane):
    data = np.frombuffer(plane, dtype=np.uint8)
    return data.reshape(plane.height, plane.line_size)[:, :plane.width].copy()


def write_plane(plane, data):
    if data.shape != (plane.height, plane.width):
        raise ValueError('Unexpected plane dimensions')
    buffer = bytearray(plane.buffer_size)
    padded = np.frombuffer(buffer, dtype=np.uint8).reshape(plane.height, plane.line_size)
    padded[:, :plane.width] = data
    plane.update(bytes(buffer))


def unpack_video_frame(frame):
    frame = frame.reformat(format='yuv420p')
    return tuple(read_plane(p) for p in frame.planes[:3])


def make_video_frame(Y, U, V, pts, time_base):
    h, w = Y.shape
    frame = av.VideoFrame(w, h, format='yuv420p')
    for plane, data in zip(frame.planes, (Y, U, V)):
        write_plane(plane, data)
    frame.pts = pts
    frame.time_base = time_base
    return frame


def rescale(value, from_base, to_base):
    return round(Fraction(value) * from_base / to_base)


def video_frames(filename):
    with av.open(filename) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield unpack_video_frame(frame)
