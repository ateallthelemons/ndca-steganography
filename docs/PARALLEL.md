# Multi-core frame processing

Encoding, saved-video verification, and decoding used to handle one frame at a
time on one CPU core. Each frame's work is independent of every other frame, so
the tools now spread frames across all available cores. The embedding maths,
coding profiles, and on-disk format are unchanged. A video encoded with any
number of workers has identical pixels and decodes with any version.

Tracked in issue #1, branch `1-parallelisation`.

## Why frames are independent

Which bytes go into which frame depends only on the frame's shape, the
message length, the frame index, and the profile. `frame_chunk()` in
`core/layout.py` computes this without looking at any other frame:

```python
plan = frame_chunk(U.shape, len(message), index, profile)
part = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
```

`embed_chunk(U, part, profile)` and `extract_chunk(U, profile)` then only touch
that frame's U plane. Nothing carries over between frames, so different frames
can run in different processes at the same time.

What has to stay in order is reading the input video, writing the output video,
and feeding extracted bytes to the decoder's `MessageReader`. Those steps stay
in the main process.

## Overview

```text
main process                         worker processes (one per core)
------------                         -------------------------------
read frame 1 ──► send U plane ─────► embed_chunk ─┐
read frame 2 ──► send U plane ─────► embed_chunk ─┤  run at the same time
read frame 3 ──► send U plane ─────► embed_chunk ─┤
   ...                                             │
write frame 1 ◄── results come back in frame order ┘
write frame 2 ◄──
   ...
```

Only the U plane goes to a worker. The Y and V planes and the timestamp stay in
the main process until the frame is written.

## New module: `core/parallel.py`

The module has one helper, `ordered_map`. It takes a stream of tasks and yields
their results **in the order the tasks were given**, however the workers finish:

```python
def ordered_map(tasks, workers=None):
    """Yield (context, result) for each (function, args, context) task, in order."""
```

Each task is a tuple `(function, args, context)`:

- `function(*args)` runs in a worker process. It must be a module-level function
  so it can be pickled, which is why the encoder adds `embed_frame` (see below).
- `context` never leaves the main process. It holds what the caller needs once
  the result arrives, such as the Y/V planes and timestamp of the frame.
- If `function` is `None`, the result is `None` and no worker is used. Frames
  after the end of the message use this: they still have to be written, but
  there is nothing to embed.

Behaviour:

- **Order:** results are kept in a queue and released only from its front, so
  frame *n* is never released before frame *n − 1*.
- **Memory bound:** at most `2 × workers` tasks are in flight at once. The main
  process stops reading new frames until the oldest one finishes, so a long video
  is never loaded into memory all at once.
- **Lazy input:** `tasks` can be a generator. Frames are decoded from the input
  video only as space opens up in the queue.
- **Errors:** an exception raised in a worker is re-raised in the main process
  when that frame's result is reached. The `finally` block shuts the pool down and
  cancels pending work, including when the caller stops early (the decoder does
  this once it has the whole message).
- **Single process:** with `workers <= 1` no pool is created and everything runs
  inline, as before. This is useful for debugging and profiling.

`default_workers()` returns `os.cpu_count()`, or 1 if that is unknown.

## Encoder: `video_encode.py`

### `embed_frame` (new)

A module-level wrapper that workers can run:

```python
def embed_frame(U, part, profile, number):
    """Embed one frame's chunk; runs in a worker process."""
    try:
        U, _ = embed_chunk(U, part) if profile == LEGACY else embed_chunk(U, part, profile)
        return U
    except EmbeddingError as error:
        raise EmbeddingError(f'Embedding failed at frame {number}: {error}') from error
```

It returns only the modified U plane. `embed_chunk` also returns the trellis
cost, which the encoder never used. The frame number is passed in because the
worker has no other way to know it, and the error message keeps the same wording
as before.

### `encode`

The demux/decode loop moved into an inner generator `tasks()`. For each video
frame it does the same checks as before (resolution, capacity, timestamp), then
yields one task:

```python
function, args = None, ()
if total < len(message):
    plan = frame_chunk(U.shape, len(message), index, profile)
    part = message[plan.offset:plan.offset+plan.length].ljust(plan.size, b'\0')
    function, args = embed_frame, (U, part, plan.profile, index + 1)
    total += plan.length
yield function, args, (Y, U, V, pts)
```

Audio packets are still muxed straight away inside the generator, in the same
place as before. The main loop then writes the frames in order:

```python
for (Y, U, V, pts), embedded in ordered_map(tasks(), workers):
    encoded = make_video_frame(Y, U if embedded is None else embedded, V, pts, time_base)
    for out_packet in output_video.encode(encoded):
        target.mux(out_packet)
    frames += 1
```

`total` now counts bytes *scheduled* rather than bytes already embedded, because
scheduling runs ahead of the workers. It still reaches `len(message)` only if
every byte was assigned to a frame, so the "Video too short" check is unchanged.
A worker error stops the run before any later frame is written. As before, the
temporary file is deleted and the destination is left untouched.

`encode()` takes a new keyword argument `workers=None` (all cores).

### `verify_saved_video`

This now runs in parallel too. Its inner `tasks()` generator reads the saved
file, checks dimensions, and schedules `extract_chunk(U, plan.profile)` for each
frame that holds message bytes. It passes the expected bytes as the context. The
main loop compares each result to the expected bytes in frame order, so a
failure still names the first bad frame. The frame-count and byte-count checks at
the end are unchanged. It also takes `workers=None`.

## Decoder: `video_decode.py`

The decoder cannot run in parallel from the first frame. The profile for the
body frames is stored in the NDV4 bootstrap header, which is spread across the
first few frames and always uses the legacy profile. `recover()` therefore runs in
two phases:

1. **Bootstrap, in order.** Frames go to `reader.feed_frame(U)` one at a time, as
   before, until the reader has parsed a header (`reader.header`) or recognised a
   legacy NDV2/NDV3 message (`reader.legacy`). `feed_frame` already handles the
   frame on which the header ends, including the profile-1 vs other-profile
   difference in how its spare bytes are used.
2. **Body, in parallel.** The profile is now known. Every later frame is
   extracted on the workers and its bytes go to `reader.feed()` in order:

```python
profile = get_profile(reader.header.profile_id) if reader.header is not None else LEGACY
tasks = ((extract_chunk, (U, profile), None) for U in planes)
for _, chunk in ordered_map(tasks, workers):
    result = reader.feed(chunk)
    if result is not None:
        return result
```

This matches what `feed_frame` does once a header is known: it calls
`self.feed(extract_chunk(plane, profile))`. The reader's state machine and the
error correction and integrity checks are not changed. When the message is
complete, `recover` returns straight away. Leaving the loop closes `ordered_map`,
which cancels the extra frames that were scheduled in advance.

`recover()` and `decode()` take a new keyword argument `workers=None`.

## Command-line option

`video_encode.py` and `video_decode.py` accept:

```bash
--workers N   # CPU processes for per-frame work (default: all cores; 1 disables)
```

Examples:

```bash
python video_encode.py materials/in.mp4 materials/secret.pdf out.mkv            # all cores
python video_encode.py materials/in.mp4 materials/secret.pdf out.mkv --workers 8
python video_decode.py out.mkv --workers 1                                       # single process
```

`video_capacity.py` does no embedding work and is unchanged.

## Validation

Tested on a synthetic 640×360, 120-frame FFV1 clip on a 32-core machine.

| Test | Result |
|---|---|
| Legacy profile, 300-byte payload: encode time | 16.7 s → 1.7 s |
| Legacy profile: decode time | 1.9 s → 0.7 s |
| `stc3-r2-d40-crf`, 400-byte payload: encode time (`--workers 1` vs all) | 15.0 s → 1.8 s |
| Decoded pixels of every frame, old vs new encoder | identical |
| Decoded pixels, `--workers 1` vs all cores | identical |
| Payload round trip (both profiles, both worker settings) | byte-exact |
| Payload too large for the video | same `ValueError`, no output file left behind |

The output `.mkv` files are not byte-identical between runs, but neither were
two runs of the original code. The difference is MKV container metadata near the
start of the file, not frame data. Compare decoded frames, not file hashes.

The speed-up grows with video length and resolution, since there are more frames
to keep every core busy. On short clips, starting the worker processes takes a
noticeable share of the time.

## Limits and possible next steps

- **Video I/O is still single-threaded.** Reading the input and FFV1 encoding run
  in the main process. On a machine with many cores this may become the
  bottleneck.
- **Bootstrap frames are decoded in order.** These are only the first few frames,
  so the cost is small.
- **Per-frame work is still slow Python.** Each worker still loops over blocks one
  at a time (`calculate_costs`, `reconstruct_carrier`, `extract_chunk`). Batching
  these with NumPy would speed up every worker, and the gain multiplies with the
  multi-core speed-up.
- **GPU.** Not needed for now. GPU floating-point results can differ slightly
  from the CPU's, which could flip bits whose coefficient sits on a rounding
  boundary, so extraction should stay on the CPU.
