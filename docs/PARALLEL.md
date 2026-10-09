# Multi-core and batched frame processing

Encoding, saved-video verification, and decoding used to handle one frame at a
time on one CPU core, and within each frame one 16×16 block at a time. Two
independent changes speed this up:

1. **Multi-core:** frames are independent of each other, so the tools spread
   them across all available cores (`core/parallel.py`).
2. **Batched maths:** within a frame, the per-block wavelet and SVD work is done
   for all blocks in a few NumPy calls, not thousands of small ones
   (`core/ndca.py`, `core/video_stego.py`). See
   [Batched per-frame maths](#batched-per-frame-maths).

The embedding maths, coding profiles, and on-disk format are unchanged. Output
pixels are bit-identical to the original code for any number of workers, and
videos made before these changes decode as before.

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

The decoder first detects the bootstrap mapping through `FrameReader`, which
tries the legacy/NDV4 mapping and the NDV5 robust mapping. `recover()` runs in
two phases:

1. **Bootstrap, in order.** Frames go to `reader.feed_frame(U)` until a reader
   is selected or the message completes. This preserves repeated NDV5 headers,
   bootstrap padding, and profile switching. Legacy messages can finish here.
2. **Body, in parallel.** Once selected, the header determines the body profile.
   Workers run `extract_chunk_with_confidence(U, profile)`. Both bytes and
   per-byte confidence scores go to the selected reader's `feed()` in frame
   order, preserving confidence-guided Reed–Solomon repair and integrity checks.

When recovery finishes early, a `finally` block explicitly closes `ordered_map`
to shut down the pool and cancel queued work. End-of-video validation still
uses `FrameReader.finish()`.

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

## Batched per-frame maths

### Where the time went

Profiling one 1080p frame (`stc3-r2-d40-crf`) showed that the old code spent
almost all its time on Python overhead, not arithmetic:

- `calculate_costs` looped over all ~2,000 blocks, calling `pywt.dwt2` and
  `np.linalg.svd` on each.
- `reconstruct_carrier` ran for each carrier and each bit value. It tried up to
  four targets, and every attempt did a DWT, SVD, inverse DWT, rounding, and a
  second DWT and SVD: about 12,000 small library calls per frame.
- `extract_chunk` looped over the carriers with one DWT and SVD each.

Each block is only 16×16 (an 8×8 LL band), so the fixed cost of each call
outweighed the work it did.

### Bit-exactness requirement

A bit is read as `rint(S[0] / delta) % 2`. If a rewrite changed `S[0]` by even
one floating-point rounding step, a coefficient sitting on a rounding boundary
could read the other way, and videos encoded by the old code might stop decoding.
So the batched code reproduces the original arithmetic **exactly**, not just
approximately:

- **Haar transform.** For Haar, `pywt` computes `even*s + odd*s` (low band) and
  `even*s - odd*s` (high band) with `s = sqrt(0.5)`, along the rows first and
  then the columns. The inverse reverses this, columns first. The batched
  functions do the same operations in the same order on whole stacks of blocks.
- **SVD and matrix product.** `np.linalg.svd` on a stack of matrices runs the
  same LAPACK routine on each one, and a stacked `@` gives the same result as one
  2-D product per block. Both were checked to be bit-identical to the per-block
  calls.

### New helpers in `core/ndca.py`

They work on arrays of shape `(..., 16, 16)` and match the single-block
functions, which are kept as the reference:

| Batched helper | Matches |
|---|---|
| `image_blocks(image)` | view of the plane as a `(rows, cols, 16, 16)` grid |
| `haar_dwt2(blocks)` | `pywt.dwt2(block, 'haar')` |
| `haar_idwt2(LL, details)` | `pywt.idwt2((LL, details), 'haar')` |
| `get_coefficients(blocks)` | `get_coefficient(block)` |
| `decode_bits(values, delta)` | `decode_bit(value, delta)` |

`calculate_costs` now computes all block coefficients in one call. It also
computes the neighbour differences with array slices, comparing each block with
the one above (`blocks[1:] - blocks[:-1]`) and the one to its left:

```python
blocks = image_blocks(image)
coefficients = get_coefficients(blocks)

upper = np.mean(np.abs(blocks[1:] - blocks[:-1]), axis=(2, 3))
left = np.mean(np.abs(blocks[:, 1:] - blocks[:, :-1]), axis=(2, 3))
differences = np.full((rows, cols), np.nan)
differences[1:, 0] = upper[:, 0]
differences[0, 1:] = left[0, :]
differences[1:, 1:] = (upper[:, 1:] + left[1:, :]) / 2
```

The top-left block has no neighbours and stays `NaN`, as before.

### `reconstruct_carriers` in `core/video_stego.py`

This replaces the per-carrier loop over `reconstruct_carrier` (the single-block
function stays as the reference). The sequential version, for one carrier and
one bit:

1. lists the targets `i * delta` within `±2·delta` of the coefficient whose
   index `i` has the right parity,
2. sorts them by distance, ties going to the smaller target,
3. returns the first target whose rebuilt, rounded pixels change by at most
   `delta` and still decode to the bit.

The batched version tries **every** target for **every** carrier at once, then
picks the same one:

- At most five target indices fit in the `±2·delta` window, so each carrier gets
  five slots `low, low+1, …, low+4`. Slots past `high` are masked off.
- One SVD per carrier gives `U, S, Vt`. `S[0]` is replaced by each slot's target,
  giving an `(n, 5, 8, 8)` stack of new LL bands. One inverse Haar call then
  rebuilds all `n × 5` candidate blocks.
- Each candidate is marked usable if it is in range, changes no pixel by more
  than `delta`, and decodes to its own parity.
- For each bit, `argmin` over the distance of the usable candidates of that
  parity picks the winner. Slots are in ascending target order and `argmin`
  returns the first minimum, so ties go to the smaller target, as before.

It returns the chosen pixels, the movement, and a `found` mask for both bits of
every carrier. `embed_chunk` builds the trellis costs from these in one step, and
carriers with no usable candidate get `+inf`, as before:

```python
keep = np.arange(2) == cover[:, None]
costs = np.where(keep, movement * factors[:, None], movement / factors[:, None])
costs[~found] = np.inf
```

The chosen blocks are written back with explicit pixel indices, not through
`image_blocks(result)`. That reshape is a copy, not a view, when the plane size
is not a multiple of 16, and writes into a copy would be lost.

`extract_chunk` now gathers the carrier blocks and decodes them in one call:

```python
blocks = image_blocks(image).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)[positions]
carrier = decode_bits(get_coefficients(blocks), profile.delta)
```

The trellis (`stc.embed`, `stc_encode`) is unchanged. Each step depends on the
previous one, so it cannot be batched the same way. It takes about 0.03 s per
frame, so it is not worth optimising yet.

### Validation

- Old and new `calculate_costs` on 10 frames of a 1080p video, both cost models:
  identical (`NaN` compared as equal).
- Old and new `embed_chunk`/`extract_chunk` on 5 frames (one cropped to 263×475,
  not a multiple of 16) × all 13 profiles: identical output pixels, trellis
  costs and extracted bytes.
- Full encode of a 1080p, 1,437-frame video with an EPUB payload, on 1 and on 32
  workers: every frame's pixels identical to the original code's output. All
  decodes recover the payload byte-exact, including decoding a video encoded by
  the original code.
- Per 1080p frame, embedding went from 1.62 s to 0.25 s and extraction from
  0.11 s to 0.02 s.

## Limits and possible next steps

- **Video I/O is now the bottleneck on many cores.** Reading the input and FFV1
  encoding run in the main process alone. With batched maths, each frame is fast
  enough that workers spend time waiting for it. Encoding several frames ahead,
  or letting FFmpeg use threads for decoding and FFV1 encoding, may help.
- **Bootstrap frames are decoded in order.** These are only the first few frames,
  so the cost is small.
- **The remaining per-frame time is mostly SVD itself.** Further gains would need
  a cheaper way to get `S[0]` (e.g. from the eigenvalues of `LLᵀLL`), but that
  changes the floating-point results and would have to be proven never to flip
  a decoded bit.
- **GPU.** Not needed for now. GPU floating-point results can differ slightly
  from the CPU's, which could flip bits whose coefficient sits on a rounding
  boundary, so extraction should stay on the CPU.
