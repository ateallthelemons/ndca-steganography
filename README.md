# NDCA Steganography

Hide a file in a video's U chroma plane and recover its bytes and original filename.
The tool uses one NDV5 pipeline: Haar/LL-SVD, height-three matrix STC at rate 1/2,
delta 80, CRF costs, repeated protected headers, and confidence-guided Reed–Solomon
recovery. It targets resilience to YouTube recompression.

The current decoder recovered a complete 48,485-byte file from one tested YouTube
H.264 rendition. The AV1 rendition failed. This is a demonstrated recovery, not a
general reliability guarantee. Resizing, cropping, trimming, or changing frame
alignment can prevent recovery.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Runtime dependencies are
PyAV, NumPy, and reedsolo. Run commands from the project root.

## Usage

```bash
python video_capacity.py input.mp4 --filename message.txt
python video_encode.py input.mp4 message.txt encoded.mkv
python video_decode.py encoded.mkv recovered.txt
```

No profile flags are needed. The capacity command scans all frames and includes
filename, checksums, RS parity, repeated headers, and frame padding. Without
`--filename` it reports an upper bound for a one-byte filename. Capacity does not
guarantee that every cover can embed successfully or survive recompression.

Encoding writes lossless FFV1 in an MKV container; use an `.mkv` output path.
It preserves the first video stream's timestamps and copies compatible audio
streams. The destination must differ from the input video and payload, including
symlink and hard-link aliases. A temporary output is closed, reopened, and checked
for exact embedded bytes before replacing the destination. Failed encoding or
verification preserves an existing destination.

To restore the stored filename, give an existing output directory or omit the
output argument to use the current directory. These forms refuse to overwrite an
existing file. An explicit output filename replaces that file after successful
recovery. Choose a destination distinct from the input video; explicit decode
output does not currently have alias protection or atomic replacement.

Encoding, verification, and decoding use all CPU cores by default. Pass
`--workers N` to encode/decode to limit processes, or `--workers 1` for serial
execution. Progress is written to stderr about every five seconds as frames finish.

## Algorithm

1. Convert frames to 8-bit YUV420p; divide U into 16×16 blocks.
2. Compute each block's largest Haar LL singular value and upper/left neighbor
   differences. Exclude the first block, which has no such neighbor.
3. Use CRF neighbor costs and matrix STC to choose quantized carrier bits.
4. Check candidate bits after reconstruction, rounding, and clipping; reject
   infeasible chunks. Re-extract every completed embedded plane.
5. Verify every encoded header, body, parity byte, and frame-padding byte in the
   saved FFV1 output before publishing it.

The receiver repairs independent header copies, then deinterleaves and repairs
the body. Failed body codewords get bounded retries using received coefficient
confidence as erasure rankings. Conflicting repairs and failed integrity checks
are rejected before writing a recovered file.

If the existing video decoder cannot repair a body word, it makes one additional
pass with a likelihood-based confidence ranking. Successful first-pass results
are returned immediately. This receiver-only fallback preserves NDV5 files and
payload capacity; see [confidence recovery](docs/DECODER_CONFIDENCE.md) for the
model, validation and limits.

See [coding](docs/CODING.md), [NDV5 format](docs/NDV5.md), and
[parallel processing](docs/PARALLEL.md) for implementation details.

## Structure

```text
core/config.py        Current file mapping and internal numerical settings
core/ndca.py          Batched Haar/SVD, carrier positions, neighbor costs
core/stc.py           General height-three matrix STC
core/video_stego.py   Reconstruction, embedding, extraction, confidence
core/ecc.py           Reed–Solomon and interleaving
core/framing.py       Protected metadata, integrity, incremental reader
core/layout.py        Shared header/body frame allocation
core/video_io.py      YUV planes and timestamps
core/parallel.py      Ordered, bounded per-frame multiprocessing
core/progress.py      Progress reporting
video_encode.py       Encode and verify a file
video_decode.py       Recover a file
video_capacity.py     Calculate exact file capacity
```

## Limits

- Recovery depends on the transcoding conditions. The tested YouTube H.264
  rendition recovered exactly; AV1 did not. Resizing, cropping, and dropped or
  duplicated frames can disrupt carrier alignment, which the current decoder
  does not restore.
- Files are limited to 256 MiB because messages are buffered in memory.
- Inputs need even dimensions, sufficient U-plane capacity, and usable timestamps.
  Midstream resolution changes are unsupported.
- FFV1 output can be large. Conversion to 8-bit YUV420p can reduce source bit depth.
  Subtitles, other video streams, and full color metadata are not preserved.
- Filenames and payloads are neither compressed nor encrypted. CRC32 detects
  accidental corruption; it is not authentication.

## Reference

Jie Gan, Hong Zhang, Xinrui Xie, and Yun Cao (2025).
[NDCA: a neighboring block differences-based cost assignment method for robust video steganography on social networks](https://doi.org/10.1186/s13635-025-00214-6).
The implementation follows the paper's structure; it is not a byte-exact
reproduction of the authors' experimental setup.
