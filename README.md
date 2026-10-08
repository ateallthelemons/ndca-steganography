# NDCA Steganography

A file-hiding tool inspired by **Neighboring block Differences-based Cost Assignment (NDCA)** for video steganography. It hides a file in a video's U chroma plane and recovers its bytes and original filename.

The encoder writes lossless FFV1 video in an MKV container and verifies recovery before saving the result. Recovery after lossy transcoding is not guaranteed.

## Setup

From the project root, create a Python virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The main runtime libraries are PyAV for video I/O, NumPy for arrays and SVD, PyWavelets for wavelet transforms, and `reedsolo` for error correction.

## Project structure

```text
core/
    ecc.py              # Fixed Reed–Solomon codes and byte interleaving
    framing.py          # NDV4/NDV5 framing, integrity, capacity, and legacy parsing
    frame_reader.py     # Automatic bootstrap format detection
    profiles.py         # Immutable coding profiles and CLI settings
    layout.py           # Bootstrap/body frame allocation and exact capacity
    stc.py              # Height-three matrix STC encoder and extractor
    ndca.py             # Wavelet/SVD embedding, costs, and trellis coding
    video_io.py         # YUV planes, video frames, and timestamps
    video_stego.py      # Per-frame capacity, embedding, and extraction
video_encode.py         # Hide a file in a video
video_decode.py         # Recover a hidden file
video_capacity.py       # Calculate maximum file size
requirements.txt
README.md
```

Run the commands below from the project root. The internal modules live in `core/`; they are imported by the command scripts.

## Usage

Create `materials/` if needed and place your own input video and payload there; these files are not included in the repository.

Encoding, saved-video verification, and decoding print progress to stderr roughly
every five seconds as frames finish, showing frame counts, byte progress, and
elapsed time. Each stage also reports when it starts and completes.

### Check capacity

Check how large a file a video can carry with the current embedding settings:

```bash
python video_capacity.py materials/input.mp4
python video_capacity.py materials/input.mp4 --filename message.txt
```

This scans all video frames and accounts for the protected bootstrap, filename,
checksum, Reed–Solomon parity, and codeword padding. Without `--filename`, it
reports the upper bound for a one-byte filename;
with it, the limit is exact for that basename (the payload file need not exist).
This measures storage capacity, not resilience to lossy transcoding.

### Encode

Hide any file type, provided the video has enough capacity. This example uses a text file, but images, PDFs, archives, and other binary files use the same command:

```bash
python video_encode.py materials/input.mp4 materials/message.txt materials/encoded.mkv
```

Use an `.mkv` output filename. The encoder uses FFV1; changing the extension to `.mp4` does not switch codecs. It processes the first video stream and copies audio streams without re-encoding them, subject to output-container compatibility.

Choose an output path distinct from the input video and payload; aliases, including
symlinks and hard links, are rejected. The encoder writes a temporary MKV beside
the destination, closes it, and verifies recovery before moving it into place.
An existing destination is replaced only on success and is preserved on failure.
Verification adds a full decoding pass over the output video.

The default `legacy` profile preserves the original four-state trellis, delta 40,
and CRF costs. To use the validated height-three matrix STC with configurable
body settings:

```bash
python video_encode.py materials/input.mp4 materials/message.txt materials/encoded.mkv --stc matrix --rate 1/2 --delta 40 --cost crf
python video_capacity.py materials/input.mp4 --filename message.txt --stc matrix --rate 1/2 --delta 40 --cost crf
```

Supported matrix STC rates are `1/2` and `1/3`, quantization steps are `30`, `40`,
and `50`, and cost models are `crf` and `cqp`. Use the same settings for capacity
and encoding. Alternatively select a named profile, such as
`--profile stc3-r2-d40-cqp`; named profiles cannot be combined with individual
settings. Supplying individual settings selects matrix STC unless `--stc legacy`
is explicit. Decoding reads the profile from the protected header automatically.
Cost selection describes the anticipated channel; output remains FFV1.

For a stronger experimental compression candidate, explicitly select:

```bash
python video_capacity.py materials/input.mp4 --filename message.txt --profile experimental-robust80
python video_encode.py materials/input.mp4 materials/message.txt materials/encoded-robust80.mkv --profile experimental-robust80
```

`experimental-robust80` uses matrix STC at rate 1/2, delta 80, and CRF costs for
both the header and body. It writes NDV5 with three independently protected
header copies, each in its own group of whole frames. This allows larger pixel
changes and adds header overhead. Recovery after YouTube or other lossy
transcoding has succeeded for one tested YouTube H.264 rendition with the
confidence-guided decoder; the AV1 rendition still fails. This is not yet a
general reliability result. Select it by name; the ordinary
`--delta` options and default profile are unchanged. Decoding detects it without
extra flags. See [NDV5 format](docs/NDV5.md).

### Decode

Choose an explicit destination:

```bash
python video_decode.py materials/encoded.mkv materials/recovered.txt
```

To restore the original filename inside an existing directory:

```bash
mkdir -p materials/decoded
python video_decode.py materials/encoded.mkv materials/decoded
```

Or omit the output argument to restore the original filename in the current directory:

```bash
python video_decode.py materials/encoded.mkv
```

For the example above, the directory form creates `materials/decoded/message.txt`;
omitting the output creates `./message.txt`. Both restore the stored filename and
refuse to overwrite an existing file. An explicit output file path replaces an
existing file at that path. Output directories must already exist.

Ordinary profiles use NDV4; `experimental-robust80` uses NDV5. The decoder repairs the bootstrap before interpreting any
metadata, then repairs and checks the filename and payload before writing.
It reports corrected byte counts for the bootstrap and body; NDV5 also reports
how many of its three header copies were recoverable. When ordinary body RS
decoding fails, the decoder automatically retries with low-confidence bytes
marked as erasures and reports how many codewords needed this fallback. It uses
only the received pixels, requires all successful trials for a word to agree,
and still validates the complete file before writing. This adds no embedded
redundancy and works with existing NDV4/NDV5 videos. Older NDV2/NDV3
videos remain supported without error correction; NDV2 requires an explicit output
path because it has no stored filename. Files are currently limited to 256 MiB
of payload, including legacy recovery, because this implementation buffers data.

## How it works

Before pixel embedding, the encoder protects a fixed bootstrap with RS(96,32).
NDV5 embeds three copies of this protected bootstrap with its stronger mapping;
it requires at least one valid copy and rejects conflicting valid copies.
It protects `filename + payload + CRC32` with RS(60,20), interleaving groups of
up to eight codewords to spread bursts of adjacent byte errors.

1. Decode frames into 8-bit YUV420p and divide the U plane into 16×16 blocks.
2. Apply a Haar wavelet transform to each block and take the largest singular value of its LL subband.
3. Calculate modulation costs from neighboring-block differences and coefficient changes.
4. Use the selected trellis and CRF/CQP cost model to choose carrier bits, then
   quantize using the profile's step. Matrix STC uses an explicit height-three
   parity-check matrix; the legacy profile retains its original four-state code.
5. Test quantization targets after inverse transforms, rounding, and clipping;
   let the trellis choose only recoverable carrier bits. Re-extract each completed
   U plane and reject any chunk that does not match exactly.
6. Encode with FFV1, close and reopen the temporary MKV, and verify the full
   encoded bootstrap, body, parity, and final-chunk padding before publishing
   the file. This check requires exact bytes before RS repair: a newly encoded
   lossless output must not need error correction.

Extraction reverses the trellis mapping, repairs the bootstrap, deinterleaves the
body, applies RS repair, and verifies both integrity checks. NDV4 stores profile
IDs. Profile 1 fixes Haar, 16×16 blocks, `DELTA = 40`, and the current trellis and
carrier mapping for both bootstrap and body. Profiles 2–13 use the same fixed
bootstrap, then switch to the selected body settings at the next frame boundary.
Unsupported profiles are rejected. See [coding settings](docs/CODING.md).

### Capacity

Each block supplies one carrier bit. One block is excluded because it has no
upper or left neighbor. With `B` full blocks in the U plane, the legacy profile
uses `3 × message_bits + 2` carriers:

```text
bytes_per_frame = max(0, (B - 3) // 24)
raw_capacity = frames × bytes_per_frame
required_bytes = 96 + 60 × ceil((filename_utf8_bytes + file_bytes + 4) / 20)
maximum_file_bytes = 20 × floor((raw_capacity - 96) / 60) - filename_utf8_bytes - 4
```

For NDV4 matrix STC at rate `1/w`, body capacity is `floor((B - 1) / (8*w))` bytes per
frame. Reserve `H = ceil(96 / legacy_bytes_per_frame)` complete bootstrap frames.
When `frames >= H`, use `raw_capacity = 96 + (frames - H) × body_bytes_per_frame`
in the formula above. This excludes unused bootstrap-frame slots and accounts for
RS overhead and codeword padding. Fewer than H frames cannot carry a new-profile
file. STC rate counts ECC-encoded bits per carrier; payload utilization is a
separate setting, and the net file rate is lower after RS and metadata.

For NDV5, reserve `3 × ceil(96 / body_bytes_per_frame)` bootstrap frames and
288 logical header bytes. Each copy's spare frame slots are discarded. The capacity
CLI includes these copies and all padding; see [the NDV5 formula](docs/NDV5.md).

The capacity CLI handles videos too small for the bootstrap and caps the result
at the 256 MiB implementation limit. The filename and header are stored once per
file, not once per frame. With 16×16 blocks and the legacy profile:

| Video resolution | Raw embedded bytes per frame (before ECC overhead) |
| --- | ---: |
| 640×360 | 9 |
| 1280×720 | 36 |
| 1920×1080 | 82 |

A negative maximum means even an empty file cannot fit. Payloads are stored without compression or encryption.

### File format and error correction

NDV4 has a fixed 96-byte protected bootstrap followed by the protected body.
The 32-byte bootstrap data (`>4sBBBBQHQ2sI`) contains the magic, embedding and ECC
profile IDs, interleaving depth, flags, payload length, filename length, encoded
body length, reserved bytes, and a CRC32 over its first 28 bytes. RS(96,32) adds
64 parity bytes and can correct up to 32 erroneous bytes in this block.

The body contains the UTF-8 basename, payload, and a big-endian CRC32 over both.
It is zero-padded to 20-byte blocks; RS(60,20) adds 40 parity bytes per block and
can correct up to 20 erroneous bytes per codeword, including errors in parity.
Groups of up to eight codewords are serialized column-first. The last group uses
only its actual codewords, with no extra group padding.

Both codes use GF(256), primitive polynomial `0x11d`, generator 2, and first
consecutive root 0. Correction limits are per-codeword byte counts, not bit counts.
For failed body words, the decoder can also try 1–32 erasures ranked by received
coefficient confidence. The mixed budget is `2 * unknown_errors + erasures <= 40`;
unknown errors count only outside the erased positions. Low-confidence rankings
are heuristic, so this does not guarantee recovery beyond 20 wrong bytes.
Beyond the applicable limits RS may fail or miscorrect; CRC checks remain mandatory.
CRC32 is an accidental-corruption check, not authentication. See
[the NDV4 wire format](docs/NDV4.md) for exact layouts and decoding rules.

Legacy NDV3 uses an unprotected 18-byte `>4sQIH` header followed by filename and
payload; its CRC covers only the payload. NDV2 uses a 16-byte `>4sQI` header and
has no filename. New encoding does not emit these formats.

## Limitations

- Recovery after lossy transcoding is not guaranteed. Resizing, cropping,
  trimming, duplicated or dropped frames, and frame-rate conversion can prevent
  recovery by changing carrier alignment.
- Nominal capacity does not guarantee successful embedding. The encoder rejects
  chunks that cannot be recovered within its bounded pixel and coefficient changes.
- Error correction has finite limits: up to 32 erroneous bytes per bootstrap copy and
  20 unknown erroneous bytes per body codeword. Confidence-guided body erasures
  can recover some additional cases, but may fail or be ambiguous. Legacy NDV2/NDV3 headers are unprotected.
- Inputs need even dimensions, usable timestamps, and sufficient U-plane capacity.
  Midstream resolution changes are unsupported.
- FFV1 output can be large. Conversion to 8-bit YUV420p changes the source pixel
  format and can reduce bit depth. Only the first video stream and compatible
  audio streams are retained; color metadata, subtitles, and other streams are
  not fully preserved.
- Explicit decode file destinations are overwritten without alias checks or atomic
  replacement. Choose a destination distinct from the input video.
- Payloads and filenames are unencrypted. CRC detects accidental corruption and
  does not provide authentication.

## Reference

Jie Gan, Hong Zhang, Xinrui Xie, and Yun Cao (2025). [NDCA: a neighboring block differences-based cost assignment method for robust video steganography on social networks](https://doi.org/10.1186/s13635-025-00214-6).
