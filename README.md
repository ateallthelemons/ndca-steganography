# NDCA Steganography

A personal, experimental implementation inspired by **Neighboring block Differences-based Cost Assignment (NDCA)** for video steganography. It hides a file in a video's U chroma plane and recovers its bytes and original filename.

The long-term goal is recovery after lossy compression, including social-network transcoding. The current implementation writes lossless FFV1 video in an MKV container; reliable recovery after lossy transcoding is not yet established.

## Setup

From the project root, create a Python virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The main runtime libraries are PyAV for video I/O, NumPy for arrays and SVD, and PyWavelets for wavelet transforms.

## Project structure

```text
core/
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

### Check capacity

Check how large a file a video can carry with the current embedding settings:

```bash
python video_capacity.py materials/input.mp4
python video_capacity.py materials/input.mp4 --filename message.txt
```

This scans all video frames for an accurate count and subtracts header and filename
overhead. Without `--filename`, it reports the upper bound for a one-byte filename;
with it, the limit is exact for that basename (the payload file need not exist).
This measures storage capacity, not resilience to lossy transcoding.

### Encode

Hide any file type, provided the video has enough capacity. This example uses a text file, but images, PDFs, archives, and other binary files use the same command:

```bash
python video_encode.py materials/input.mp4 materials/message.txt materials/encoded.mkv
```

Use an `.mkv` output filename. The encoder uses FFV1; changing the extension to `.mp4` does not switch codecs. It processes the first video stream and copies audio streams without re-encoding them, subject to output-container compatibility.

Choose a new output path, distinct from the input video and payload. The encoder opens the output for writing and removes it if encoding fails.

### Decode

Choose an explicit destination:

```bash
python video_decode.py materials/encoded.mkv materials/recovered.txt
```

Or omit the output argument to restore the original filename in the current directory:

```bash
python video_decode.py materials/encoded.mkv
```

For the example above, this creates `./message.txt`. Automatic restoration refuses to overwrite an existing file. An explicit output path replaces an existing file at that path. Output directories must already exist.

The decoder verifies the payload's CRC32 before writing it. Older NDV2 videos remain supported but require an explicit output path because their headers do not contain a filename.

## How it works

1. Decode frames into 8-bit YUV420p and divide the U plane into 16×16 blocks.
2. Apply a Haar wavelet transform to each block and take the largest singular value of its LL subband.
3. Calculate modulation costs from neighboring-block differences and coefficient changes.
4. Use an educational four-state syndrome-trellis code to choose carrier bits, then quantize the selected singular values with a step of 40.
5. Reconstruct the modified U plane and encode the video with FFV1. Extraction reverses the bit mapping and checks the recovered payload.

The embedding settings are defined in `core/ndca.py` and must match during encoding and decoding; they are not stored in the header.

### Capacity

Each block supplies one carrier bit, and the trellis needs `3 × message_bits + 2` carriers. One block is excluded because it has no upper or left neighbor. With `B` full blocks in the U plane:

```text
bytes_per_frame = max(0, (B - 3) // 24)
maximum_file_bytes = frames × bytes_per_frame - 18 - filename_utf8_bytes
```

The filename and header are stored once per file, not once per frame. With the current 16×16 blocks:

| Video resolution | Embedded bytes per frame |
| --- | ---: |
| 640×360 | 9 |
| 1280×720 | 36 |
| 1920×1080 | 82 |

A negative maximum means even an empty file cannot fit. Payloads are stored without compression or encryption.

### File header

The NDV3 header contains the magic, payload byte length, payload CRC32, and UTF-8
filename byte length (`>4sQIH`), followed by the filename and the raw file bytes.
Only the basename is stored. The fixed header uses 18 bytes; filename bytes also
count toward the video's capacity. CRC32 checks the payload, not the filename.

## Current limitations

- Lossy recompression can corrupt hidden data. There is no error-correction layer; the trellis minimizes embedding cost, and CRC32 only detects payload corruption.
- Resizing, cropping, dropping frames, or changing frame order can prevent recovery.
- Input must have even dimensions, usable frame timestamps, and enough U-plane capacity. Midstream resolution changes are unsupported.
- FFV1 output can be substantially larger than the input video. Frames are converted to 8-bit YUV420p, so original bit depth and pixel format are not preserved.
- The implementation is a learning prototype, not a complete reproduction or validation of the paper's results.

## Reference

Jie Gan, Hong Zhang, Xinrui Xie, and Yun Cao (2025). [NDCA: a neighboring block differences-based cost assignment method for robust video steganography on social networks](https://doi.org/10.1186/s13635-025-00214-6).
