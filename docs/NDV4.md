# NDV4 wire format

NDV4 uses a fixed bootstrap followed by a protected body. All integers are
unsigned, big-endian. Data bits within bytes are embedded most-significant first.
Lengths refer to bytes. There is no compression, encryption, or authentication.

## Physical embedding profile 1

Both bootstrap and body use the existing 8-bit YUV420p U-plane mapping: 16×16
blocks, Haar DWT, largest LL singular value, rounded quantization parity with
DELTA=40, and the four-state trellis with `3 * message_bits + 2` carriers.
Carrier positions are the existing evenly spaced block indices excluding block 0.
Per-frame capacity is `max(0, (full_U_blocks - 3) // 24)` bytes. Concatenate
extracted frame chunks in frame order. This profile and the bootstrap RS settings
must be known before reading any metadata. Body profile 1 uses the same mapping.
Changing extraction behavior requires a new profile implementation; the header
cannot bootstrap an arbitrary new physical mapping.

## Configurable body profiles 2–13

The bootstrap always uses profile 1, including its legacy trellis, delta 40, and
CRF embedding costs. Profile 1 retains its original continuous header/body packing.
For IDs 2–13, reserve `ceil(96 / bootstrap_bytes_per_frame)` complete initial
frames for the bootstrap, zero-fill its last frame's spare slots, and start the
body in the following frame with the decoded profile. Discard the spare bootstrap
slots during recovery; they are not body data. Saved-file verification checks
their zeros. Never change extraction settings before repairing the bootstrap.

| ID | Name | STC rate | Delta | Cost |
| --- | --- | --- | ---: | --- |
| 1 | legacy | approaches 1/3 | 40 | CRF |
| 2 | stc3-r2-d30-crf | 1/2 | 30 | CRF |
| 3 | stc3-r2-d30-cqp | 1/2 | 30 | CQP |
| 4 | stc3-r2-d40-crf | 1/2 | 40 | CRF |
| 5 | stc3-r2-d40-cqp | 1/2 | 40 | CQP |
| 6 | stc3-r2-d50-crf | 1/2 | 50 | CRF |
| 7 | stc3-r2-d50-cqp | 1/2 | 50 | CQP |
| 8 | stc3-r3-d30-crf | 1/3 | 30 | CRF |
| 9 | stc3-r3-d30-cqp | 1/3 | 30 | CQP |
| 10 | stc3-r3-d40-crf | 1/3 | 40 | CRF |
| 11 | stc3-r3-d40-cqp | 1/3 | 40 | CQP |
| 12 | stc3-r3-d50-crf | 1/3 | 50 | CRF |
| 13 | stc3-r3-d50-cqp | 1/3 | 50 | CQP |

New profiles keep Haar and 16×16 blocks, with a height-three banded STC matrix.
At rate `1/w`, each message row uses `w` carriers. The column masks are `(7, 5)`
for `w=2` and `(7, 5, 3)` for `w=3`, low bit first. For message row `r`, carrier
column `w*r+j` contributes bit `t` of its mask to syndrome row `r+t` when that
row exists. Columns are truncated at the final message row; the trellis starts
and ends in state zero, without tail carriers. Extraction is `H y mod 2`.
For B full blocks, body bytes per frame are `floor((B-1)/(8*w))`. Select exactly
`w * message_bits` carriers using `np.linspace(1, B-1, count, dtype=int)`.

These fixed matrices are independently implemented and checked against exhaustive
minimum-cost solutions; they are not claimed to be the exact matrices used by
the NDCA authors. See [coding validation](CODING.md).

## Reed–Solomon settings

Both codes are shortened byte-symbol codes over GF(256), primitive polynomial
`0x11d`, generator `2`, first consecutive root `0`. The implementation uses
`reedsolo==1.7.0` with explicit `c_exp=8`, `prim=0x11d`, `generator=2`, `fcr=0`.
Encoding is systematic: data bytes precede parity bytes before interleaving.

- Bootstrap: `RSCodec(64, nsize=96, ...)`, exactly 32 data + 64 parity bytes.
- ECC profile 1 body: `RSCodec(40, nsize=60, ...)`, exactly 20 data + 40 parity bytes.

Bootstrap decoding uses unknown-error correction. Body decoding tries ordinary
correction first, then retries failed words using received-pixel confidence when
available. The byte-only `MessageReader.feed()` API retains ordinary correction
unless confidence is explicitly supplied.
The guaranteed correction limits are 32 and 20 erroneous bytes respectively per
codeword for unknown errors, including errors in parity. Beyond these limits decoding may fail or
miscorrect. CRC checks are therefore mandatory after correction.

For body retries, the receiver measures each coefficient's distance to the nearest
quantization boundary, takes the minimum over the carriers supporting each STC
message bit, then the minimum over eight bits per byte. Confidence follows the
same frame-padding removal and deinterleaving as data. Each failed word gets at
most 32 additional RS trials, erasing its 1–32 least-confident bytes in stable
wire-derived order. All successful trials must yield identical data; otherwise
the word is rejected as ambiguous. A word with no successful trial is rejected.
The mixed correction condition is `2e + s <= 40`, with e the unknown errors
outside s erased positions. This receiver heuristic changes no wire bytes, rates,
profile IDs, or bootstrap behavior. Whole-file integrity checks remain mandatory.

## Bootstrap

The first 96 extracted bytes are one protected bootstrap. RS-decode this block
**before checking its magic**. Its 32 corrected data bytes have this layout:

| Offset | Bytes | Field | Accepted value |
| --- | ---: | --- | --- |
| 0 | 4 | Magic/version | ASCII `NDV4` |
| 4 | 1 | Embedding profile | 1–13, as registered above |
| 5 | 1 | Body ECC profile | 1 |
| 6 | 1 | Body interleaving depth | 8 |
| 7 | 1 | Flags | 0 |
| 8 | 8 | Payload length | 0 through 268,435,456 in this implementation |
| 16 | 2 | UTF-8 basename length | 1 through 65,535 |
| 18 | 8 | Encoded body length | Must equal formula below |
| 26 | 2 | Reserved | Both zero |
| 28 | 4 | CRC32 | `zlib.crc32` of bytes 0–27 |

Python struct: `>4sBBBBQHQ2sI`. Validate the header CRC, identifiers, flags,
reserved bytes, length bounds, and derived body size before consuming the body.
No length field is used to preallocate an output buffer. The 256 MiB payload
limit also applies to legacy recovery and reflects the current buffering design.

## Body

Raw body bytes are:

```text
UTF-8 basename | payload | CRC32(basename bytes + payload)
```

The checksum occupies four big-endian bytes. Zero-pad the final raw block to 20
bytes. Encode every block independently into 60 bytes; even an empty payload
has a filename and checksum, and therefore at least one body codeword.

For each group of up to eight codewords, place codewords in rows of a matrix
with 60 columns and serialize by columns. Thus a two-codeword group `A, B`
becomes `A[0], B[0], A[1], B[1], ..., A[59], B[59]`. The final group has only the
remaining codewords, not eight padded rows. All group sizes derive from the
validated encoded body length; no additional delimiters are present.

Reverse this permutation, RS-decode each codeword, concatenate data bytes, and
validate the raw-body checksum and zero padding. Decode and validate the basename
only after integrity checks. Empty names, `.`/`..`, slash, backslash, NUL, colon,
invalid UTF-8, and overlong names are rejected. Never write a recovered file until
all correction and integrity checks succeed.

## Capacity and padding

For payload length P and UTF-8 basename length N:

```text
body_codewords = ceil((N + P + 4) / 20)
encoded_body_length = 60 * body_codewords
total_message_length = 96 + encoded_body_length
```

For profile 1, C is `frames * bootstrap_bytes_per_frame`. For profiles 2–13,
let H be the reserved bootstrap frame count and S be body bytes per frame:

```text
H = ceil(96 / bootstrap_bytes_per_frame)
C = 96 + (frames - H) * S    # requires frames >= H
```

If there are fewer than H frames, or even `total_message_length(P=0)` does not
fit, no file fits. Otherwise the maximum payload is:

```text
min(268435456, 20 * floor((C - 96) / 60) - N - 4)
```

After framing, zero-pad the last video chunk to full per-frame capacity. This
outer padding is not an RS codeword and is ignored during file recovery.
The saved-FFV1 verification pass checks every framed byte and outer padding
against the encoder's original bytes, before any error correction.
This includes both bootstrap-frame padding and body-frame padding. All frame
allocation calculations are shared through `core/layout.py`.

## Legacy detection and diagnostics

Try repaired NDV4 first once 96 bytes are available. If it fails and the original
first four bytes equal `NDV2` or `NDV3`, try that legacy layout with its original
payload CRC. If the video ends before 96 bytes, legacy parsing is attempted at
EOF. Thus even NDV4 magic corrupted into a legacy magic is repaired as NDV4 when
within the correction limit. Legacy formats have no ECC.

The decoder reports corrected byte counts for the bootstrap, total body, and
maximum per body codeword. Internally `Recovery.body_corrections` stores every
codeword's count of actually changed bytes (excluding correct erased bytes).
`Recovery.body_erasures` stores the first successful erasure count for each word,
or zero when ordinary correction succeeded. The CLI reports fallback usage.
Truncation, unsupported profiles, uncorrectable blocks, invalid
metadata, and failed checksums cause an exception before output is written.
Interleaving addresses adjacent-byte bursts, not alignment loss from frame drops,
resizing, cropping, or reordered frames.
