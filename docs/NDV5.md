# NDV5 wire format

All integers are unsigned and big-endian; data bits within bytes are embedded
most-significant first. Lengths are in bytes. The file tools use only this mapping.

## Physical mapping

Header and body use the 8-bit YUV420p U plane, Haar DWT on 16×16 blocks, the largest
LL singular value, rounded quantization parity at the explicitly selected delta
(`base`: 80; `robust`: 120), and height-three matrix
STC at rate 1/2. Column masks are `(7, 5)`, low bit first. For message row r,
carrier `2*r+j` contributes bit t of mask j to syndrome row r+t when that row
exists. Columns truncate at the final row. Initial and terminal trellis states are
zero; there are no extra tail carriers. Extraction is `H y mod 2`.

For B full blocks, each frame holds S = floor((B-1)/16) bytes. Choose exactly
16*S carriers using `np.linspace(1, B-1, 16*S, dtype=int)`. Block zero is excluded.
These settings must be supplied before decoding metadata; no guessed mapping is accepted.
The profile name/delta is not encoded in the bootstrap. Both supported profiles
retain embedding ID 14 and identical framing for compatibility with tested files.
The caller must retain the profile separately and use it for both encoding and
decoding; ID 14 alone does not distinguish the two strengths.

## Header groups

The logical message is three identical 96-byte protected headers followed by the
protected body. Each header occupies H = ceil(96/S) whole frames. Groups start at
frame indices 0, H, and 2H; the body starts at 3H. Zero-fill the unused slots of
each group's last frame and discard those slots during recovery.

Repair and validate all three copies independently. At least one must be valid;
all valid copies must agree on metadata. Reject conflicts. Report the number of
valid copies and the fewest corrected bytes among them. Consecutive groups do not
guarantee survival of correlated damage across all copies.

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

The video decoder first uses the minimum-margin policy above. Only an
uncorrectable body word (no candidate, not an ambiguous repair) triggers one
additional video pass using likelihood confidence. CRC, padding, header,
truncation and I/O errors do not trigger this fallback. It retains the same hard
decisions, RS parameters, 1–32 erasure trials per failed word, and ambiguity and
integrity checks. The alternative score uses a Gaussian lattice model with fixed
sigma=delta/2, propagates carrier reliability through the STC XOR supports, and
multiplies approximate bit reliabilities within each byte. It is a heuristic
ranking, not a calibrated probability. The selected delta is used in both passes.
Byte-only readers retain their
existing behavior; the video-level retry requires re-reading received pixels.

## Bootstrap

Each 96-byte copy is one protected bootstrap. RS-decode this block
**before checking its magic**. Its 32 corrected data bytes have this layout:

| Offset | Bytes | Field | Accepted value |
| --- | ---: | --- | --- |
| 0 | 4 | Magic/version | ASCII `NDV5` |
| 4 | 1 | Embedding profile | 14 |
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
No length field is used to preallocate an output buffer. Encoding and decoding
stream groups of at most eight RS words (160 raw / 480 encoded bytes). The final
group may be shorter. CRC32 is updated incrementally, and decoded CLI output
is published atomically from a temporary file only after CRC and padding checks
pass. Confidence scores are retained only for the current group. The former
256 MiB implementation limit is removed; both wire lengths must fit unsigned
64-bit fields. `MessageWriter` reads from a caller-owned binary source;
`MessageReader(sink)` requires a binary destination. `recover(..., sink=...)`
requires a seekable destination so a confidence retry can reset provisional
output. Recovery returns metadata and aggregate repair statistics, not payload
bytes or per-codeword histories. The former whole-message Python APIs are
removed; the NDV5 wire format is unchanged.

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

For F frames, B full U blocks, N UTF-8 filename bytes, and P payload bytes:

```text
S = floor((B - 1) / 16)
H = ceil(96 / S)
encoded_body_length = 60 * ceil((N + P + 4) / 20)
total_message_length = 288 + encoded_body_length
C = 288 + (F - 3*H) * S
maximum_payload = min(268435456, 20 * floor((C - 288) / 60) - N - 4)
```

Require S > 0 and F >= 3H. If even an empty payload does not fit, no file fits.
Zero-pad the last body frame to its full chunk capacity. This outer padding is
ignored during recovery. Saved-FFV1 verification requires exact bytes before RS
repair, including all header copies, their frame padding, body parity, and final
frame padding. Frame allocation is shared through `core/layout.py`.

The decoder reports corrected byte counts, recoverable header copies, and erasure
fallback usage. Truncation, unsupported metadata, uncorrectable data, ambiguity,
and failed integrity checks prevent output. Interleaving does not address frame
drops, resizing, cropping, or reordered frames.
