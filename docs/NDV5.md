# NDV5 experimental robust profile

NDV5 is selected only with `--profile experimental-robust80` (profile ID 14).
It is a compression-resilience candidate, not a guarantee of YouTube recovery.
The production decoder's confidence-guided body RS fallback has recovered the
complete 48,485-byte EPUB from the tested H.264 rendition of `2Mx3Aq-qK0I`.
The AV1 rendition still fails. This is one demonstrated successful round trip;
see the [audit](NDCA_IMPLEMENTATION_AUDIT.md) for the experiment and limitations.
Existing NDV2/NDV3/NDV4 decoding and NDV4 profile IDs 1–13 remain unchanged.

## Fixed physical mapping

The bootstrap and body both use Haar DWT/LL-SVD on 16×16 blocks in the 8-bit
YUV420p U plane. The height-three matrix STC has rate 1/2, columns `(7, 5)`, and
the same carrier ordering and termination as the NDV4 matrix profiles. Delta is
80 and embedding uses CRF costs. These settings are known before reading metadata.
Reconstruction searches within twice delta in coefficient space and limits pixel
changes to delta, as for other profiles. The decoder does not accept a caller's
guessed delta or unvalidated metadata as a mapping instruction.

## Header copies and framing

The 32-byte header structure and RS(96,32) parameters are specified in
[NDV4](NDV4.md). NDV5 changes the magic to ASCII `NDV5` and embedding profile to
14. ECC profile 1, interleaving depth 8, zero flags/reserved bytes, length checks,
and header CRC32 are unchanged. Version and profile must match.

The logical message is:

```text
96-byte protected header | identical header | identical header | protected body
```

Each copy occupies its own group of complete frames. With S bytes per frame,
one group uses H = ceil(96 / S) frames. Copy groups start at frame indices 0, H,
and 2H. Zero-fill each group's last frame's spare slots. The body starts at frame
3H. Copies are in distinct, consecutive frame groups; there is no gap or placement
near the end of the video. This protects against damage localized to one group,
but does not guarantee survival of correlated damage across all three groups.

Repair each copy independently, then validate its CRC, version/profile, lengths,
flags and reserved bytes. Wait for all three groups before consuming body data.
At least one valid copy is required. All valid copies must agree on metadata;
otherwise reject the video. Invalid copies are discarded. Use the valid copy
requiring the fewest RS byte corrections for the reported header correction count;
also report the number of independently recoverable copies. There is no majority
voting over unvalidated header bits.

The RS(60,20) body, interleaving, basename, payload CRC32 and zero padding are
identical to NDV4. The body's declared length excludes all header copies and frame
padding. The receiver can use the confidence-guided erasure fallback documented
in [NDV4](NDV4.md), with no format change. No recovered file is written before
the complete body passes validation.

## Detection and compatibility

The video reader attempts the existing NDV4/legacy mapping and the fixed NDV5
mapping until a header validates. Each attempt buffers only its own extracted
bytes. An invalid first NDV5 copy must not stop the remaining copies from being
tried. Once a header validates, its reader owns the rest of the extraction; body
failures remain failures. Legacy short messages are finalized at end of video.

Older decoders cannot read NDV5. Use the updated decoder for the upload trial.
Resizing, trimming, dropped/duplicated frames and frame reordering are unsupported.

## Exact capacity and lossless verification

For F frames, B full U-plane blocks and UTF-8 filename length N:

```text
S = floor((B - 1) / 16)
H = ceil(96 / S)
reserved_bootstrap_frames = 3 * H
C = 288 + (F - 3*H) * S       # requires F >= 3*H and S > 0
encoded_size = 288 + 60 * ceil((N + payload_bytes + 4) / 20)
maximum_payload = min(268435456, 20 * floor((C - 288) / 60) - N - 4)
```

If the reserved frames or even an empty file do not fit, no file fits. The encoder
checks reconstructed pixels and verifies every byte of all three copies, body,
and outer padding from the saved FFV1 file. A lossless output is rejected even
when RS or another header copy could conceal an embedding error.
