# Coding profiles and validation

The encoder retains `legacy` by default. Configurable profiles use a binary
syndrome-trellis code with an explicit height-three banded parity-check matrix.
The implementation follows the matrix/trellis construction described by Filler,
Judas, and Fridrich; see the [authors' STC toolbox and publications](https://dde.binghamton.edu/download/syndrome/).
It is an independent Python implementation, not a wrapper or a byte-for-byte port
of that toolbox. The specific fixed matrices are part of this project's wire
format; matching NDCA's h=3 and rate=0.5 does not reproduce all its experiments.

## Optimization and termination

For candidate bits y, message m, and absolute per-candidate costs rho:

```text
minimize sum(rho[i, y[i]]) subject to H y = m over GF(2)
```

Both zero and one may have nonzero costs. +Infinity forbids a candidate that
cannot survive reconstruction within the embedding limits. If there is no finite
path, fail explicitly before backtracking. NaN and negative costs are rejected.
No assumption is made that the original carrier bit is the cheapest choice.

Height h=3 uses eight syndrome states. After each group of w carriers, constrain
the low state bit to the corresponding message bit and shift the state right.
Columns truncate at the last syndrome row. Initial and terminal states are zero;
there are no extra termination carriers. Each bit contributes to at most three
syndrome rows. This is source coding for embedding, not an error-correcting
channel code. RS remains the error-correction layer.

The legacy encoder has separate exhaustive tests for its original four-state,
`3*m+2` carrier construction. Keeping that implementation preserves old extraction
and gives a measured baseline. Tests for both implementations enumerate all
candidate vectors on small problems and compare feasibility, extracted syndromes,
reported costs, and globally minimal total costs, including forbidden candidates.
The matrix test also fixes an explicit expected H independently of extraction.

## Cost models

The formulas follow [NDCA Eqs. 15–17](https://link.springer.com/article/10.1186/s13635-025-00214-6).
For each block, compute mean absolute pixel differences from its upper and/or
left neighbor, averaging the two when both exist. Subtract using floating point.
Block zero has no neighbor and is excluded. Normalize over all eligible full
blocks in the frame **before carrier selection**:

```text
d = (difference - minimum) / (maximum - minimum + 1e-6)
f_crf = 1 - d + 1e-6
f_cqp = d + 1e-6
keep_cost = actual_keep_target_movement * f
flip_cost = actual_flip_target_movement / f
```

A flat frame yields d=0. The epsilon prevents division by zero in CQP; this can
make flipping expensive, but never silently makes an infeasible candidate usable.
Boundary blocks use their one available neighbor. Cost profiles describe the
anticipated channel; `--cost cqp` does not change the FFV1 output codec.

The reconstruction checks are a project-specific extension: try same-parity,
nonnegative quantization centers within `2 * delta` of the coefficient, with at
most `delta` pixel change. Use the actual successful target movement for costs.
Re-extract the final pixels and require exact saved-FFV1 bytes before any RS repair.

## Settings and capacity

The named `experimental-robust80` profile adds delta 80 with rate 1/2 and CRF
costs, including stronger bootstrap embedding and three protected header copies.
It uses [NDV5](NDV5.md) and is opt-in. The NDV4 settings below remain unchanged.

Profiles expose rates 1/2 and 1/3, delta 30/40/50, and CRF/CQP costs. The profile
ID fixes the matrix, rate, quantization, and cost model. Unsupported combinations
are rejected. The bootstrap remains legacy delta 40 and consumes separate whole
frames for new profiles. The decoder learns body settings from that repaired
bootstrap without requiring command-line arguments. See [NDV4](NDV4.md).

STC rate is the ratio of ECC-encoded bits to carrier bits. It is distinct from
useful payload occupancy, quantization strength, and the RS rate. At STC rate
1/2 with RS(60,20), asymptotic useful rate is 1/6 bit per carrier before headers
and padding. Exact capacity uses frame allocation, whole-byte rounding, protected
metadata, filename bytes, CRC, RS parity, and both kinds of padding.

## Integrated validation

The production receiver retains quantizer-boundary margins through the STC
mapping and byte deinterleaving. Failed body RS words get bounded erasure retries;
conflicting candidates are rejected, and whole-file CRC/padding validation remains
required. This decoder-only change consumes no carrier or parity capacity. See
[NDV4 receiver details](NDV4.md). The focused regression suite is
`python -m unittest discover -s tests -p test_decoder_confidence.py -v`.

Run `python -m unittest discover -s tests -v`. Tests cover cost formulas, flat and
boundary cases, all registered profiles through pixel reconstruction and saved
FFV1 videos, bootstrap switching, capacity boundaries, legacy decoding, ECC repair,
and failure cleanup. Small exhaustive STC tests establish correctness of the
implemented matrices; channel robustness must be measured separately.

Use the same benchmark manifest and useful payload for paired comparisons:

```bash
python -m benchmarks run benchmark-runs/fixtures/manifest.json --output benchmark-runs/stc-crf --stc matrix --rate 1/2 --delta 40 --cost crf
python -m benchmarks run benchmark-runs/fixtures/manifest.json --output benchmark-runs/stc-cqp --stc matrix --rate 1/2 --delta 40 --cost cqp
python -m benchmarks compare benchmark-runs/stc-crf/results.json benchmark-runs/stc-cqp/results.json --output benchmark-runs/cost-comparison
```

The benchmark knows the expected profile only for raw-error diagnostics; actual
file recovery still discovers the profile through the protected header. Analyze
header failures separately: body settings cannot repair an unreadable bootstrap.
Do not choose defaults from a single clip or assume higher rate/quantization or a
larger trellis automatically improves robustness. Compare recovery with distortion,
useful capacity, execution time, and results on independent holdout clips.

A local integration sweep tested all 13 profiles on textured synthetic footage
and the first 40 frames of `input0.mp4`, with an eight-byte seeded payload under
FFV1, H.264 CRF 26, and QP 26. All 26 FFV1 cases passed; 57 of 78 total cases
recovered exactly. Delta 30 added body failures; delta 40/50 matched the legacy
recovery count, while the real clip's CRF 26 bootstrap failure persisted for every
profile. These results do not establish a general improvement, so the default
remains legacy. The local, gitignored reports are in `benchmark-runs/profiles/`,
including `summary.md` and per-profile comparisons against legacy.
