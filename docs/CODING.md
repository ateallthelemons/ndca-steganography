# Coding implementation

The file pipeline uses height-three matrix STC at rate 1/2, delta 80, CRF costs,
and the physical mapping specified in [NDV5](NDV5.md).

## Matrix STC

`core/stc.py` minimizes `sum(costs[i, y[i]])` subject to `H y = message` over GF(2).
Both candidate bits can have nonzero costs; infinity forbids a candidate.
The implementation rejects infeasible paths and invalid costs. It uses eight
syndrome states, constrains and shifts a row after each group of carriers, and
truncates columns at the last row. There are no termination carriers.

The general engine retains both column-mask definitions: `(7, 5)` for rate 1/2
and `(7, 5, 3)` for rate 1/3. `Settings` in `core/config.py` supplies explicit width,
delta, and cost inputs to numerical functions. File tools use `CURRENT` only;
these internal options do not constitute supported alternate file formats.
Changes to extraction settings must agree between encoder and decoder.

STC minimizes embedding cost; RS repairs channel errors. The matrix implementation
follows the [STC construction](https://dde.binghamton.edu/download/syndrome/), but
its masks are not claimed to match the NDCA authors' unspecified matrices.

## Costs and reconstruction

Compute floating-point mean absolute differences from each block's upper and/or
left neighbor, averaging when both exist. Exclude block zero and normalize over
all eligible full blocks in the frame before selecting carriers:

```text
d = (difference - minimum) / (maximum - minimum + 1e-6)
f_crf = 1 - d + 1e-6
f_cqp = d + 1e-6
keep_cost = actual_keep_target_movement * f
flip_cost = actual_flip_target_movement / f
```

Both formulas remain available internally. The working pipeline uses CRF.
Cost selection describes the anticipated channel; output is always FFV1.

For each bit, try nonnegative same-parity centers within twice delta of the
original coefficient. Reconstruct, round, and clip pixels, permitting at most
delta pixel change. Choose the nearest recoverable target, breaking ties toward
the smaller target. Mark unavailable bits infeasible before trellis optimization.
Re-extract the completed plane and verify the saved file exactly.

The batched Haar/SVD implementation is the runtime path. Rounding, clipping,
carrier order, and matrix termination all affect recovery and must stay consistent.
Confidence uses realized coefficient distance from quantization boundaries and
follows the matrix support and byte packing; details are in [NDV5](NDV5.md).
