# Parallel frame processing

Encoding, saved-file verification, and body extraction run independent frame work
through `core/parallel.py`. A bounded queue keeps a small number of tasks per worker
in flight and returns results in frame order. Video I/O, audio muxing, timestamps,
and incremental message assembly remain in the main process.

The decoder consumes all three padded bootstrap groups in order before scheduling
body extraction. Workers return bytes and confidence together. Early completion
closes the result iterator and shuts down the pool; worker errors propagate.

`--workers N` controls processes for encode/decode. The default uses all CPU cores;
`--workers 1` runs inline. Batched Haar/SVD processing within each frame is retained
in either mode. Worker count does not change the embedding mathematics or mapping.

The explicitly selected profile settings travel with each worker task (or bound
extraction function), including saved-file verification and likelihood retries.
Workers do not infer a profile from process-global defaults.
