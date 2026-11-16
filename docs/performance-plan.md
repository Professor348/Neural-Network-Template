# Readable performance improvements

Preserve the existing public API, numerical checks and device semantics.

CPU: establish before/after synthetic benchmarks; reuse forward/backprop storage
across samples, avoid intermediate output copies and eliminate allocation merely
to count parameters. Keep deterministic sample reduction order and no unsafe math.

GPU: replace the serial reference operation with separate parallel kernels for
forward neurons, output/hidden deltas, parameter gradients, SGD candidate and
commit. Each work item owns an output, eliminating floating point atomics.
Loss uses parallel blocks and a small scaled final reduction. Keep in-order
kernel boundaries as global synchronization, with atomic error flags and a
separate commit kernel so invalid candidate updates never partially commit.
Cache compiled programs/contexts per GPU; keep queues and scratch session-private.
Reuse capacity-based buffers across batches. GPU math remains entirely on one
GPU and no unsupported operator is moved to the CPU.

Tests: retain the old serial GPU algebra only as a test reference, compare new
parallel kernels against CPU across varied shapes, activations, uneven batch
sizes and numeric failures. Clearly label host simulation. Real hardware tests
must continue to skip/fail-required honestly when no OpenCL GPU is available.
Benchmark GPU only on real GPU; do not claim speedup from structural changes.
