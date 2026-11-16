# Optional Intel NPU inference

Use OpenVINO Runtime 2024.6+ through private C++ implementation files, optional
at build time. The public API remains free of SDK types. NPU is inference-only;
training/gradients never run on an NPU and explicit training requests fail.

Preserve float64 defaults. Add NpuOptions.allowReducedPrecision (default false)
and Network set/getNpuOptions so applications explicitly permit float32 I/O and
float16 internal NPU computation. CPU/GPU continue to use float64. Basic NPU
availability is distinct from compatibility with this model/precision/batch.

Resolve actual model capabilities before numerical execution. AUTO checks NPU,
GPU then CPU; incompatible precision or full-graph compilation may be skipped
before execution. Explicit requests cannot fall back. Runtime inference failures
are propagated and never retried on CPU/GPU.

Graph construction uses current canonical weights, static exact batch dimensions,
dense/activation nodes, and optional full loss reduction on device. Compile only
to concrete NPU IDs; query every node and verify compiled execution devices.
Never use OpenVINO AUTO, MULTI, HETERO or CPU fallback.

Cache a bounded number of immutable compiled snapshots keyed by exact complete
architecture, weight bits, batch size, device and prediction/loss mode. Allocate
private requests per call; keep all mutable request data session-private.

Test policy/dispatch with an isolated test-only backend and simulated discovery,
keeping it separate from physical NPU execution. Test actual graph construction
with OpenVINO on CPU when the SDK is available, clearly labeling that as graph
validation. No NPU hardware is available and the user waived hardware validation.
The OpenVINO SDK is not installed here; package download was blocked by DNS in
the restricted execution environment. Enabled SDK compilation must be reported
as unverified unless a real SDK becomes available.
