# GPU backend implementation (initial historical design)

The single-work-item design below was replaced by parallel kernels, runtime caching
and workspace reuse; see `performance-plan.md` and the current README.

Implement optional OpenCL 1.2 GPU execution with float64 support for the full
existing dense/activation/MSE/SGD model family. Keep the default build CPU-only.
Reuse the pinned Khronos headers (with license) from the backup, not old kernels.

Use a correctness-first reference kernel, with one work item performing complete
forward/backward/update calculations in deterministic loop order. This executes
all numerical work on the selected GPU but intentionally does not promise GPU
speedups; parallel matrix kernels are future optimization work.

- Add hardware-conditional parity/gradient/update/XOR/switching tests first.
- Discover only CL_DEVICE_TYPE_GPU with fp64, compiler and availability; sort
  platform/vendor/device identifiers deterministically, retain runtime ordinal
  for identical devices, and build the production kernel before advertising usability.
- Add RAII session and buffers; embed kernels so installed libraries are portable.
- Dispatch predict/evaluate/gradients/train using a per-operation execution object.
- Keep session parameters authoritative within a call, copy updates to canonical
  host parameters after each successful batch, discard session at return.
- Never retry devices after execution starts. Distinguish disabled backend,
  missing platform/runtime, missing GPU, unsupported precision and build failures.
- Validate kernel source with Clang; test CPU-only and enabled OpenCL builds,
  installed linking, runtime diagnostics and accurate hardware test skips.

Environment: OpenCL loader and NVIDIA ICD files installed, zero OpenCL platforms;
NVIDIA driver cannot be contacted. Physical execution cannot be validated here.
