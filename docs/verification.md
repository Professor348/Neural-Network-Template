# Verification, 2026-11-15

## Current NPU source integration

Optional OpenVINO 2024.6+ Intel NPU inference is implemented in private SDK files.
The public API adds explicit `NpuOptions.allowReducedPrecision` opt-in; default
float64 behavior and CPU/GPU training remain unchanged. AUTO checks precision and
complete static graph preparation before selecting NPU; explicit requests never
fall back, and failures after inference begins never retry another device.

The implementation verifies complete query support and exact physical placement,
uses a bounded eight-entry compiled snapshot cache keyed by all current canonical
weight bits, and creates private mutable requests. Full loss graphs include
residuals, squares and reduction on NPU. No NPU training or other-vendor backend
is claimed. Independent source review found no remaining blocking issues.

Verified after integration (2026-11-15):

- CPU-only and OpenCL-enabled GCC Release suites: 11 passed, physical GPU test skipped.
- Clang OpenCL-enabled Release: 11 passed, physical GPU test skipped.
- AddressSanitizer/UBSan: 11 passed, GPU skipped, leak detection disabled because
  ptrace supervision prevents LeakSanitizer operation.
- NPU policy: precision opt-in, inference/training separation, explicit rejection
  and pre-execution AUTO skip behavior.
- Production Execution/Selection code with isolated test-only NPU boundary:
  complete prediction/loss dispatch, compile rejection versus execution failure,
  no runtime fallback, latest parameter snapshots and training rejection.
- Actual SDK-independent snapshot/conversion logic: exact cache key changes for
  low double bits even when their float conversion is equal, topology, signed
  zero, device, batch and operation; malformed shapes and out-of-range data reject.
- Enabling NN_ENABLE_NPU correctly reports the missing OpenVINO SDK dependency.

**Not verified:** a real OpenVINO-enabled C++ build, SDK graph validation, NPU
inference/precision/performance or actual NPU driver behavior. The SDK is absent
and the attempted download failed DNS resolution in this restricted environment.
The conditional `npu_graph_cpu_validation` target is supplied for checking actual
SDK graph semantics on CPU once the SDK is installed; that is not physical NPU
execution. The user explicitly waived NPU hardware validation.

GPU setup/access diagnostics and required hardware test commands are in
[gpu-testing.md](gpu-testing.md). No driver or host configuration was changed.

## Current optimized implementation

The original serial GPU design below is historical. Production now uses separate
parallel forward, delta, parameter-gradient, loss-block, SGD staging and commit
kernels. Immutable contexts/programs are cached by GPU; queues, entry handles and
mutable buffers remain private. Buffer and upload capacities are reused across
batches. Explicit CPU selection avoids GPU discovery/compiler startup entirely.

CPU forward and backpropagation reuse workspaces, group independent neuron sums,
skip unused original-input derivatives, and avoid parameter-count copies. See
[cpu-performance.md](cpu-performance.md) for exact paired benchmark medians:
prediction 5.580 to 2.125 ms, gradients 8.872 to 5.127 ms, and two-epoch training
38.381 to 17.979 ms on the documented synthetic workload. No unsafe fast-math,
architecture-specific compiler requirement or timing assertion was introduced.

Current checks (2026-11-15):

- GCC Debug/Release with OpenCL enabled: eight tests passed, hardware skipped.
- CPU-only Release: eight tests passed, hardware skipped.
- Clang 23.1.5 OpenCL-enabled Release: eight passed, hardware skipped.
- OpenCL-enabled AddressSanitizer/UBSan: eight passed, hardware skipped;
  leak detection disabled because the environment prevents LeakSanitizer operation.
- Clang OpenCL C 1.2 source validation, warnings as errors: passed.
- Parallel production kernel host algebra: all activations, batches 1/3/17/257,
  uneven layer widths, gradients, scaled loss and atomic overflow rejection passed.
- The real GPU session/runtime code also passed a host-only OpenCL shim check:
  strict argument sizes, bounded transfers, deferred upload lifetimes, allocation
  reuse over growing/shrinking batches, all operation modes, cached runtime,
  resident SGD state and overflow rejection. This is not physical GPU execution.
- Independent review found no blocking indexing, synchronization, ownership,
  race or numerical issues. The OpenCL target-reuse CMake follow-up was fixed.
- Installed static package plus independent consumer: linked and learned XOR.

`kernel_reference` is now explicitly a preserved historical serial oracle;
`parallel_kernels` tests the current production source on the host. Neither is
reported as physical GPU execution. Actual GPU execution, concurrency and speed
remain **unverified**: the runtime exposes zero OpenCL platforms and cannot
contact the NVIDIA driver. GPU benchmark skips rather than reporting CPU timings.

## Initial GPU implementation follow-up (historical)

Optional real OpenCL GPU execution is now implemented for forward inference,
scaled MSE, backpropagation and atomic minibatch SGD. The correctness reference
kernel runs one work item and makes no performance claim. GPU discovery excludes
OpenCL CPU devices and probes runtime availability, float64 and production build
support before advertising execution. NPU remains unimplemented.

| Check | Result |
|---|---|
| CPU-only Release suite | 5 passed; hardware execution skipped |
| OpenCL-enabled Debug and Release suites | 5 passed; hardware execution skipped |
| OpenCL-enabled AddressSanitizer/UBSan | 5 passed; hardware skipped; leak detection disabled due to ptrace limitation |
| Exact production kernel source under Clang OpenCL C 1.2, warnings as errors | Passed |
| Host-only production kernel algebra | Forward/MSE/gradients/SGD across all four activations and overflow atomicity passed; not hardware execution |
| Installed GPU-enabled static library consumed via find_package | Built and ran; AUTO correctly selected CPU and reported missing OpenCL platforms |
| Parent project already defining OpenCL::OpenCL | Configure/build passed |
| Required-GPU hardware test mode | Correctly returned failure rather than skip with no available GPU |
| Whitespace check | Passed |

`clinfo` reports zero platforms; `nvidia-smi` cannot communicate with the driver.
Real GPU parity, finite differences, SGD, large MSE, overflow rejection, XOR and
CPU/GPU switching tests compile but **could not execute here**. No physical GPU
validation or speedup is claimed. `NN_REQUIRE_GPU_TESTS=1` makes this a hard failure
on hardware CI. No driver changes or installations were made.

Reproduce GPU verification:

```sh
cmake -S . -B build-gpu -DNN_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu --parallel
ctest --test-dir build-gpu --output-on-failure
# On a compatible physical GPU host:
NN_REQUIRE_GPU_TESTS=1 ctest --test-dir build-gpu -R accelerator_execution --output-on-failure
```

## Original CPU rewrite baseline (historical)

Backup reference supplied by the user: `103b00b8df4342294b51767903ee9c66197750ce`
(`backup before refactor`). Remote backup verification was waived by the user.
No push or history rewrite was performed by the rewrite task.

Environment: Linux, GCC 16.2.1, CMake, C++17; CPU execution only.

| Check | Result |
|---|---|
| Debug configure/build and complete CTest suite | 4 passed, accelerator execution skipped |
| Release configure/build and complete CTest suite | 4 passed, accelerator execution skipped |
| AddressSanitizer + UndefinedBehaviorSanitizer suite | 4 passed, accelerator execution skipped, with leak detection disabled |
| Library-only build, examples/tests OFF | Passed |
| Install and separate `find_package(nn)` consumer | Passed; consumer built and ran XOR |
| XOR example | Loss 0.254057 to 0.000245703; all four outputs correct at threshold 0.5 |
| Central differences | All parameters, all four activations, tolerance 1e-7 |
| Exact one-step SGD | Full batch, all four activations, tolerance 1e-14 |
| Large finite MSE | Residual 2e154 among four observations yields finite MSE 1e308 |
| Nonfinite candidate update | Exception propagated; entire current batch left unchanged |
| Whitespace check | `git diff --check` passed |

The first sanitizer invocation failed all five targets at process exit because
LeakSanitizer cannot run under this environment's ptrace supervision. Rerunning
with `ASAN_OPTIONS=detect_leaks=0` passed AddressSanitizer/UBSan. Leak detection
was therefore not verified. No memory safety or undefined behavior diagnostic
was reported in the successful run.

Independent code review found no blocking CPU numerical or selection defects.
Its suggested overflow-update regression and scaled loss accumulation were added.
At this baseline GPU/NPU execution was unimplemented (GPU support is superseded
by the follow-up above);
synthetic inventory policy tests are not hardware execution tests. Other operating
systems and compilers were not tested.

Reproduce sanitizer checks:

```sh
cmake -S . -B build-sanitized -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-sanitized --parallel
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-sanitized --output-on-failure
```

Reproduce installed consumer check (after building Release):

```sh
cmake --install build-release --prefix /tmp/nn-install
cmake -S tests/consumer -B /tmp/nn-consumer-build -DCMAKE_PREFIX_PATH=/tmp/nn-install
cmake --build /tmp/nn-consumer-build --parallel
/tmp/nn-consumer-build/consumer
```
