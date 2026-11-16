# NPU tests

The physical suite uses the public library API and can be compiled without the
OpenVINO SDK. Its numerical cases require an actual compatible Intel NPU and an
OpenVINO-enabled library build. CPU/mock execution never counts as NPU success.
There is deliberately no NPU training implementation to test: training and
parameter-gradient requests must be rejected on an explicitly selected NPU.

| CTest case | Scope |
|---|---|
| `npu_hardware_inference` | Actual NPU predictions versus CPU reference across activations, output coordinates and static batches |
| `npu_hardware_loss` | Complete device-computed MSE versus CPU, including multi-output reduction |
| `npu_hardware_synchronization` | Latest CPU-trained/replaced weights, compiled graph snapshots, changing batches and call-only overrides |
| `npu_hardware_contracts` | Explicit reduced-precision permission, training/gradient rejection and invalid selection contracts |

All discovered NPU indices are exercised. Actual prediction selection must report
the requested NPU type/index, and every numerical output must be finite. Mixed
absolute/relative tolerances reflect explicitly permitted FP16 computation rather
than requiring float64 equality. Cases fail for numerical mismatches or a graph
rejected after a usable NPU was discovered; model-compiler errors do not become
silent passes. Support remains model/operator/batch/precision dependent.

Without a usable NPU each case returns 77 and CTest reports **Skipped**. Setting
`NN_REQUIRE_NPU_TESTS=1` instead makes missing hardware/runtime a test failure. This
lets the suite live in CPU-only projects while preventing a dedicated NPU CI job
from silently skipping all hardware coverage. Cases share a CTest resource lock
so compilation/cache load on the same physical device is serialized.

On a future NPU machine:

```sh
cmake -S . -B build-npu -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_NPU=ON \
  -DOpenVINO_DIR=/path/containing/OpenVINOConfig.cmake
cmake --build build-npu --parallel
NN_REQUIRE_NPU_TESTS=1 ctest --test-dir build-npu -R '^npu_hardware_' --output-on-failure
```

For visible device names and diagnostics, add `--verbose`. `NN_REQUIRE_NPU_TESTS`
does not enable the backend or install a runtime; it changes the missing-hardware
test outcome. Ordinary runs on this device can omit that environment variable:

```sh
ctest --test-dir build -R '^npu_hardware_' --output-on-failure
```

A broader NPU-focused run includes the always-executable software checks:

```sh
ctest --test-dir build -L npu --output-on-failure
```

- `npu_policy`: simulated discovery and compile-only compatibility decisions.
- `npu_snapshot`: SDK-independent range/shape checks and exact cache identities.
- `npu_dispatch_mock`: real dispatch code at a test-only SDK boundary.
- `npu_graph_cpu_validation`: conditional OpenVINO CPU-plugin graph semantics;
  available only when `NN_ENABLE_NPU=ON`, with no claim of NPU execution.

These checks are labeled separately from the four physical `hardware` cases.
The `nn_npu_inference` example is an additional demonstration, not a substitute
for required-hardware tests. No NPU hardware execution is claimed for development
on this machine. The real OpenVINO-enabled SDK build remains a separate validation
step; hardware test source compilation alone does not validate the SDK backend.
