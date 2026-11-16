# Public API demos

Every program links `nn::nn`, generates its own small dataset in code and has no
command-line arguments or model/dataset files. Applications own logging and
presentation; none of the demos is part of the library target.

Build from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_BUILD_EXAMPLES=ON
cmake --build build --parallel
```

| Program | Source | Demonstrates |
|---|---|---|
| `nn_xor` | `xor.cpp` | Seeded XOR learning, AUTO selection and call-only overrides |
| `nn_regression` | `regression.cpp` | Nonlinear scalar regression and application-owned progress callbacks |
| `nn_classification` | `classification.cpp` | Nonlinear 2D classification with held-out accuracy |
| `nn_devices` | `devices.cpp` | Discovery, capabilities, configured versus actual selection and overrides |
| `nn_npu_inference` | `npu_inference.cpp` | CPU training followed by explicit reduced-precision NPU inference |

Run each executable directly, for example `./build/nn_regression`. The learning
demos return a nonzero exit status if their numerical success criteria fail.
Their fixed seeds and small bounded workloads are intended as starting examples,
not production architectures or accelerator speed benchmarks.

CTest smoke-tests all built demos:

```sh
ctest --test-dir build -L example --output-on-failure
```

`nn_npu_inference` returns 77 with an explicit skip diagnostic when no NPU backend
and device are available. It never labels CPU predictions as NPU execution. To
use it on an actual Intel NPU, configure `NN_ENABLE_NPU=ON` with OpenVINO Runtime
2024.6+ and the NPU driver. GPU and NPU build options remain independent.

Disable examples with `-DNN_BUILD_EXAMPLES=OFF`. See
[physical NPU tests](../docs/npu-testing.md) for tests that can require hardware.
