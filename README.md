# Neural network library template

A small C++17 feedforward neural network library, written for correctness and
readability. Applications link `nn::nn` and control architecture, training,
logging and presentation through ordinary C++ calls. There is no command-line
interface. The separate XOR executable is an example application.

The original head-tracking application is preserved in commit
`103b00b8df4342294b51767903ee9c66197750ce` (`backup before refactor`), supplied
by the user. Its datasets, models, camera/UI code, vendored dependencies and
old training implementations have been removed from this working tree.

## Build and test

Requires CMake 3.16+ and a C++17 compiler. CPU builds have no external dependencies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/nn_xor
```

Tests check central finite differences for every parameter for all activations,
exact SGD updates, XOR learning, deterministic initialization/training, invalid
inputs, callback propagation, selection policy and call-only overrides.
Accelerator execution runs on a compatible OpenCL GPU, otherwise it is explicitly
**skipped**. Set `NN_REQUIRE_GPU_TESTS=1` in hardware CI to make missing/failed
GPU discovery a failure instead of a skip. `parallel_kernels` compiles the exact production kernels as host C++ to check
their algebra/indexing over batch sizes 1, 3, 17 and 257 and uneven feature widths.
`kernel_reference` retains the old serial kernel only as a test oracle. Both are
expressly host-only tests and never reported as accelerator execution.
`opencl_session_mock` additionally runs the real session/runtime orchestration
against a test-only OpenCL shim: it checks argument ABI, buffer bounds, deferred
upload lifetimes, increasing/decreasing batch sizes, allocation reuse, resident
SGD state and rejected-update atomicity. It never links the physical OpenCL loader
or registers a simulated GPU with the library.
Assertions use exceptions, so checks remain active in Release builds.

For a library-only build, use `-DNN_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF`.

## Demo programs

All demos use the public API and synthetic data generated in code:

| Target | Demonstrates |
|---|---|
| `nn_xor` | Nonlinear XOR learning and device selection |
| `nn_regression` | Nonlinear regression with progress callbacks |
| `nn_classification` | Nonlinear 2D classification with held-out accuracy |
| `nn_devices` | Discovery, capabilities and per-operation overrides |
| `nn_npu_inference` | CPU training followed by explicit NPU inference |

Build normally, then run `./build/nn_regression` or another target. With examples
and testing enabled, `ctest --test-dir build -L example --output-on-failure`
runs demo smoke tests. The NPU demo skips with status 77 when no compatible NPU is
available. [Demo details](examples/README.md) include build and run instructions.

Physical NPU tests are registered independently as `npu_hardware_inference`,
`npu_hardware_loss`, `npu_hardware_synchronization` and `npu_hardware_contracts`.
They compile on CPU-only machines and skip without hardware; on an NPU host use
`NN_REQUIRE_NPU_TESTS=1 ctest --test-dir build-npu -R '^npu_hardware_' --output-on-failure`
to require actual execution. See [NPU testing](docs/npu-testing.md) for coverage,
precision tolerances and the distinction from SDK-free and graph-validation tests.

## Link from an application

Inside another CMake project:

```cmake
add_subdirectory(path/to/Neural-Network neural-template)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE nn::nn)
```

Or install and consume as a package:

```sh
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(nn 1 CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE nn::nn)
```

Configure the consuming project with `-DCMAKE_PREFIX_PATH=/your/install/prefix`.
Public headers have no hardware SDK dependencies.

## Public API

See [the complete example](examples/xor.cpp). The core calls are:

```cpp
// Two input features, initialization seed 7, automatic device preference.
nn::Network network(2, 7);
network.addDense(4, nn::Activation::Tanh);
network.addDense(1, nn::Activation::Sigmoid);
network.setLoss(nn::Loss::MeanSquaredError);
// Samples by features, generated directly in the application.
const nn::Dataset inputs{{0,0}, {0,1}, {1,0}, {1,1}};
// One output label per sample.
const nn::Dataset targets{{0}, {1}, {1}, {0}};
// Per-call training configuration; optimizer is minibatch SGD.
nn::TrainingOptions options;
options.epochs = 6000;
options.batchSize = 4;
options.optimizer.learningRate = 0.5;
options.seed = 123;
// Training summary includes initial/final loss and actual device selection.
const auto result = network.train(inputs, targets, options);
// Predictions include output samples and the inference selection report.
const auto prediction = network.predict(inputs);
```

`train` optionally takes a synchronous `std::function<void(const nn::Progress&)>`
as its fourth argument. The callback receives the completed epoch number and
full-dataset loss. The library performs no logging. Callback exceptions propagate;
completed batch updates remain applied. Do not mutate or recursively train a
network from its own callback. Network instances require external synchronization
for concurrent mutation or mutation during inference.

All numerical computation uses `double`. Available activations are `Linear`,
`Tanh`, `Sigmoid` (overflow-safe evaluation), and `Relu` (derivative zero at zero).
Weights use Xavier uniform initialization and biases start at zero. The loss is
mean squared error averaged over **both samples and output features**, with no
implicit factor of one half. Each batch gradient is averaged over its actual
sample count, including the last partial batch. SGD applies `parameter -=
learningRate * gradient` with no momentum, weight decay or hidden optimizer state.

Constructor seeds control initialization. `TrainingOptions::seed` controls
per-call sample shuffling independently. Repeated calls restart the shuffle RNG;
identical models/calls reproduce results on the same standard library and build.
C++ standard library distributions and shuffles do not guarantee identical
sequences across different implementations. Loss reduction need not be monotonic
for arbitrary architectures, learning rates or minibatches.

`gradients` exposes analytic mean-loss derivatives for diagnostics.
`parameters` and `setParameters` use layer order, then output-major weights,
then biases. Parameter count and finiteness are validated before replacement.
The template intentionally omits serialization, convolution, autodiff and batching
acceleration. Add those only with independent numerical tests.

Invalid dimensions, empty/ragged/nonfinite datasets, bad enum values and invalid
settings throw `std::invalid_argument`. Device selection and nonfinite computed
results throw `std::runtime_error`. Updates are staged per batch so a nonfinite
candidate update is never partially applied; earlier batches remain committed.
Allocation failures propagate normally. No device retry occurs after execution.

## Device selection and inspection

A configured preference is distinct from the device actually executing a call.
`getDevice()` returns the preference (possibly AUTO); `getDeviceName()` returns
`"Auto"` or the explicit device name. Each `Prediction` and `TrainingResult`
returns `selection.device` and `selection.skipped` for the actual operation.

```cpp
// Equivalent constructor accepts type, optional index, then initialization seed.
nn::Network network(2, nn::DeviceType::CPU, 0, 7);
network.setDevice(nn::DeviceType::CPU, 0);
// Call-only automatic override leaves persistent CPU 0 preference unchanged.
const auto output = network.predict(inputs, nn::DevicePreference{nn::DeviceType::AUTO, std::nullopt});
```

`train`, `predict`, `evaluate` and `gradients` accept an optional last
`DevicePreference` override. Explicit types with no index mean index zero.
AUTO with **any** index is invalid. Indices are zero-based within a device type;
the reference backend exposes the host CPU as one device (CPU 0), rather than
one device per core. OpenCL GPU indices are assigned among usable devices sorted by vendor, device
name, driver version and OpenCL version, with original platform/device enumeration
ordinal breaking ties for identical devices. This immutable process-lifetime
snapshot keeps indices stable within a process; indices may change across driver
updates or reboots. Restart the process after fixing or changing drivers. The
policy scans devices of each type in this documented inventory order.

`nn::getAvailableDevices()` lists only executable devices with type, index, name,
backend and training/inference flags. `nn::isDeviceAvailable(preference)` tests
usability; its optional second `Operation` argument tests training or inference.
Invalid preferences return false from this predicate. Basic usability never
promises compatibility with arbitrary future operators or precision formats.
The complete model family is float64 dense layers with the activations
and loss described above; CPU and the optional OpenCL GPU backend support that
entire family for both training and inference.

AUTO considers NPU, GPU, then CPU and records skips. Training and inference are
resolved separately; an inference-only descriptor cannot satisfy training. The
internal pure policy tests use synthetic inventories, which cannot register
numerical execution backends or cause CPU computation to be labeled as an
accelerator. Explicit unavailable devices, invalid indices and unsupported
operations throw without fallback.

## Backend status and requirements

| Backend | Execution | Requirements and limits |
|---|---|---|
| CPU | Implemented; tested on Linux with GCC | Standard C++17, float64 dense inference and SGD training. Other OS/compiler combinations have not been tested here. |
| GPU | Implemented via optional OpenCL; builds and parallel kernel algebra tested; **physical GPU execution unverified here** | OpenCL 1.2+ GPU runtime/ICD and loader library; GPU must expose `cl_khr_fp64`, nonzero double precision support, availability and source compiler. |
| NPU | Optional Intel NPU/OpenVINO inference source implemented; **SDK build and NPU hardware unverified here** | OpenVINO Runtime 2024.6+ SDK, Intel NPU plugin/driver and supported hardware; explicit reduced-precision opt-in. Training unsupported. |

Enable OpenCL GPU execution explicitly (CPU-only remains the default):

```sh
cmake -S . -B build-gpu -DNN_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu --parallel
ctest --test-dir build-gpu --output-on-failure
./build-gpu/nn_xor
```

The OpenCL loader/development library must be linkable as `OpenCL`. Pinned Khronos
headers are bundled privately under `third_party/OpenCL` with their upstream
license and provenance. No GPU SDK appears in public headers. The installed
package locates the OpenCL loader for consuming applications. CPU-only packages
have no OpenCL dependency.

GPU selection uses the same public API:

```cpp
// Explicit GPU zero; throws if hardware, runtime, precision or compiler is unsuitable.
nn::Network network(2, nn::DeviceType::GPU, 0, 7);
network.addDense(4, nn::Activation::Tanh);
network.addDense(1, nn::Activation::Sigmoid);
// Options, sample matrices and targets are configured as in the example above.
const auto result = network.train(inputs, targets, options);
// This call uses CPU without changing the configured GPU preference.
const auto prediction = network.predict(inputs, nn::DevicePreference{nn::DeviceType::CPU, 0});
```

**Execution design:** separate parallel kernels process batched forward neurons,
output/hidden derivatives, parameter gradients and SGD candidates. Each work item
owns its output; float64 gradients use deterministic sample reductions without
floating-point atomics. Loss reduces independent blocks, then combines their
scaled sums to preserve large finite MSE values. A shared integer error flag and
separate validation/commit launches prevent partial numerical updates. In-order
queue boundaries synchronize all dependent stages. Unsafe fast-math is disabled.

Compiled programs and GPU-only contexts are cached per device. Each operation
has private queues, kernel handles, parameters and workspaces, so concurrent
models do not race over kernel arguments or mutable storage. Buffer capacities
and upload storage are reused across training batches, including a smaller final
batch. Shuffling, batching, callbacks and transfers are host orchestration;
forward/loss/gradient/SGD arithmetic stays on the selected GPU. Successful batches
synchronize latest weights to canonical host parameters. Later CPU/GPU calls and
per-call overrides start from those weights; model parameter caches do not
persist between public calls.

CPU execution reuses activation and derivative scratch across samples, avoids
intermediate matrix copies, skips unused input derivatives, and groups four
independent neuron sums to expose vectorization while preserving each neuron's
reduction order. The implementation uses portable C++ rather than architecture
intrinsics or unsafe numerical flags. This improves a measured medium synthetic
workload; it is not a guarantee of best performance for every architecture.

Optional benchmarks:

```sh
cmake -S . -B build-perf -DCMAKE_BUILD_TYPE=Release -DNN_BUILD_BENCHMARKS=ON -DNN_ENABLE_GPU=ON
cmake --build build-perf --parallel
./build-perf/nn_benchmark
./build-perf/nn_gpu_benchmark
```

The GPU benchmark skips with exit code 77 if no executable GPU is present. CPU
before/after medians and method are in [CPU performance](docs/cpu-performance.md).
The measured workload improved about 2.6x for inference and 2.1x for training.
Actual GPU performance remains **unmeasured** on this host. These readable
parallel kernels are not vendor-tuned BLAS or tiled matrix kernels; small models
may still run faster on CPU because of launch/transfer overhead. AUTO selects
capability priority, not predicted speed. Further tuning needs physical GPU
benchmarks and the same numerical checks.

Discovery requests only `CL_DEVICE_TYPE_GPU`; OpenCL CPU devices are never used
or labeled as GPUs. It checks availability, compiler, fp64 and runtime version,
and successfully compiles the production kernel before advertising each usable
GPU. Diagnostics distinguish a disabled backend, no OpenCL platforms (missing or
broken vendor runtime/driver), no GPU exposed by an installed platform,
unsupported float64/compiler/version, kernel build logs, invalid index and
execution API failures. Loader discovery alone cannot always distinguish absent
hardware from a broken driver; messages state only what is known. AUTO may skip
unusable devices **before** execution; runtime/kernel/allocation failures after
selection propagate without retry or CPU fallback.

The development host reports zero OpenCL platforms and cannot communicate with
the installed NVIDIA driver. Both CPU-only and OpenCL-enabled builds were tested,
as were OpenCL C syntax, host-only parallel kernel algebra and installed
linking. The real GPU parity, central-difference, update, XOR and cross-device
synchronization tests are present but skipped here. NVIDIA/AMD/Intel hardware and
Windows/macOS execution have not been verified. No measured GPU speedup is claimed.

## Optional Intel NPU inference

Build against an installed OpenVINO Runtime 2024.6+ development package:

```sh
cmake -S . -B build-npu -DNN_ENABLE_NPU=ON -DOpenVINO_DIR=/path/containing/OpenVINOConfig.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-npu --parallel
ctest --test-dir build-npu --output-on-failure
```

GPU and NPU build flags are independent and can both be enabled. Public headers
expose no OpenVINO SDK types. Installed NPU-enabled packages find OpenVINO Runtime
as a transitive dependency. CPU-only builds do not need either accelerator SDK.

**Capability and precision:** this backend targets Intel NPUs supported by the
OpenVINO NPU plugin on its supported Linux/Windows systems. It is inference-only;
training/gradient requests reject explicit NPU selection and AUTO skips NPU for
training. Other vendors' NPUs are not implemented. NPU hardware computes in FP16,
so float64 remains the default on CPU/GPU and an NPU cannot silently reduce it.
Applications must explicitly permit reduced-precision NPU inference:

```cpp
// Train on CPU using canonical double parameters, as in examples/xor.cpp.
network.setNpuOptions({true}); // Permit f32 NPU I/O and FP16 internal arithmetic.
network.setDevice(nn::DeviceType::NPU, 0);
// Graph support/compilation are checked before any inference executes.
const auto predicted = network.predict(inputs);
// Include all MSE operations on NPU; throws if the whole loss graph is unsupported.
const double loss = network.evaluate(inputs, targets);
```

`getNpuOptions()` reports that permission; changing it never changes weights or
CPU/GPU training precision. With AUTO, no precision opt-in means NPU is skipped
with a reason. With opt-in, the first compatible complete NPU graph may be used.
Basic NPU discovery/`isDeviceAvailable` checks do not guarantee compatibility with
this model, batch, precision or operation.

Discovery lists only actual OpenVINO `NPU`/`NPU.<id>` devices, sorted
lexicographically by runtime ID, with zero-based public indices into that list.
It reports plugin/driver errors without claiming it can always distinguish a
missing NPU from an unavailable driver. Every prediction/loss graph uses static
exact batch dimensions and includes the latest complete canonical parameter
snapshot. Every graph node must be assigned by `query_model`; compilation names
one concrete NPU, and reported execution placement must agree. OpenVINO AUTO,
HETERO, MULTI and CPU fallback are never enabled. Unsupported model preparation
can be skipped by library AUTO **before execution**; runtime inference failures
propagate without fallback. Explicit requests always fail rather than falling back.

For speed, a process-wide Core and bounded eight-entry compiled-model cache are
shared. Cache keys include the exact device, batch, prediction/loss mode, full
architecture and every canonical double parameter bit, so CPU/GPU training and
`setParameters` cannot reuse stale NPU weights. Requests and input/output tensors
are session-private. Float32 upload/widening are host transfers; forward math and
complete MSE reduction stay in the compiled NPU graph. Reduced-precision rounding,
range and overflow behavior differ from float64; output finiteness is checked.

**Verification status:** the SDK is not installed in this execution environment,
and an attempted SDK download failed DNS resolution. Real OpenVINO-enabled
compilation, driver execution, NPU correctness and NPU performance are therefore
unverified. Default/CPU/GPU builds and isolated production dispatch tests remain
verifiable without it. NPU hardware validation was intentionally not attempted.
This is optional source integration, not a claim of validated NPU support. The
conditional `npu_graph_cpu_validation` test checks actual OpenVINO graph semantics
on its CPU plugin when the SDK is available; it is never counted as NPU hardware
execution. Always-available snapshot and mocked-dispatch tests cover actual key,
conversion, precision/selection and no-fallback logic without the SDK.

Official references: [OpenVINO NPU device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
and [NPU driver/setup requirements](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-npu.html).

For the existing GPU backend, see [physical GPU testing and session access](docs/gpu-testing.md).

## Structure and starting a project

- `include/nn/`: public API; one class/struct per matching header.
- `src/numerics/`: scalar activation functions and derivatives.
- `src/backends/`: execution dispatch and CPU reference math.
- `src/backends/opencl/`: optional GPU discovery, resource ownership and embedded kernel.
- `src/backends/openvino/`: optional Intel NPU discovery, graph compilation and inference.
- `src/devices/`: real discovery and pure deterministic selection policy.
- `src/Network.cpp`, `src/Training.cpp`, `src/Validation.cpp`: model API, training
  orchestration and precondition checks.
- `examples/`: caller-owned application entry point, separate from the library.
- `tests/`: numerical, learning, validation, selection and hardware skip checks.

Copy or fork this tree, rename the CMake project if desired, and add your own
application target linked to `nn::nn`. Generate or load data in that application;
keep datasets/checkpoints outside the library source. Replace XOR with your own
validated feature/target matrices. Keep the gradient tests when adding activations,
losses or layers. Prefer this CPU implementation as the numerical reference for
backend comparisons.
