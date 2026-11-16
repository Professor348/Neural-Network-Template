# CPU performance

The CPU backend keeps scalar `double` arithmetic and serial sample ordering. Four independent output-neuron sums share each input load; each individual sum still visits inputs in ascending order. This exposes independent work to the compiler without reassociating reductions, enabling fast-math, requiring CPU-specific instructions, or introducing threads. A scalar tail handles output widths not divisible by four.

Prediction, loss, and batch backpropagation allocate layer activation rows once per invocation and reuse them across samples. Inputs remain caller-owned. Backpropagation swaps two reusable derivative rows, uses contiguous gradient updates, and omits the unused derivative with respect to the network's input. Scratch is local to a call; model parameters are never cached across calls.

`parameters()` reserves its exact output count. `setParameters()` counts stored elements directly rather than allocating and copying the existing parameter vector just to validate the replacement length. It still validates all candidate scalars before making any assignments.

## Reproduce the benchmark

`tests/cpu_performance.cpp` is a standalone benchmark with deterministic synthetic data. It measures public CPU inference, loss, gradients, two training epochs, and repeated parameter replacement. It has two warmups per operation and reports the median of 15 measurements, or nine for training. It has no timing assertions. Training restores the same parameters before every repetition; that restoration is included in its reported time. An observable checksum consumes results.

Build and run from the repository root:

```sh
cmake -S . -B build-cpu-performance -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_GPU=OFF -DNN_BUILD_BENCHMARKS=ON
cmake --build build-cpu-performance -j 4
./build-cpu-performance/nn_benchmark
```

For a before/after comparison, compile a separately named executable against each implementation's Release static library using the same benchmark source and compiler flags. Keep both executables, then run them consecutively under the same machine load. On Linux, `taskset -c 0 ./build-cpu-performance/nn_benchmark` pins a run to CPU 0 when that CPU is available. The benchmark intentionally does not set affinity itself.

## Observed before/after results

Measured on 2026-11-15 with GCC 16.2.1 (20260810), an Intel Core i9-9900 at its normal variable clock, Linux x86-64, and CPU affinity 0. Both libraries used CMake Release flags `-O3 -DNDEBUG`, with neither fast-math nor architecture-specific flags. The baseline executable was statically linked against the original CPU implementation before edits; it was retained and rerun immediately before the final optimized executable. Each cell below is the benchmark's within-run median, from that final paired run.

The network is `128 -> 192 (tanh) -> 128 (ReLU) -> 32 (linear)`: 53,600 parameters, 128 samples, 128 input features and 32 targets per sample. SGD uses batch size 32, learning rate 0.001, two epochs, and no shuffling.

| Operation | Before | After | Before / after |
| --- | ---: | ---: | ---: |
| Predict 128 samples | 5.580 ms | 2.125 ms | 2.63x |
| Evaluate 128 samples | 5.582 ms | 2.063 ms | 2.71x |
| Gradient, 128 samples | 8.872 ms | 5.127 ms | 1.73x |
| Train, two epochs, batch 32 | 38.381 ms | 17.979 ms | 2.13x |
| `setParameters`, 1,000 calls | 77.431 ms | 34.894 ms | 2.22x |

Both runs printed checksum `24.682` at the benchmark's three-decimal display precision. That checksum is only an anti-elision observation; correctness is checked independently. These measurements describe this synthetic workload and compiler on this host. Allocation savings matter differently for small networks, and the grouped sums' benefit varies by compiler, activation, layer width, and hardware. Timings also vary with clock frequency and concurrent work.

## Correctness checks

`tests/cpu_regression.cpp` compares grouped feedforward with an independent scalar neuron loop and tests all four activations on widths `3 -> 5 -> 7 -> 2`, including grouped output tails and growing/shrinking scratch widths. Batch gradients are compared with independent single-sample gradients and central finite differences for every parameter. Invalid parameter lengths and nonfinite values must leave the model unchanged.

Standalone regression invocation:

```sh
c++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -Iinclude -Isrc tests/cpu_regression.cpp build-cpu-performance/libnn.a -o /tmp/nn-cpu-regression
/tmp/nn-cpu-regression
ctest --test-dir build-cpu-performance --output-on-failure
```

The dedicated regression passed. The integrated Release suite passed its CPU, numerical, learning, device, serial-kernel reference, and parallel-kernel algebra tests; the accelerator execution test skipped because that configuration had no enabled accelerator backend. Numerical checks retain stable scaled MSE accumulation and explicit rejection of nonfinite forward results, gradients, and candidate parameter updates.
