# Testing on a physical GPU

The optional backend uses **OpenCL 1.2 or newer on a GPU**, with a source compiler,
`cl_khr_fp64` and nonzero double-precision support. An OpenCL CPU implementation
does not qualify. CUDA installation alone does not establish OpenCL availability.
The library has a C++ API; the commands below run its test and benchmark programs.

## First distinguish host support from session access

Run these read-only checks in a normal EndeavourOS terminal, outside the Codex
session, and compare their output with the same commands inside the session:

```sh
nvidia-smi
clinfo -l
clinfo
ls -l /dev/nvidia*
```

`nvidia-smi` should list the card. `clinfo` must expose a GPU device with the
requirements above; inspect its device type, availability, compiler availability,
OpenCL version, extensions and double-precision properties.

| Result | Next step |
| --- | --- |
| Host commands work; session commands fail or lack device nodes | Run the build and tests from the host terminal, or use an execution environment that explicitly supports GPU access. |
| Host `nvidia-smi` works; host `clinfo` has no GPU | Check the OpenCL ICD loader/vendor runtime on the host. |
| Host `nvidia-smi` also fails | Investigate the host driver/kernel before testing the library; see below. |
| GPU appears but lacks fp64 or a source compiler | This GPU/runtime cannot execute this backend. Use a compatible device/runtime. |

In the inspected session, the running 7.3.4 kernel had a corresponding
`/usr/lib/modules` directory, and `/proc/driver/nvidia/version` reported the loaded
NVIDIA Open Kernel Module **620.70.06**. Installed packages were
`nvidia-utils 620.70.06-1`, `opencl-nvidia 620.70.06-1` and
`nvidia-open 620.70.06-3`. Nevertheless, the session exposed **no `/dev/nvidia*`
nodes**, `clinfo` reported zero platforms and `nvidia-smi` could not communicate
with the driver. This is consistent with missing GPU device passthrough; it does
not establish that the host driver is broken. Seeing host kernel/module metadata
does not prove that this session can open the hardware.

No Codex GPU-access toggle has been verified for this environment. Reinstalling
already present host packages cannot by itself expose devices to an isolated
session. Do not create device nodes manually or broadly relax device permissions.

## Build, require execution, then benchmark

From the repository root in the environment where `clinfo` sees the usable GPU:

```sh
cmake -S . -B build-gpu -DNN_ENABLE_GPU=ON -DNN_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu --parallel
NN_REQUIRE_GPU_TESTS=1 ctest --test-dir build-gpu -R accelerator_execution --output-on-failure
NN_REQUIRE_GPU_TESTS=1 ctest --test-dir build-gpu --output-on-failure
./build-gpu/nn_gpu_benchmark
```

The build needs CMake, a C++17 compiler and the OpenCL loader/development library
(`ocl-icd` on Arch). OpenCL headers are bundled in this repository. A vendor GPU
runtime is required at execution time: Arch provides NVIDIA's runtime in
[`opencl-nvidia`](https://archlinux.org/packages/extra/x86_64/opencl-nvidia/).
The loader and vendor runtime are separate components; see
[Arch's OpenCL guidance](https://wiki.archlinux.org/title/GPGPU) and
[`ocl-icd`](https://archlinux.org/packages/extra/x86_64/ocl-icd/).

`NN_REQUIRE_GPU_TESTS=1` makes absent or rejected GPU discovery fail the hardware
test. Without it, CTest marks return code 77 as **Skipped**, which is not physical
GPU verification. For device-specific success output, rerun the focused command
with `--verbose`. A passing hardware test checks CPU/GPU parity, finite
differences, SGD updates, XOR learning and cross-device parameter visibility.
`parallel_kernels`, `kernel_reference` and `opencl_session_mock` run on the host;
their passes do not establish physical GPU execution.

The benchmark prints median timings and a checksum only when an executable GPU
is available. `SKIP` with exit code 77 means GPU performance remains unmeasured.
Its timings are not assertions of a speedup. Record the GPU name, driver/runtime,
build type and benchmark output when comparing results.

## Only if the host checks fail

Collect host evidence before changing packages:

```sh
uname -r
ls /usr/lib/modules
cat /proc/driver/nvidia/version
modinfo -F version nvidia
pacman -Q nvidia-open nvidia-utils opencl-nvidia ocl-icd clinfo
lspci -k -d ::03xx
journalctl -b -k | rg -i 'nvidia|nvrm|nouveau'
```

A missing package reported by `pacman -Q` is a diagnostic; it is not an instruction
to install every named package. If a kernel/driver update preceded the failure,
first reboot the host into its installed matching kernel and retry the checks.
An updated kernel is not running until reboot; see
[Arch troubleshooting after kernel upgrades](https://wiki.archlinux.org/title/General_troubleshooting#Cannot_use_some_peripherals_after_kernel_upgrade).

If problems persist, use the GPU model and kernel from these checks to follow
[Arch's NVIDIA driver selection](https://wiki.archlinux.org/title/NVIDIA) and
[troubleshooting](https://wiki.archlinux.org/title/NVIDIA/Troubleshooting).
Confirm the installed module package targets the running kernel and the loaded
driver agrees with its user-space libraries. For a working NVIDIA driver but
missing OpenCL, check the installed `ocl-icd` loader, `opencl-nvidia` runtime and
its files under `/etc/OpenCL/vendors`, then rerun `clinfo`.

Choose driver packages through EndeavourOS/Arch package management, using the
current support table rather than forcing `nvidia-open`: NVIDIA's open modules
support Turing and newer, and older cards need a suitable supported driver branch.
See [NVIDIA's compatibility list](https://github.com/NVIDIA/open-gpu-kernel-modules#compatible-gpus).
The package versions above describe the inspected session, not versions to pin.
No driver installation or host repair was performed as part of these checks.
