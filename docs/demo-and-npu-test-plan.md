# Demo programs and future NPU hardware tests

Keep the library API unchanged. Applications link nn::nn, own logging and generate
all data in code. Add seeded nonlinear regression and classification demos,
a device-selection/inspection demo, and a CPU-trained/NPU-inference demo alongside
existing XOR. No dataset/model artifacts or application command-line interface.

Register examples as separate optional build targets and smoke tests. CPU training
examples must demonstrate quantitative learning; the NPU example skips with 77
when no compatible NPU is discovered.

Add separately reported physical NPU tests for inference parity, complete MSE,
parameter synchronization/cache invalidation and selection/precision/training
contracts. Use declared mixed absolute/relative FP16 tolerances, test actual NPU
indices, verify selected device, and never label CPU/mock computation as hardware.
Missing NPU skips by default; NN_REQUIRE_NPU_TESTS=1 makes absence a failure.
Register each hardware scenario separately with npu/hardware labels. Existing
SDK-free policy/snapshot/mock tests and conditional OpenVINO CPU graph validation
remain distinct.

Verify default and GPU-enabled builds, CPU demos and all executable software tests;
verify NPU skips and required-hardware failures. No NPU hardware or SDK-enabled
execution claim is made. Document commands and known SDK/hardware limitations.
