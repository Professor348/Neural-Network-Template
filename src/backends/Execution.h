#pragma once
#include <nn/Dense.h>
#include <nn/DeviceInfo.h>
#include <nn/DevicePreference.h>
#include <nn/Selection.h>
#include <nn/NpuOptions.h>
#include <memory>
namespace nn::detail {
class OpenClSession;
class NpuSession;
class Execution {
public:
    /* Resolve complete model execution before numerics. preference: call/configured device; operation: training/inference;
     * layers: canonical model; samples: static batch; withLoss: compile full loss graph when true; npu: precision opt-in.
     * AUTO may skip graph compilation incompatibilities before execution; explicit requests throw instead of falling back. */
    Execution(DevicePreference preference, Operation operation, const std::vector<Dense>& layers,
        std::size_t samples, bool withLoss, NpuOptions npu = {});
    /* Return actual selection and pre-execution skip reasons; does not change configured network preference. */
    const Selection& selection() const;
    /* Destroy internal backend session; no exceptions. */
    ~Execution();
    /* Run complete inference for inputs on bound device; returns samples by outputs or throws numerical/runtime error. */
    Dataset predict(const Dataset& inputs);
    /* Run complete MSE reduction for inputs/targets on bound device; returns scalar or throws numerical/runtime error. */
    double loss(const Dataset& inputs, const Dataset& targets);
    /* Compute mean-loss derivatives for inputs/targets on bound device; returns parameter-order gradient or throws. */
    std::vector<double> gradient(const Dataset& inputs, const Dataset& targets);
    /* Execute one atomic batch SGD update for inputs/targets with rate; returns new parameters for canonical host sync.
     * GPU session mutates only on successful update; CPU staging does not mutate layers. Throws without retry. */
    std::vector<double> step(const Dataset& inputs, const Dataset& targets, double rate);
private:
    // Actual resolved device and higher-priority rejection reasons.
    Selection selection_;
    // Current canonical layers; caller synchronizes returned SGD parameters after each successful batch.
    const std::vector<Dense>& layers_;
    // Optional real GPU session; numerical work remains entirely on the resolved device.
#ifdef NN_ENABLE_GPU
    std::unique_ptr<OpenClSession> gpu_;
#endif
#ifdef NN_ENABLE_NPU
    // Optional inference-only NPU session, fully compiled before numerical execution.
    std::unique_ptr<NpuSession> npu_;
#endif
};
}
