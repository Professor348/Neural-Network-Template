#include "Execution.h"
#include "Cpu.h"
#include "numerics/Functions.h"
#include <stdexcept>
#include "devices/Selection.h"
#ifdef NN_ENABLE_NPU
#include "openvino/NpuSession.h"
#endif
#ifdef NN_ENABLE_GPU
#include "opencl/OpenClSession.h"
#endif
namespace nn::detail {
/* Resolve preference/operation for layers and samples, optionally including loss and reduced-precision NPU permission.
 * Compiles NPU graphs before selection; only those compatibility failures may be skipped by AUTO before execution. */
Execution::Execution(DevicePreference preference, Operation operation, const std::vector<Dense>& layers,
    std::size_t samples, bool withLoss, NpuOptions npu) : layers_(layers) {
#ifndef NN_ENABLE_NPU
    (void)samples;
    (void)withLoss;
#endif
    /* Check candidate for complete-model compatibility before any numerical execution; returns a reason or accepts it.
     * candidate: basic-usable device; captures session/model/static shape and explicit NPU precision permission. */
    const auto suitable = [&](const DeviceInfo& candidate) -> std::optional<std::string> {
        if (candidate.type != DeviceType::NPU) return std::nullopt;
        if (!npu.allowReducedPrecision) return "NPU requires setNpuOptions({true}) to permit f32 I/O and FP16 internal inference";
#ifdef NN_ENABLE_NPU
        try { npu_ = std::make_unique<NpuSession>(candidate.index, layers, samples, withLoss, true); }
        catch (const std::runtime_error& error /* complete-model preparation failed before execution */) { return error.what(); }
        return std::nullopt;
#else
        return "NPU backend disabled at build time";
#endif
    };
    selection_ = resolve(preference, operation, suitable);
    if (selection_.device.type == DeviceType::CPU) return;
#ifdef NN_ENABLE_GPU
    if (selection_.device.type == DeviceType::GPU) {
        gpu_ = std::make_unique<OpenClSession>(selection_.device.index, layers);
        return;
    }
#endif
#ifdef NN_ENABLE_NPU
    if (selection_.device.type == DeviceType::NPU && npu_) return;
#endif
    throw std::runtime_error("Selected execution backend is disabled");
}
/* Return the device that will execute this entire operation and reasons higher-priority candidates were skipped. */
const Selection& Execution::selection() const { return selection_; }
/* Release backend session; GPU release cannot throw. */
Execution::~Execution() = default;
/* Infer inputs on the selected device; returns row-major output matrix; propagates backend failures. */
Dataset Execution::predict(const Dataset& inputs) {
#ifdef NN_ENABLE_NPU
    if (npu_) return npu_->predict(inputs);
#endif
#ifdef NN_ENABLE_GPU
    if (gpu_) {
        // Output rows in caller's sample order; allocate row metadata once.
        Dataset result;
        result.reserve(inputs.size());
        // Flat output buffer computed entirely on the selected GPU.
        const auto flat = gpu_->run(inputs, {}, 0);
        // Width of each returned row.
        const std::size_t width = layers_.back().outputs;
        // Sample position used only to format copied output data.
        for (std::size_t sample = 0; sample < inputs.size(); ++sample)
            result.emplace_back(flat.begin() + sample * width, flat.begin() + (sample + 1) * width);
        return result;
    }
#endif
    return predictBatch(layers_, inputs);
}
/* Return complete mean loss for inputs/targets using the bound backend; no fallback or parameter mutation. */
double Execution::loss(const Dataset& inputs, const Dataset& targets) {
#ifdef NN_ENABLE_NPU
    if (npu_) return npu_->loss(inputs, targets);
#endif
#ifdef NN_ENABLE_GPU
    if (gpu_) return gpu_->run(inputs, targets, 1).front();
#endif
    return meanLoss(layers_, inputs, targets);
}
/* Return complete parameter gradient for inputs/targets on the bound backend; no model mutation. */
std::vector<double> Execution::gradient(const Dataset& inputs, const Dataset& targets) {
#ifdef NN_ENABLE_NPU
    if (npu_) throw std::runtime_error("OpenVINO NPU supports inference only; gradients are unsupported");
#endif
#ifdef NN_ENABLE_GPU
    if (gpu_) return gpu_->run(inputs, targets, 2);
#endif
    return batchGradient(layers_, inputs, targets);
}
/* Stage atomic SGD with inputs/targets and rate on bound backend; returns updated parameter vector for host synchronization. */
std::vector<double> Execution::step(const Dataset& inputs, const Dataset& targets, double rate) {
#ifdef NN_ENABLE_NPU
    if (npu_) throw std::runtime_error("OpenVINO NPU supports inference only; training is unsupported");
#endif
#ifdef NN_ENABLE_GPU
    if (gpu_) return gpu_->run(inputs, targets, 3, rate);
#endif
    // CPU reference parameter gradient.
    const auto derivatives = batchGradient(layers_, inputs, targets);
    // Staged candidate parameter vector, returned only after full validation.
    std::vector<double> updated;
    updated.reserve(derivatives.size());
    // Layer whose parameters are flattened in canonical order.
    for (const auto& layer : layers_) {
        updated.insert(updated.end(), layer.weights.begin(), layer.weights.end());
        updated.insert(updated.end(), layer.biases.begin(), layer.biases.end());
    }
    // Parameter coordinate for SGD and finiteness validation.
    for (std::size_t index = 0; index < updated.size(); ++index) {
        updated[index] -= rate * derivatives[index];
        requireFinite(updated[index]);
    }
    return updated;
}
}
