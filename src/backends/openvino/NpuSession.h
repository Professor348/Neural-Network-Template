#pragma once
#include <nn/Dense.h>
#include <nn/DeviceInfo.h>
#include <memory>
namespace nn::detail {
class NpuState;
/* Return discovered physical NPU descriptors in lexicographic OpenVINO runtime-ID order.
 * No inference runs; model-specific support is checked by NpuSession; runtime failures return an empty vector. */
std::vector<DeviceInfo> npuDevices();
/* Return discovery/index diagnostics; no inference runs, and missing hardware and unknown driver state are distinguished. */
std::string npuDiagnostic();
class NpuSession {
public:
    /* Prepare an inference-only session: index selects the discovered NPU, layers supplies current canonical
     * double parameters, samples fixes the exact batch, loss selects a complete MSE graph, and
     * allowReducedPrecision explicitly permits float32 I/O/weights and float16 NPU arithmetic.
     * Validates the entire graph, queries support and compiles before any inference; throws runtime_error
     * on unsupported precision/capability/runtime, without executing numerics or using another device. */
    NpuSession(std::size_t index, const std::vector<Dense>& layers, std::size_t samples, bool loss,
        bool allowReducedPrecision);
    /* Release this session's private request; shared bounded compiled-model snapshots remain cached; does not throw. */
    ~NpuSession();
    /* Prevent copying mutable request ownership; other is the session that cannot be copied. */
    NpuSession(const NpuSession& other) = delete;
    /* Prevent copying mutable request ownership; other is the session that cannot be assigned; no output. */
    NpuSession& operator=(const NpuSession& other) = delete;
    /* Infer the prepared prediction graph for inputs of the exact prepared shape; return widened float32 outputs.
     * Serializes use of this private request; throws on invalid data, runtime errors or nonfinite results; no fallback. */
    Dataset predict(const Dataset& inputs);
    /* Infer the prepared MSE graph for inputs/targets of the exact prepared shapes; return the device-reduced scalar.
     * Serializes use of this private request; throws on invalid data, runtime errors or nonfinite results; no fallback. */
    double loss(const Dataset& inputs, const Dataset& targets);
private:
    // Implementation owns the private request and hides every SDK type from this dispatch-facing header.
    std::unique_ptr<NpuState> state_;
};
}
