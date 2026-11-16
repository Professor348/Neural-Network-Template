#include "NpuSession.h"
#include "NpuRuntime.h"
#include "NpuState.h"
#include <new>
#include <stdexcept>
namespace nn::detail {
/* Return discovered NPU-only descriptors; no model/inference runs; Core initialization errors yield an empty inventory. */
std::vector<DeviceInfo> npuDevices() {
    try { return npuRuntime().devices(); }
    catch (const std::bad_alloc& error /* preserve host memory exhaustion rather than publishing a false empty inventory */) {
        (void)error;
        throw;
    }
    catch (const std::exception& error /* runtime not initialized; npuDiagnostic retains its concrete failure on query */) {
        (void)error;
        return {};
    }
}
/* Return discovery diagnostics, including Core initialization errors; no numerical execution or device fallback occurs. */
std::string npuDiagnostic() {
    try { return npuRuntime().diagnostic(); }
    catch (const std::bad_alloc& error /* preserve host memory exhaustion rather than classifying it as runtime absence */) {
        (void)error;
        throw;
    }
    catch (const std::exception& error /* initialization failed before an inventory could be published */) {
        return std::string("NPU OpenVINO runtime unavailable; hardware/driver status unknown: ") + error.what();
    }
}
/* Prepare index/layers/samples/loss with explicit allowReducedPrecision consent; full compile precedes any request execution.
 * Wrap SDK errors as runtime_error so callers may reject AUTO candidates before selecting/executing a device. */
NpuSession::NpuSession(std::size_t index, const std::vector<Dense>& layers, std::size_t samples, bool loss,
    bool allowReducedPrecision) {
    if (!allowReducedPrecision)
        throw std::runtime_error("NPU precision unsupported: canonical float64 execution is unavailable; "
            "explicitly enable NpuOptions.allowReducedPrecision to permit float32 I/O/weights and float16 NPU arithmetic");
    try {
        // Complete model support, static shapes, precision and exclusive placement are established without inference.
        const auto compiled = npuRuntime().compiled(index, layers, samples, loss);
        state_ = std::make_unique<NpuState>(compiled, samples, layers.front().inputs, layers.back().outputs, loss);
    } catch (const std::bad_alloc& error /* host allocation failure must not become an unsupported AUTO candidate */) {
        (void)error;
        throw;
    } catch (const std::exception& error /* unsupported model, compilation or request allocation failure before any inference */) {
        throw std::runtime_error(std::string("NPU operation preparation failed before inference: ") + error.what());
    }
}
/* Release implementation/private request in SDK-aware translation unit; shared compiled snapshots remain cached; no exceptions. */
NpuSession::~NpuSession() = default;
/* Infer prepared prediction graph for inputs and return widened outputs; errors propagate after selection without fallback. */
Dataset NpuSession::predict(const Dataset& inputs) { return state_->predict(inputs); }
/* Infer entire prepared MSE graph for inputs/targets and return its device scalar; errors propagate without fallback. */
double NpuSession::loss(const Dataset& inputs, const Dataset& targets) { return state_->loss(inputs, targets); }
}
