// SDK-free production dispatch integration test. Sentinel results are NOT NPU hardware results.
#include "backends/Execution.h"
#include "backends/openvino/NpuSession.h"
#include "npu_dispatch_mock_state.h"
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
// Number of attempted fake graph preparations, including intentionally rejected graphs.
std::size_t preparations = 0;
// Number of fake prediction invocations after successful preparation.
std::size_t predictions = 0;
// Number of fake complete loss graph invocations.
std::size_t losses = 0;
// Whether fake graph preparation must reject the entire model before numerical execution.
bool rejectGraph = false;
// Whether already prepared fake requests must throw during inference.
bool rejectInference = false;

/* Require condition; message describes the failed invariant. Throws runtime_error on failure. */
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

/* Require action to throw runtime_error containing fragment; both arguments describe the expected failure. */
void requireThrows(const std::function<void()>& action, const char* fragment) {
    try { action(); }
    catch (const std::runtime_error& error /* actual production dispatch diagnostic */) {
        require(std::string(error.what()).find(fragment) != std::string::npos, "Unexpected rejection diagnostic");
        return;
    }
    throw std::runtime_error("Expected runtime_error was not propagated");
}

/* Return whether selection records fragment in a higher-priority rejection diagnostic; no execution occurs. */
bool skipped(const nn::Selection& selection, const char* fragment) {
    // Earlier candidate rejection whose explanation is inspected by this assertion.
    for (const auto& reason : selection.skipped)
        if (reason.find(fragment) != std::string::npos) return true;
    return false;
}
}

namespace nn::detail {
/* Return one inference-only fake NPU descriptor; no discovery, SDK, driver or hardware is involved. */
std::vector<DeviceInfo> npuDevices() {
    return {{DeviceType::NPU, 0, "Fake dispatch NPU", "test-only-NOT-NPU-hardware", false, true}};
}

/* Return the fake inventory diagnostic; no runtime loading or numerical execution occurs. */
std::string npuDiagnostic() { return "Test-only inventory; NOT NPU hardware"; }

/* Prepare fake index with copied layers, exact samples, loss graph flag, and required precision permission.
 * Throws before execution when rejectGraph is set; this substitutes only the SDK boundary. */
NpuSession::NpuSession(std::size_t index, const std::vector<Dense>& layers, std::size_t samples,
    bool loss, bool allowReducedPrecision) : state_(std::make_unique<NpuState>()) {
    ++preparations;
    require(index == 0 && allowReducedPrecision, "Dispatch supplied invalid NPU preparation options");
    if (rejectGraph) throw std::runtime_error("simulated graph compilation failure");
    state_->layers = layers;
    state_->samples = samples;
    state_->withLoss = loss;
}

/* Release this test session's private snapshot; performs no SDK or hardware work. */
NpuSession::~NpuSession() = default;

/* Accept inputs of the prepared batch size and return parameter-dependent sentinel rows; no CPU numerics run. */
Dataset NpuSession::predict(const Dataset& inputs) {
    ++predictions;
    if (rejectInference) throw std::runtime_error("simulated inference runtime failure");
    require(inputs.size() == state_->samples, "Dispatch changed the prepared batch size");
    return Dataset(inputs.size(), {1000 + state_->layers.front().weights.front() + state_->layers.front().biases.front()});
}

/* Accept inputs/targets only for a complete prepared loss graph and return its unique sentinel scalar. */
double NpuSession::loss(const Dataset& inputs, const Dataset& targets) {
    ++losses;
    if (rejectInference) throw std::runtime_error("simulated loss runtime failure");
    require(state_->withLoss && inputs.size() == state_->samples && targets.size() == inputs.size(),
        "Dispatch did not prepare or invoke the complete loss graph");
    return 9000 + state_->layers.front().weights.front();
}
}

/* Exercise real Execution/Selection/CPU code against only a fake NPU SDK boundary; no NPU hardware is tested. */
int main() {
    // Tiny canonical CPU model whose real prediction is 7, distinct from every fake NPU sentinel.
    std::vector<nn::Dense> layers{{1, 1, nn::Activation::Linear, {2}, {1}}};
    // One sample fixes the production execution's static batch.
    const nn::Dataset inputs{{3}};
    // One target yields a real CPU mean squared error of 36.
    const nn::Dataset targets{{1}};
    // Explicit opt-in required before constructing any NPU session.
    const nn::NpuOptions reduced{true};
    // Default full-precision AUTO must reject fake NPU before its constructor runs.
    nn::detail::Execution fullPrecision({}, nn::Operation::Inference, layers, 1, false);
    require(fullPrecision.selection().device.type == nn::DeviceType::CPU && preparations == 0 &&
        skipped(fullPrecision.selection(), "setNpuOptions"), "AUTO did not report precision rejection before CPU selection");
    require(fullPrecision.predict(inputs) == nn::Dataset{{7}}, "AUTO CPU execution returned incorrect output");
    /* Attempt explicit NPU without precision permission; captured layers supplies the unchanged canonical model. */
    requireThrows([&layers] {
        // Construction itself must reject this request before any numerical execution.
        nn::detail::Execution execution({nn::DeviceType::NPU, 0}, nn::Operation::Inference, layers, 1, false);
    }, "setNpuOptions");
    require(preparations == 0, "Default precision constructed an NPU session");

    // Opted-in AUTO must use a successfully prepared fake NPU session.
    nn::detail::Execution first({}, nn::Operation::Inference, layers, 1, false, reduced);
    require(first.selection().device.type == nn::DeviceType::NPU && preparations == 1 && predictions == 0,
        "AUTO did not select prepared NPU before inference");
    require(first.predict(inputs) == nn::Dataset{{1003}}, "Opted-in prediction fell through to CPU numerics");
    layers.front().weights.front() = 5;
    // A later explicit session must capture changed canonical parameters privately.
    nn::detail::Execution latest({nn::DeviceType::NPU, 0}, nn::Operation::Inference, layers, 1, false, reduced);
    require(latest.predict(inputs) == nn::Dataset{{1006}} && first.predict(inputs) == nn::Dataset{{1003}},
        "Sessions did not capture independent latest-parameter snapshots");
    layers.front().weights.front() = 2;

    rejectGraph = true;
    // AUTO may skip a preparation failure before any backend numerical invocation.
    const auto predictionsBeforeCompile = predictions;
    nn::detail::Execution compileFallback({}, nn::Operation::Inference, layers, 1, false, reduced);
    require(compileFallback.selection().device.type == nn::DeviceType::CPU &&
        skipped(compileFallback.selection(), "simulated graph compilation failure") && predictions == predictionsBeforeCompile,
        "AUTO did not skip graph compilation failure before execution");
    require(compileFallback.predict(inputs) == nn::Dataset{{7}}, "Compile rejection did not bind real CPU execution");
    /* Explicit NPU preparation failure must propagate; layers/reduced define the requested complete model. */
    requireThrows([&layers, reduced] {
        // This explicitly requested graph cannot be prepared and must never bind CPU execution.
        nn::detail::Execution execution({nn::DeviceType::NPU, 0}, nn::Operation::Inference, layers, 1, false, reduced);
    }, "simulated graph compilation failure");
    rejectGraph = false;

    rejectInference = true;
    /* A successfully selected AUTO NPU must propagate a later inference failure; first/inputs are prepared request data. */
    requireThrows([&first, &inputs] { first.predict(inputs); }, "simulated inference runtime failure");
    require(first.selection().device.type == nn::DeviceType::NPU, "Runtime failure changed the bound backend");
    rejectInference = false;

    // Training capability rejection must happen before preparation despite opted-in precision.
    const auto preparationsBeforeTraining = preparations;
    nn::detail::Execution training({}, nn::Operation::Training, layers, 1, false, reduced);
    require(training.selection().device.type == nn::DeviceType::CPU &&
        skipped(training.selection(), "unsupported requested operation"), "AUTO did not skip inference-only NPU for training");
    /* Explicit training must fail without NPU preparation; layers/reduced identify the otherwise compatible model. */
    requireThrows([&layers, reduced] {
        // Inference-only capability must reject this explicit training execution during selection.
        nn::detail::Execution execution({nn::DeviceType::NPU, 0}, nn::Operation::Training, layers, 1, false, reduced);
    }, "unsupported requested operation");
    require(preparations == preparationsBeforeTraining, "Training attempted NPU graph preparation");
    /* Inference-bound NPU must reject gradient requests; first/inputs/targets describe the invalid training call. */
    requireThrows([&first, &inputs, &targets] { first.gradient(inputs, targets); }, "inference only");
    /* Inference-bound NPU must reject updates; first/inputs/targets describe the invalid training call. */
    requireThrows([&first, &inputs, &targets] { first.step(inputs, targets, 0.1); }, "training is unsupported");

    // Complete NPU loss preparation and invocation must use its own graph instead of CPU or prediction fallthrough.
    nn::detail::Execution lossGraph({}, nn::Operation::Inference, layers, 1, true, reduced);
    // Prediction count must remain fixed during both successful and failed complete-loss invocations.
    const auto predictionsBeforeLoss = predictions;
    require(lossGraph.loss(inputs, targets) == 9002 && losses == 1 && predictions == predictionsBeforeLoss,
        "withLoss did not execute the complete NPU loss graph");
    rejectInference = true;
    /* A loss runtime failure must propagate without CPU recomputation; lossGraph/inputs/targets identify the prepared graph. */
    requireThrows([&lossGraph, &inputs, &targets] { lossGraph.loss(inputs, targets); }, "simulated loss runtime failure");
    require(losses == 2 && predictions == predictionsBeforeLoss, "Loss runtime failure retried a different numerical path");
    std::cout << "Production NPU dispatch integration passed (mock SDK boundary; NOT NPU hardware).\n";
}
