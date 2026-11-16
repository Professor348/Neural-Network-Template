#include "backends/openvino/Snapshot.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
/* Require condition with message; no output on success, throws runtime_error on failure, works in Release builds. */
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
/* Require action to throw runtime_error; message describes failure; does not execute any accelerator graph. */
static void rejects(const std::function<void()>& action, const char* message) {
    try { action(); }
    catch (const std::runtime_error& error /* expected unsupported shape or reduced-precision data */) {
        (void)error;
        return;
    }
    throw std::runtime_error(message);
}
/* Test actual SDK-independent conversion, exact snapshot identity and graph validation; return process status, no hardware calls. */
int main() {
    try {
        // Small canonical network includes nontrivial topology, output-major weights and biases.
        const std::vector<nn::Dense> layers{{2, 2, nn::Activation::Tanh, {1.0, 0.25, -0.5, 0.75}, {0.0, -0.25}},
            {2, 1, nn::Activation::Linear, {0.5, -0.5}, {0.125}}};
        // Full identity of original canonical parameters for exact NPU.0 static prediction graph.
        const auto original = nn::detail::npuGraphKey("NPU.0", layers, 3, false);
        require(original == nn::detail::npuGraphKey("NPU.0", layers, 3, false), "identical graph keys differ");
        require(original != nn::detail::npuGraphKey("NPU.1", layers, 3, false), "device missing from cache key");
        require(original != nn::detail::npuGraphKey("NPU.0", layers, 4, false), "batch missing from cache key");
        require(original != nn::detail::npuGraphKey("NPU.0", layers, 3, true), "operation missing from cache key");
        // Mutable canonical copy used to independently change each cache-relevant property.
        auto changed = layers;
        changed.front().weights.front() = std::nextafter(1.0, 2.0);
        require(static_cast<float>(changed.front().weights.front()) == static_cast<float>(layers.front().weights.front()),
            "low canonical bit test must preserve the float32 conversion");
        require(original != nn::detail::npuGraphKey("NPU.0", changed, 3, false), "cache key lost low canonical weight bits");
        changed = layers;
        changed.front().biases.back() = std::nextafter(changed.front().biases.back(), 1.0);
        require(original != nn::detail::npuGraphKey("NPU.0", changed, 3, false), "cache key lost low canonical bias bits");
        changed = layers;
        changed.front().biases.front() = -0.0;
        require(original != nn::detail::npuGraphKey("NPU.0", changed, 3, false), "cache key lost signed zero bits");
        changed = layers;
        changed.front().activation = nn::Activation::Sigmoid;
        require(original != nn::detail::npuGraphKey("NPU.0", changed, 3, false), "activation missing from cache key");
        changed = layers;
        changed.insert(changed.begin() + 1, nn::Dense{2, 2, nn::Activation::Linear, {1, 0, 0, 1}, {0, 0}});
        require(original != nn::detail::npuGraphKey("NPU.0", changed, 3, false), "architecture missing from cache key");
        require(nn::detail::npuFloat(1.25, "test") == 1.25f, "finite float32 conversion failed");
        require(std::signbit(nn::detail::npuFloat(-0.0, "test")), "conversion lost signed zero");
        require(nn::detail::npuFloat(std::numeric_limits<float>::max(), "test") == std::numeric_limits<float>::max(),
            "finite float32 maximum was rejected");
        rejects(
            /* Check rejected float32 overflow conversion; no inputs/outputs or side effects besides expected exception. */
            [] { nn::detail::npuFloat(std::numeric_limits<double>::max(), "test"); }, "float32 overflow was accepted");
        rejects(
            /* Check rejected infinity conversion; no inputs/outputs or side effects besides expected exception. */
            [] { nn::detail::npuFloat(std::numeric_limits<double>::infinity(), "test"); }, "infinite input was accepted");
        rejects(
            /* Check rejected NaN conversion; no inputs/outputs or side effects besides expected exception. */
            [] { nn::detail::npuFloat(std::numeric_limits<double>::quiet_NaN(), "test"); }, "NaN input was accepted");
        rejects(
            /* Check zero batch for captured layers; no output, expected runtime_error. */
            [&layers] { nn::detail::npuValidateGraph(layers, 0); }, "zero static batch was accepted");
        rejects(
            /* Check oversized static tensor for captured layers; no output, expected runtime_error. */
            [&layers] { nn::detail::npuValidateGraph(layers, std::numeric_limits<std::size_t>::max()); },
            "overflowed static batch was accepted");
        rejects(
            /* Check empty architecture; no input/output, expected runtime_error. */
            [] { nn::detail::npuValidateGraph({}, 1); }, "empty architecture was accepted");
        changed = layers;
        changed.back().inputs = 1;
        rejects(
            /* Check captured inconsistent topology; no output, expected runtime_error. */
            [&changed] { nn::detail::npuValidateGraph(changed, 3); }, "inconsistent topology was accepted");
        changed = layers;
        changed.front().weights.pop_back();
        rejects(
            /* Check captured truncated weights; no output, expected runtime_error. */
            [&changed] { nn::detail::npuValidateGraph(changed, 3); }, "truncated weights were accepted");
        changed = layers;
        changed.front().biases.clear();
        rejects(
            /* Check captured missing biases; no output, expected runtime_error. */
            [&changed] { nn::detail::npuValidateGraph(changed, 3); }, "missing biases were accepted");
        changed = layers;
        changed.front().activation = static_cast<nn::Activation>(-1);
        rejects(
            /* Check captured unsupported activation; no output, expected runtime_error. */
            [&changed] { nn::detail::npuValidateGraph(changed, 3); }, "unknown activation was accepted");
        changed = layers;
        changed.front().weights.front() = std::numeric_limits<double>::max();
        rejects(
            /* Check captured parameter range rejection, including the cache-key path; no output, expected runtime_error. */
            [&changed] { nn::detail::npuGraphKey("NPU", changed, 3, false); }, "non-float32 model was accepted");
        std::cout << "NPU snapshot/conversion checks passed; no OpenVINO SDK or NPU execution involved\n";
        return 0;
    } catch (const std::exception& error /* propagate meaningful standalone test failure */) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
