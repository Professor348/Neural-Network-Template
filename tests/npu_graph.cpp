// Graph validation on the OpenVINO CPU plugin only. This test is not evidence of NPU compilation or execution.
#include "backends/openvino/Graph.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
/* Require condition with message in Debug and Release; throws runtime_error on failure, no output on success. */
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
/* Return the independent scalar reference activation for value/kind; no mutation, throws for unsupported activation. */
static double activate(double value, nn::Activation kind) {
    switch (kind) {
    case nn::Activation::Linear: return value;
    case nn::Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-value));
    case nn::Activation::Tanh: return std::tanh(value);
    case nn::Activation::Relu: return std::max(0.0, value);
    }
    throw std::runtime_error("unknown reference activation");
}
/* Compute independent CPU reference for layers and data, returning predictions; test-only, never used by production dispatch. */
static nn::Dataset reference(const std::vector<nn::Dense>& layers, const nn::Dataset& data) {
    // Independent reference sample storage, transformed one layer at a time.
    auto values = data;
    // Reference dense layer under evaluation.
    for (const auto& layer : layers) {
        // Expected per-sample outputs for the current reference layer.
        nn::Dataset next(values.size(), std::vector<double>(layer.outputs));
        // Sample whose dense dot products are independently evaluated.
        for (std::size_t sample = 0; sample < values.size(); ++sample) {
            // Output-major neuron index.
            for (std::size_t output = 0; output < layer.outputs; ++output) {
                // Reference affine value uses the same explicit f32 storage as the graph constants.
                double sum = static_cast<float>(layer.biases[output]);
                // Input coordinate of the current independent dot product.
                for (std::size_t input = 0; input < layer.inputs; ++input)
                    sum += static_cast<float>(layer.weights[output * layer.inputs + input]) * values[sample][input];
                next[sample][output] = activate(sum, layer.activation);
            }
        }
        values = std::move(next);
    }
    return values;
}
/* Execute graph on explicit CPU plugin with core and input/optional target data, returning flat f32 output.
 * This test-only path validates graph semantics; it never constructs a production NPU session or reports NPU success. */
static std::vector<float> evaluate(ov::Core& core, const std::shared_ptr<ov::Model>& graph,
    const nn::Dataset& inputs, const nn::Dataset& targets = {}) {
    // Explicit CPU target is restricted to this graph-validation test; production uses exclusively physical NPU IDs.
    auto compiled = core.compile_model(graph, "CPU", ov::hint::inference_precision(ov::element::f32),
        ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    // Independent request for each test evaluation.
    auto request = compiled.create_infer_request();
    // Input port currently filled, including targets only for a loss graph.
    for (std::size_t port = 0; port < graph->inputs().size(); ++port) {
        // Matching input dataset, selected by the model's preserved parameter order.
        const auto& data = port == 0 ? inputs : targets;
        // Writable storage supplied by this test request.
        float* destination = request.get_input_tensor(port).data<float>();
        // Next sample-major tensor element to upload.
        std::size_t offset = 0;
        // Sample under test.
        for (const auto& row : data) {
            // Feature copied directly into f32 graph I/O.
            for (const double value : row) destination[offset++] = static_cast<float>(value);
        }
        require(offset == request.get_input_tensor(port).get_size(), "test data shape differs from graph");
    }
    request.infer();
    // Device-computed graph output is copied before destroying the private test request.
    const auto output = request.get_output_tensor();
    // Read-only view into the request's float32 output storage.
    const float* source = output.data<const float>();
    return {source, source + output.get_size()};
}
/* Validate all dense activations, output-major layout, static batches, complete MSE reduction and immutable constants.
 * Return 77 only if CPU plugin is absent; successful execution expressly does not validate any NPU driver/hardware. */
int main() {
    try {
        // Test-only Core; production runtime and compiled-model cache are deliberately not used.
        ov::Core core;
        // Available runtime names determine whether the optional CPU graph-validation environment exists.
        const auto devices = core.get_available_devices();
        if (std::find(devices.begin(), devices.end(), "CPU") == devices.end()) {
            std::cout << "SKIP: OpenVINO CPU plugin unavailable for graph validation; no NPU execution attempted\n";
            return 77;
        }
        // Every supported activation is validated across nonlinear hidden and output layers.
        const nn::Activation activations[]{nn::Activation::Linear, nn::Activation::Sigmoid,
            nn::Activation::Tanh, nn::Activation::Relu};
        // Current activation under graph-semantic validation.
        for (const auto activation : activations) {
            // Asymmetric output-major matrices expose accidental transposes or bias broadcasting mistakes.
            std::vector<nn::Dense> layers{{2, 3, activation, {0.5, -0.25, 0.125, 0.75, -0.5, 0.25}, {-0.25, 0.125, 0.5}},
                {3, 2, activation, {0.25, -0.5, 0.75, -0.125, 0.5, 0.25}, {0.125, -0.25}}};
            // Static batches include the single-row boundary and a mean over multiple samples/features.
            const nn::Dataset batches[]{{{0.25, -0.5}}, {{0.25, -0.5}, {-0.75, 0.125}, {0.5, 0.75}}};
            // Current exact-size batch, with all features exactly representable as float32.
            for (const auto& inputs : batches) {
                // Dense reference is independent of OpenVINO graph construction and execution.
                const auto expected = reference(layers, inputs);
                // Actual CPU-plugin result from the production graph builder.
                const auto actual = evaluate(core, nn::detail::buildNpuGraph(layers, inputs.size(), false), inputs);
                require(actual.size() == inputs.size() * 2, "prediction shape mismatch");
                // Separate targets prevent accidentally reducing predictions or using incorrect sample scaling.
                nn::Dataset targets(inputs.size(), std::vector<double>{0.125, -0.25});
                // Independent double reference sum is test-only; production loss is entirely a graph reduction.
                double mse = 0;
                // Flattened prediction position checked against the independent reference.
                std::size_t offset = 0;
                // Reference output row compared coordinate-by-coordinate.
                for (const auto& row : expected) {
                    // Output coordinate within this reference row.
                    for (std::size_t coordinate = 0; coordinate < row.size(); ++coordinate) {
                        require(std::abs(actual[offset++] - row[coordinate]) < 2e-5, "dense/activation graph mismatch");
                        // Expected residual is independently squared and accumulated only inside this test.
                        const double difference = row[coordinate] - targets.front()[coordinate];
                        mse += difference * difference;
                    }
                }
                mse /= static_cast<double>(inputs.size() * 2);
                // Actual loss graph contains two inputs and device-side reduction of both batch and feature axes.
                const auto lossGraph = nn::detail::buildNpuGraph(layers, inputs.size(), true);
                require(lossGraph->inputs().size() == 2 && lossGraph->output().get_shape() == ov::Shape{1, 1},
                    "loss graph must include targets and a single static reduced result");
                // The CPU plugin runs the same graph in this test, validating graph semantics only.
                const auto loss = evaluate(core, lossGraph, inputs, targets);
                require(loss.size() == 1 && std::abs(loss.front() - mse) < 2e-5, "complete MSE graph mismatch");
            }
        }
        // A graph must own its parameter constants after the canonical source model changes.
        std::vector<nn::Dense> layers{{1, 1, nn::Activation::Linear, {0.5}, {0.25}}};
        // Immutable old graph snapshot built before changing canonical parameters.
        const auto oldGraph = nn::detail::buildNpuGraph(layers, 1, false);
        layers.front().weights.front() = 1.5;
        require(std::abs(evaluate(core, oldGraph, {{0.5}}).front() - 0.5f) < 1e-6f,
            "graph constants retained mutable canonical storage");
        require(std::abs(evaluate(core, nn::detail::buildNpuGraph(layers, 1, false), {{0.5}}).front() - 1.0f) < 1e-6f,
            "new graph did not include current canonical weights");
        std::cout << "OpenVINO CPU graph validation passed; NPU compilation/execution NOT tested\n";
        return 0;
    } catch (const std::exception& error /* SDK/graph semantic failure is a real failure, never a claimed NPU success */) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
