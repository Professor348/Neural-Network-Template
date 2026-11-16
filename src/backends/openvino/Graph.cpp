#include "Graph.h"
#include "Snapshot.h"
#include <openvino/opsets/opset13.hpp>
#include <stdexcept>
namespace nn::detail {
/* Convert canonical values to owned float32 constants; values is validated parameter data; returns storage, allocation may throw. */
static std::vector<float> constants(const std::vector<double>& values) {
    // Temporary converted storage is copied into OpenVINO Constant ownership by the graph builder.
    std::vector<float> converted;
    converted.reserve(values.size());
    // Canonical scalar being converted under the session's explicit precision consent.
    for (const double value : values) converted.push_back(npuFloat(value, "parameter"));
    return converted;
}
/* Build validated static float32 dense/MSE graph from layers/samples/loss; returns owned model, never executes inference. */
std::shared_ptr<ov::Model> buildNpuGraph(const std::vector<Dense>& layers, std::size_t samples, bool loss) {
    npuValidateGraph(layers, samples);
    // Static batch-by-feature input parameter; only host I/O conversion uses the CPU.
    const auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32,
        ov::Shape{samples, layers.front().inputs});
    // Complete graph input list, extended with labels only for the MSE operation.
    ov::ParameterVector parameters{input};
    // Current activation tensor; its feature width is determined statically at every layer.
    ov::Output<ov::Node> value = input;
    // Layer whose output-major canonical matrix is transposed by MatMul's transpose_b flag.
    for (const auto& layer : layers) {
        // Constants own converted copies of the current canonical weights and biases.
        const auto weights = ov::opset13::Constant::create(ov::element::f32,
            ov::Shape{layer.outputs, layer.inputs}, constants(layer.weights));
        // Broadcast across samples, retaining one independent bias per output coordinate.
        const auto biases = ov::opset13::Constant::create(ov::element::f32,
            ov::Shape{layer.outputs}, constants(layer.biases));
        // Batch-by-output matrix product remains part of the single NPU graph.
        const auto product = std::make_shared<ov::opset13::MatMul>(value, weights, false, true);
        value = std::make_shared<ov::opset13::Add>(product, biases);
        switch (layer.activation) {
        case Activation::Linear: break;
        case Activation::Sigmoid: value = std::make_shared<ov::opset13::Sigmoid>(value); break;
        case Activation::Tanh: value = std::make_shared<ov::opset13::Tanh>(value); break;
        case Activation::Relu: value = std::make_shared<ov::opset13::Relu>(value); break;
        default: throw std::runtime_error("NPU unsupported graph: unknown activation");
        }
    }
    if (loss) {
        // Labels enter the same compiled graph, preventing host evaluation or host reduction of MSE.
        const auto target = std::make_shared<ov::opset13::Parameter>(ov::element::f32,
            ov::Shape{samples, layers.back().outputs});
        parameters.push_back(target);
        // Residual, elementwise square and both-axis mean are exclusively graph operations.
        const auto difference = std::make_shared<ov::opset13::Subtract>(value, target);
        // Squaring avoids requiring a general-purpose Power implementation on the NPU.
        const auto squared = std::make_shared<ov::opset13::Multiply>(difference, difference);
        // Reduce across batch and output features; keep static rank-two [1,1] output for NPU compatibility.
        const auto axes = ov::opset13::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        value = std::make_shared<ov::opset13::ReduceMean>(squared, axes, true);
    }
    // Single output is either the entire prediction matrix or the device-computed [1,1] MSE.
    const auto result = std::make_shared<ov::opset13::Result>(value);
    // Graph exclusively contains static dense/activation nodes and optional complete MSE nodes.
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, parameters,
        loss ? "nn_npu_mse" : "nn_npu_predict");
    // Stable unique friendly-name ordinal allows complete query_model coverage checking without aliases.
    std::size_t ordinal = 0;
    // Node receiving its deterministic unique query_model identity.
    for (const auto& node : model->get_ordered_ops()) node->set_friendly_name("nn_node_" + std::to_string(ordinal++));
    model->validate_nodes_and_infer_types();
    return model;
}
}
