#include "Cpu.h"
#include "numerics/Functions.h"
#include <algorithm>
#include <cmath>
namespace nn::detail {
namespace {
/* Allocate one output row per layer in layers; returns per-call scratch reused across all samples. */
Dataset activationWorkspace(const std::vector<Dense>& layers) {
    // Layer output rows, excluding the caller-owned input row.
    Dataset values;
    values.reserve(layers.size());
    // Layer whose activation storage is allocated once.
    for (const auto& layer : layers) values.emplace_back(layer.outputs);
    return values;
}
/* Write input through layers into pre-sized values; returns final output by reference; throws on nonfinite arithmetic. */
const std::vector<double>& forwardInto(const std::vector<Dense>& layers, const std::vector<double>& input, Dataset& values) {
    // Incoming row points to caller-owned input, then each preceding layer's scratch row.
    const std::vector<double>* incoming = &input;
    // Forward layer coordinate.
    for (std::size_t index = 0; index < layers.size(); ++index) {
        // Dense transformation for this layer.
        const auto& layer = layers[index];
        // Preallocated activation row for this layer.
        auto& output = values[index];
        // Output neuron coordinate; groups expose independent sums without changing reduction order.
        std::size_t row = 0;
        for (; row + 3 < layer.outputs; row += 4) {
            // First neuron's contiguous weights in this group.
            const double* first = layer.weights.data() + row * layer.inputs;
            // Second neuron's contiguous weights in this group.
            const double* second = first + layer.inputs;
            // Third neuron's contiguous weights in this group.
            const double* third = second + layer.inputs;
            // Fourth neuron's contiguous weights in this group.
            const double* fourth = third + layer.inputs;
            // First neuron's sum, reduced in ascending input order.
            double sum0 = layer.biases[row];
            // Second neuron's independent sum.
            double sum1 = layer.biases[row + 1];
            // Third neuron's independent sum.
            double sum2 = layer.biases[row + 2];
            // Fourth neuron's independent sum.
            double sum3 = layer.biases[row + 3];
            // Input coordinate shared by all four independent reductions.
            for (std::size_t column = 0; column < layer.inputs; ++column) {
                // Feature reused across the four output neurons.
                const double feature = (*incoming)[column];
                sum0 += first[column] * feature;
                sum1 += second[column] * feature;
                sum2 += third[column] * feature;
                sum3 += fourth[column] * feature;
            }
            output[row] = activate(sum0, layer.activation);
            output[row + 1] = activate(sum1, layer.activation);
            output[row + 2] = activate(sum2, layer.activation);
            output[row + 3] = activate(sum3, layer.activation);
        }
        // Remaining neurons use the same scalar reduction when width is not divisible by four.
        for (; row < layer.outputs; ++row) {
            // Contiguous weights for this output neuron.
            const double* weights = layer.weights.data() + row * layer.inputs;
            // Weighted sum plus bias, accumulated in input order.
            double sum = layer.biases[row];
            // Input feature coordinate.
            for (std::size_t column = 0; column < layer.inputs; ++column)
                sum += weights[column] * (*incoming)[column];
            output[row] = activate(sum, layer.activation);
        }
        incoming = &output;
    }
    return *incoming;
}
}
/* Infer inputs through layers, reusing per-layer scratch; returns independent output rows; throws on numerical failure. */
Dataset predictBatch(const std::vector<Dense>& layers, const Dataset& inputs) {
    // Per-layer outputs reused for every sample in this invocation.
    auto values = activationWorkspace(layers);
    // Caller-owned result with its row metadata allocated once.
    Dataset result;
    result.reserve(inputs.size());
    // Input sample whose final activation is copied into its result row.
    for (const auto& input : inputs) result.push_back(forwardInto(layers, input, values));
    return result;
}
/* Compute scalar MSE across inputs/targets for layers; validated dimensions required; throws on overflow. */
double meanLoss(const std::vector<Dense>& layers, const Dataset& inputs, const Dataset& targets) {
    // Largest residual magnitude seen so far, used to scale squares safely.
    double scale = 0;
    // Sum of squared residuals divided by scale squared.
    double scaledSquares = 0;
    // Per-layer activations reused for every sample without copying the input row.
    auto values = activationWorkspace(layers);
    // Dataset sample index.
    for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
        // Output features for this sample.
        const auto& output = forwardInto(layers, inputs[sample], values);
        // Output feature index.
        for (std::size_t feature = 0; feature < output.size(); ++feature) {
            // Absolute residual for scaled sum-of-squares accumulation.
            const double magnitude = std::abs(output[feature] - targets[sample][feature]);
            requireFinite(magnitude);
            if (magnitude > scale) {
                // Rescale old contributions to the larger residual magnitude.
                const double ratio = scale / magnitude;
                scaledSquares = 1 + scaledSquares * ratio * ratio;
                scale = magnitude;
            } else if (magnitude != 0) {
                // Contribution normalized by the current maximum magnitude.
                const double ratio = magnitude / scale;
                scaledSquares += ratio * ratio;
            }
        }
    }
    // Normalize before squaring so a finite MSE does not overflow prematurely.
    const double rootMean = scale * std::sqrt(scaledSquares / static_cast<double>(inputs.size()) /
        static_cast<double>(targets.front().size()));
    // Final mean squared error in target units squared.
    const double loss = rootMean * rootMean;
    requireFinite(loss);
    return loss;
}
/* Backpropagate batch MSE through layers for inputs/targets; returns flattened derivatives; throws on overflow. */
std::vector<double> batchGradient(const std::vector<Dense>& layers, const Dataset& inputs, const Dataset& targets) {
    // Flattened gradient offsets for each layer.
    std::vector<std::size_t> offsets{0};
    // Dense layer whose parameter count extends the offsets.
    for (const auto& layer : layers) offsets.push_back(offsets.back() + layer.weights.size() + layer.biases.size());
    // Summed, mean-normalized parameter gradient.
    std::vector<double> gradient(offsets.back(), 0);
    // Per-layer activation rows retained for differentiation and reused between samples.
    auto values = activationWorkspace(layers);
    // Maximum layer width bounds both reusable derivative rows.
    std::size_t width = 0;
    // Dense layer whose output width contributes to derivative scratch capacity.
    for (const auto& layer : layers) width = std::max(width, layer.outputs);
    // Derivatives entering the current layer, with capacity for every layer output.
    std::vector<double> upstream(width);
    // Derivatives leaving the current layer; swapped with upstream after each hidden layer.
    std::vector<double> previous(width);
    // Sample index in the batch.
    for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
        // Final output row references this sample's reusable activation storage.
        const auto& outputs = forwardInto(layers, inputs[sample], values);
        // Output feature index.
        for (std::size_t output = 0; output < outputs.size(); ++output)
            upstream[output] = 2 * (outputs[output] - targets[sample][output]) /
                static_cast<double>(inputs.size()) / static_cast<double>(outputs.size());
        // Reverse layer index; post-decrement avoids unsigned underflow in indexing.
        for (std::size_t index = layers.size(); index-- > 0;) {
            // Current dense transformation.
            const auto& layer = layers[index];
            // Input activations refer directly to the dataset for the first layer.
            const auto& incoming = index == 0 ? inputs[sample] : values[index - 1];
            // First layer has no trainable predecessor, so its input derivative is unnecessary.
            if (index != 0) std::fill_n(previous.begin(), layer.inputs, 0.0);
            // Output neuron index.
            for (std::size_t output = 0; output < layer.outputs; ++output) {
                // Derivative of loss with respect to this neuron's preactivation.
                const double delta = upstream[output] * derivative(values[index][output], layer.activation);
                gradient[offsets[index] + layer.weights.size() + output] += delta;
                // Contiguous parameter derivative row for this output neuron.
                double* rowGradient = gradient.data() + offsets[index] + output * layer.inputs;
                // Input feature index.
                for (std::size_t input = 0; input < layer.inputs; ++input)
                    rowGradient[input] += delta * incoming[input];
                if (index != 0) {
                    // Contiguous weights used to propagate to the previous trainable layer.
                    const double* weights = layer.weights.data() + output * layer.inputs;
                    // Previous-layer feature coordinate, accumulated in output order.
                    for (std::size_t input = 0; input < layer.inputs; ++input)
                        previous[input] += weights[input] * delta;
                }
            }
            upstream.swap(previous);
        }
    }
    // Gradient scalar being validated before returning.
    for (const double value : gradient) requireFinite(value);
    return gradient;
}
}
