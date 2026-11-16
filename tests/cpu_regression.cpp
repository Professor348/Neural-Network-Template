#include <nn/Network.h>
#include "numerics/Functions.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
namespace {
/* Require condition; message identifies the failed numerical or atomicity invariant. */
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
/* Compute scalar reference outputs; layers is a validated model and input is one feature row. */
std::vector<double> scalarForward(const std::vector<nn::Dense>& layers, std::vector<double> input) {
    // Dense layer evaluated in the original scalar neuron/input order.
    for (const auto& layer : layers) {
        // Independent row for this layer's reference activations.
        std::vector<double> output(layer.outputs);
        // Output neuron coordinate.
        for (std::size_t row = 0; row < layer.outputs; ++row) {
            // Weighted sum accumulated in ascending feature order.
            double sum = layer.biases[row];
            // Incoming feature coordinate.
            for (std::size_t column = 0; column < layer.inputs; ++column)
                sum += layer.weights[row * layer.inputs + column] * input[column];
            output[row] = nn::detail::activate(sum, layer.activation);
        }
        input = std::move(output);
    }
    return input;
}
}
/* Exercise grouped neuron tails, changing scratch widths, repeated calls and atomic parameter validation. */
int main() {
    // Distinct samples expose stale activations and derivative values between samples.
    const nn::Dataset inputs{{0.2, -0.7, 0.4}, {0.8, 0.3, -0.2}, {-0.9, 0.5, 0.1}};
    // Two-output targets exercise mean reduction across samples and output features.
    const nn::Dataset targets{{0.1, 0.9}, {-0.2, 0.4}, {0.7, -0.3}};
    // Every nonlinearity is exercised with grouped and remainder neurons.
    for (const auto activation : {nn::Activation::Linear, nn::Activation::Tanh, nn::Activation::Sigmoid, nn::Activation::Relu}) {
        // Widths grow and shrink; no width is assumed to be a multiple of four.
        nn::Network network(3, nn::DeviceType::CPU, 0, 123);
        network.addDense(5, activation);
        network.addDense(7, activation);
        network.addDense(2, nn::Activation::Linear);
        // Original parameters used for finite differences and atomicity checks.
        const auto original = network.parameters();
        // Public architecture mirrored only for independent scalar feedforward evaluation.
        std::vector<nn::Dense> layers{{3, 5, activation, {}, {}}, {5, 7, activation, {}, {}},
            {7, 2, nn::Activation::Linear, {}, {}}};
        // Current flattened parameter coordinate in the mirrored model.
        std::size_t parameter = 0;
        // Mirrored layer receiving the network's exact weights and biases.
        for (auto& layer : layers) {
            layer.weights.assign(original.begin() + parameter, original.begin() + parameter + layer.inputs * layer.outputs);
            parameter += layer.weights.size();
            layer.biases.assign(original.begin() + parameter, original.begin() + parameter + layer.outputs);
            parameter += layer.biases.size();
        }
        // Batched predictions must remain independent after scratch is overwritten.
        const auto predicted = network.predict(inputs).values;
        // Sample coordinate for independent scalar comparison.
        for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
            // Scalar reference with no grouped sums or reused rows.
            const auto expected = scalarForward(layers, inputs[sample]);
            // Output coordinate for scalar comparison.
            for (std::size_t output = 0; output < expected.size(); ++output)
                require(std::abs(predicted[sample][output] - expected[output]) < 1e-14, "Grouped CPU forward mismatch");
        }
        // Analytic mean derivatives with scratch reused across all three samples.
        const auto gradient = network.gradients(inputs, targets);
        // Sum of independent single-sample gradients, normalized after summation.
        std::vector<double> independent(gradient.size(), 0);
        // Sample coordinate for gradient decomposition.
        for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
            // Derivatives computed with freshly allocated per-call scratch.
            const auto single = network.gradients({inputs[sample]}, {targets[sample]});
            // Parameter coordinate in the independent gradient sum.
            for (std::size_t index = 0; index < independent.size(); ++index) independent[index] += single[index] / inputs.size();
        }
        // Coordinate for analytic decomposition and central finite differences.
        for (std::size_t index = 0; index < original.size(); ++index) {
            require(std::abs(gradient[index] - independent[index]) < 1e-14, "CPU derivative scratch retained a prior sample");
            // Independent parameter perturbation in both directions.
            auto perturbed = original;
            perturbed[index] += 1e-6;
            network.setParameters(perturbed);
            // Loss at the upper finite-difference endpoint.
            const double above = network.evaluate(inputs, targets);
            perturbed[index] -= 2e-6;
            network.setParameters(perturbed);
            // Central finite-difference derivative for this parameter.
            const double numerical = (above - network.evaluate(inputs, targets)) / 2e-6;
            require(std::abs(numerical - gradient[index]) < 1e-7, "CPU changing-width gradient mismatch");
        }
        network.setParameters(original);
        // Invalid vector has the correct count but fails late scalar validation.
        auto invalid = original;
        invalid.back() = std::numeric_limits<double>::infinity();
        try {
            network.setParameters(invalid);
            throw std::runtime_error("Nonfinite parameters accepted");
        } catch (const std::invalid_argument&) {}
        require(network.parameters() == original, "Rejected parameters partially changed the model");
        invalid.pop_back();
        try {
            network.setParameters(invalid);
            throw std::runtime_error("Wrong parameter count accepted");
        } catch (const std::invalid_argument&) {}
        require(network.parameters() == original, "Wrong parameter count changed the model");
    }
    std::cout << "CPU grouped forward, scratch reuse, gradients and atomic parameters passed\n";
}
