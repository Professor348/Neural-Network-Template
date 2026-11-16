#include <nn/Network.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

/* Require condition; condition is the invariant, message identifies failure; throws on failure. */
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

/* Check central differences for every parameter and one independent SGD update; throws on mismatch. */
int main() {
    // Inputs have two samples with two features; targets have two outputs per sample.
    const nn::Dataset inputs{{0.2, -0.7}, {0.8, 0.3}};
    // Regression targets exercise multi-output mean reduction.
    const nn::Dataset targets{{0.1, 0.9}, {-0.2, 0.4}};
    // Each activation is tested away from ReLU's nondifferentiable origin.
    for (const auto activation : {nn::Activation::Tanh, nn::Activation::Sigmoid, nn::Activation::Linear, nn::Activation::Relu}) {
        // Deterministically initialized network under test.
        nn::Network network(2, nn::DeviceType::CPU, 0, 42);
        network.addDense(3, activation);
        network.addDense(2, nn::Activation::Linear);
        // Original flattened weights and biases.
        const auto original = network.parameters();
        // Analytic derivative of mean loss with respect to flattened parameters.
        const auto gradient = network.gradients(inputs, targets);
        // Central difference step in parameter units.
        const double epsilon = 1e-6;
        // Parameter index across all layers.
        for (std::size_t index = 0; index < original.size(); ++index) {
            // Independent perturbation of the original parameter vector.
            auto perturbed = original;
            perturbed[index] += epsilon;
            network.setParameters(perturbed);
            // Mean loss above the parameter.
            const double above = network.evaluate(inputs, targets);
            perturbed[index] -= 2 * epsilon;
            network.setParameters(perturbed);
            // Central finite difference derivative.
            const double numerical = (above - network.evaluate(inputs, targets)) / (2 * epsilon);
            require(std::abs(numerical - gradient[index]) < 1e-7, "gradient mismatch");
        }
        network.setParameters(original);
        // A single full batch SGD step with no shuffling.
        nn::TrainingOptions options;
        options.epochs = 1;
        options.batchSize = 2;
        options.shuffle = false;
        options.optimizer.learningRate = 0.05;
        network.train(inputs, targets, options);
        // Parameters after the independently checked update.
        const auto updated = network.parameters();
        // Parameter index for update comparison.
        for (std::size_t index = 0; index < original.size(); ++index) {
            require(std::abs(updated[index] - (original[index] - 0.05 * gradient[index])) < 1e-14, "SGD mismatch");
        }
    }
    // Large finite residuals whose mean square is still representable.
    nn::Network large(1, nn::DeviceType::CPU);
    large.addDense(1);
    large.setParameters({0, 0});
    // Correct MSE is 1e308, despite an individual squared residual exceeding DBL_MAX.
    const double largeLoss = large.evaluate({{0},{0},{0},{0}}, {{2e154},{0},{0},{0}});
    require(std::isfinite(largeLoss) && std::abs(largeLoss / 1e308 - 1) < 1e-14, "Stable MSE reduction");
    std::cout << "Gradient and update checks passed\n";
}
