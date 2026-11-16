#include <nn/Network.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

/* Require condition, with message identifying failure; throws runtime_error on mismatch. */
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
/* Compare scalar arrays left/right within tolerance; throws for dimension/nonfinite/numeric differences. */
void close(const std::vector<double>& left, const std::vector<double>& right, double tolerance) {
    require(left.size() == right.size(), "Parameter/output size mismatch");
    // Scalar coordinate being compared.
    for (std::size_t index = 0; index < left.size(); ++index)
        require(std::isfinite(left[index]) && std::isfinite(right[index]) && std::abs(left[index] - right[index]) < tolerance, "GPU numerical mismatch");
}
/* Check real GPU execution, finite differences, SGD and XOR; return 77 only when no compatible GPU is advertised. */
int main() {
    if (!nn::isDeviceAvailable({nn::DeviceType::GPU, 0})) {
        // Hardware CI can require execution so discovery/compiler failures cannot silently skip.
        const char* required = std::getenv("NN_REQUIRE_GPU_TESTS");
        try {
            // Explicit construction returns the actionable discovery/runtime diagnostic.
            nn::Network unavailable(1, nn::DeviceType::GPU);
        } catch (const std::runtime_error& error /* concrete device/runtime/build failure */) {
            std::cout << (required && std::string(required) == "1" ? "FAIL: " : "SKIP: ") << error.what() << '\n';
        }
        return required && std::string(required) == "1" ? 1 : 77;
    }
    // Nonlinear regression batch, two input features.
    const nn::Dataset inputs{{0.2,-0.7}, {0.8,0.3}};
    // Two output features per sample.
    const nn::Dataset targets{{0.1,0.9}, {-0.2,0.4}};
    // Activation tested with genuine GPU execution.
    for (const auto activation : {nn::Activation::Linear, nn::Activation::Tanh, nn::Activation::Sigmoid, nn::Activation::Relu}) {
        // CPU numerical reference.
        nn::Network cpu(2, nn::DeviceType::CPU, 0, 42);
        cpu.addDense(3, activation);
        cpu.addDense(2);
        // Identical model sent to the GPU.
        auto gpu = cpu;
        gpu.setDevice(nn::DeviceType::GPU, 0);
        // Original parameters for finite difference perturbations.
        const auto original = gpu.parameters();
        // Actual GPU gradient.
        const auto gradient = gpu.gradients(inputs, targets);
        close(gradient, cpu.gradients(inputs, targets), 1e-10);
        require(std::abs(gpu.evaluate(inputs, targets) - cpu.evaluate(inputs, targets)) < 1e-10, "GPU loss mismatch");
        // Perturbed parameter index.
        for (std::size_t index = 0; index < original.size(); ++index) {
            // Independent central difference perturbation.
            auto values = original;
            values[index] += 1e-6;
            gpu.setParameters(values);
            // Mean loss above parameter.
            const double above = gpu.evaluate(inputs, targets);
            values[index] -= 2e-6;
            gpu.setParameters(values);
            require(std::abs((above - gpu.evaluate(inputs, targets)) / 2e-6 - gradient[index]) < 1e-7, "GPU finite difference mismatch");
        }
        gpu.setParameters(original);
        // One batch update for CPU/GPU parameter comparison.
        nn::TrainingOptions options;
        options.epochs = 1;
        options.batchSize = 2;
        options.shuffle = false;
        options.optimizer.learningRate = 0.05;
        cpu.train(inputs, targets, options);
        // GPU result proves that training selected the requested device.
        const auto result = gpu.train(inputs, targets, options);
        require(result.selection.device.type == nn::DeviceType::GPU, "Training mislabeled CPU execution");
        close(cpu.parameters(), gpu.parameters(), 1e-10);
        // Per-call CPU prediction must see latest GPU-trained parameters.
        const auto overridden = gpu.predict(inputs, nn::DevicePreference{nn::DeviceType::CPU, 0});
        // CPU prediction reference after same update.
        const auto expected = cpu.predict(inputs);
        close(overridden.values[0], expected.values[0], 1e-10);
        require(gpu.getDevice().type == nn::DeviceType::GPU, "Override changed preference");
        gpu.setDevice(nn::DeviceType::CPU);
        gpu.train(inputs, targets, options);
        cpu.train(inputs, targets, options);
        gpu.setDevice(nn::DeviceType::GPU);
        close(gpu.predict(inputs).values[0], cpu.predict(inputs).values[0], 1e-10);
    }
    // Synthetic XOR truth table.
    const nn::Dataset xorInputs{{0,0},{0,1},{1,0},{1,1}};
    // Synthetic XOR labels.
    const nn::Dataset xorTargets{{0},{1},{1},{0}};
    // Explicit GPU model with reproducible initialization.
    nn::Network model(2, nn::DeviceType::GPU, 0, 7);
    model.addDense(4, nn::Activation::Tanh);
    model.addDense(1, nn::Activation::Sigmoid);
    // GPU minibatch SGD settings, including a final partial batch.
    nn::TrainingOptions options;
    options.epochs = 6000;
    options.batchSize = 3;
    options.optimizer.learningRate = 0.5;
    options.seed = 123;
    // Full training result computed on GPU.
    const auto result = model.train(xorInputs, xorTargets, options);
    require(result.finalLoss < 0.005 && result.finalLoss < result.initialLoss * 0.05, "GPU XOR learning failed");
    // Prediction selection must be GPU as well as training.
    const auto prediction = model.predict(xorInputs);
    require(prediction.selection.device.type == nn::DeviceType::GPU, "GPU inference selection");
    // XOR sample index.
    for (std::size_t index = 0; index < xorInputs.size(); ++index)
        require(std::abs(prediction.values[index][0] - xorTargets[index][0]) < 0.15, "GPU XOR prediction");
    // Large finite residual whose mean square must remain representable.
    nn::Network large(1, nn::DeviceType::GPU);
    large.addDense(1);
    large.setParameters({0, 0});
    // Mean loss from a GPU reduction, not host post-processing.
    const double largeLoss = large.evaluate({{0},{0},{0},{0}}, {{2e154},{0},{0},{0}});
    require(std::isfinite(largeLoss) && std::abs(largeLoss / 1e308 - 1) < 1e-14, "GPU large MSE");
    // Linear model that produces an overflowing candidate update from finite data.
    nn::Network overflow(2, nn::DeviceType::GPU);
    overflow.addDense(1);
    overflow.setParameters({1,1,0});
    // Canonical parameters before rejected batch.
    const auto stable = overflow.parameters();
    options.epochs = 1;
    options.optimizer.learningRate = std::numeric_limits<double>::max();
    // Whether GPU numerical error propagated to the application.
    bool failed = false;
    try { overflow.train({{1,1}}, {{0}}, options); }
    catch (const std::runtime_error&) { failed = true; }
    require(failed && overflow.parameters() == stable, "GPU batch update atomicity");
    std::cout << "GPU execution passed: " << result.selection.device.name << ", loss " << result.initialLoss << " -> " << result.finalLoss << '\n';
}
