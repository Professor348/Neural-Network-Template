#include <nn/Network.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

/* Train XOR twice; return zero on reproducible convergence, throw on failure. */
int main() {
    // Four binary inputs generated directly in source.
    const nn::Dataset inputs{{0,0}, {0,1}, {1,0}, {1,1}};
    // XOR labels in matching sample order.
    const nn::Dataset targets{{0}, {1}, {1}, {0}};
    // First seeded model.
    nn::Network first(2, nn::DeviceType::CPU, 0, 7);
    first.addDense(4, nn::Activation::Tanh);
    first.addDense(1, nn::Activation::Sigmoid);
    // Identically seeded model used to check repeatability.
    nn::Network second(2, nn::DeviceType::CPU, 0, 7);
    second.addDense(4, nn::Activation::Tanh);
    second.addDense(1, nn::Activation::Sigmoid);
    // Caller-configured training with minibatches and a deterministic shuffle seed.
    nn::TrainingOptions options;
    options.epochs = 6000;
    options.batchSize = 3;
    options.optimizer.learningRate = 0.5;
    options.seed = 123;
    // Progress callback invocation count.
    std::size_t callbacks = 0;
    // Callback owns presentation policy; progress is the completed epoch report.
    const auto callback = [&callbacks](const nn::Progress& progress) {
        if (!std::isfinite(progress.loss)) throw std::runtime_error("nonfinite progress");
        ++callbacks;
    };
    // Training summary includes initial and final loss and actual selection.
    const auto result = first.train(inputs, targets, options, callback);
    second.train(inputs, targets, options);
    if (result.finalLoss > 0.005 || result.finalLoss >= result.initialLoss * 0.05)
        throw std::runtime_error("XOR did not learn");
    if (callbacks != options.epochs || first.parameters() != second.parameters())
        throw std::runtime_error("reproducibility/callback failure");
    // Predictions for the truth table.
    const auto prediction = first.predict(inputs);
    // Sample index in the XOR truth table.
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (std::abs(prediction.values[index][0] - targets[index][0]) > 0.15)
            throw std::runtime_error("XOR classification failure");
    }
    std::cout << "XOR loss " << result.initialLoss << " -> " << result.finalLoss << '\n';
}
