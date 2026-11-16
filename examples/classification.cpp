#include <nn/Network.h>
#include <iomanip>
#include <iostream>

/* Learn whether a generated 2D point lies inside a circle using CPU SGD with MSE.
 * Inputs: none. Outputs: held-out classification accuracy and MSE; returns 0 when both checks pass,
 * 1 otherwise. Public API validation/backend exceptions propagate to the caller. */
int main() {
    // Two-feature training points on an evenly spaced square grid.
    nn::Dataset inputs;
    // Binary labels: 1 inside radius 0.65, 0 outside.
    nn::Dataset targets;
    // Horizontal coordinate index in the training grid.
    for (std::size_t column = 0; column <= 12; ++column) {
        // Vertical coordinate index in the training grid.
        for (std::size_t row = 0; row <= 12; ++row) {
            // Horizontal coordinate normalized to [-1, 1].
            const double x = -1.0 + static_cast<double>(column) / 6.0;
            // Vertical coordinate normalized to [-1, 1].
            const double y = -1.0 + static_cast<double>(row) / 6.0;
            inputs.push_back({x, y});
            targets.push_back({x * x + y * y < 0.65 * 0.65 ? 1.0 : 0.0});
        }
    }
    // Seeded CPU classifier with nonlinear features and a bounded binary output.
    nn::Network network(2, nn::DeviceType::CPU, 0, 19);
    network.addDense(12, nn::Activation::Tanh);
    network.addDense(1, nn::Activation::Sigmoid);
    network.setLoss(nn::Loss::MeanSquaredError);
    // Minibatch SGD with reproducible sample shuffling.
    nn::TrainingOptions options;
    options.epochs = 1600;
    options.batchSize = 16;
    options.optimizer.learningRate = 0.5;
    options.seed = 123;
    // Training summary reports actual execution and loss improvement.
    const auto trained = network.train(inputs, targets, options);
    // Grid-cell centers form a held-out dataset distinct from every training point.
    nn::Dataset heldOutInputs;
    // Exact circle-membership labels for held-out points.
    nn::Dataset heldOutTargets;
    // Horizontal coordinate index in the held-out grid.
    for (std::size_t column = 0; column < 12; ++column) {
        // Vertical coordinate index in the held-out grid.
        for (std::size_t row = 0; row < 12; ++row) {
            // Held-out horizontal midpoint.
            const double x = -1.0 + (static_cast<double>(column) + 0.5) / 6.0;
            // Held-out vertical midpoint.
            const double y = -1.0 + (static_cast<double>(row) + 0.5) / 6.0;
            heldOutInputs.push_back({x, y});
            heldOutTargets.push_back({x * x + y * y < 0.65 * 0.65 ? 1.0 : 0.0});
        }
    }
    // Inference result contains sigmoid scores for each held-out point.
    const auto predictions = network.predict(heldOutInputs);
    // Number of correct binary decisions at the application's 0.5 threshold.
    std::size_t correct = 0;
    // Shared index into held-out labels and predicted scores.
    for (std::size_t sample = 0; sample < heldOutInputs.size(); ++sample) {
        if ((predictions.values[sample][0] >= 0.5) == (heldOutTargets[sample][0] == 1.0)) {
            ++correct;
        }
    }
    // Fraction of held-out examples classified correctly.
    const double accuracy = static_cast<double>(correct) / static_cast<double>(heldOutInputs.size());
    // Mean squared error includes confidence as well as thresholded correctness.
    const double heldOutLoss = network.evaluate(heldOutInputs, heldOutTargets);
    std::cout << std::fixed << std::setprecision(6)
              << "Training MSE: " << trained.initialLoss << " -> " << trained.finalLoss
              << " on " << trained.selection.device.name << '\n'
              << "Held-out accuracy: " << correct << '/' << heldOutInputs.size()
              << " (" << 100.0 * accuracy << "%)\nHeld-out MSE: " << heldOutLoss << '\n';
    return trained.finalLoss < trained.initialLoss * 0.2 && accuracy >= 0.95 && heldOutLoss < 0.04 ? 0 : 1;
}
