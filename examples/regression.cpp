#include <nn/Network.h>
#include <iomanip>
#include <iostream>

/* Learn y = x*x from generated samples using CPU SGD and predict unseen inputs.
 * Inputs: none. Outputs: application-selected progress and held-out MSE; returns 0 on convergence,
 * 1 on insufficient accuracy. Public API validation/backend exceptions propagate to the caller. */
int main() {
    // Evenly spaced scalar training inputs in [-1, 1].
    nn::Dataset inputs;
    // Matching nonlinear regression targets.
    nn::Dataset targets;
    // Index of the generated training sample, including both endpoints.
    for (std::size_t sample = 0; sample <= 40; ++sample) {
        // Scalar input coordinate for this sample.
        const double x = -1.0 + 2.0 * static_cast<double>(sample) / 40.0;
        inputs.push_back({x});
        targets.push_back({x * x});
    }
    // Seeded CPU network: tanh learns curvature and linear output represents a real value.
    nn::Network network(1, nn::DeviceType::CPU, 0, 17);
    network.addDense(12, nn::Activation::Tanh);
    network.addDense(1, nn::Activation::Linear);
    network.setLoss(nn::Loss::MeanSquaredError);
    // Full-batch SGD settings, independent of the initialization seed.
    nn::TrainingOptions options;
    options.epochs = 2500;
    options.batchSize = inputs.size();
    options.optimizer.learningRate = 0.1;
    options.seed = 123;
    std::cout << std::fixed << std::setprecision(6);
    /* Print occasional completed epochs according to application policy.
     * Input: progress contains epoch and MSE. Output: a console line every 500 epochs; no intentional exceptions. */
    const auto logProgress = [](const nn::Progress& progress /* Completed epoch and its full training MSE. */) {
        if (progress.epoch % 500 == 0) {
            std::cout << "Epoch " << progress.epoch << ": MSE " << progress.loss << '\n';
        }
    };
    // Training summary records loss improvement and the actual CPU execution device.
    const auto trained = network.train(inputs, targets, options, logProgress);
    // Midpoints absent from the generated training grid.
    const nn::Dataset heldOutInputs{{-0.925}, {-0.675}, {-0.325}, {0.025}, {0.375}, {0.725}, {0.925}};
    // Exact mathematical targets for the held-out inputs.
    nn::Dataset heldOutTargets;
    // One held-out input row, containing its single feature.
    for (const auto& row : heldOutInputs) {
        heldOutTargets.push_back({row[0] * row[0]});
    }
    // Predictions use the network's configured CPU preference.
    const auto predictions = network.predict(heldOutInputs);
    // Public loss evaluation on data that did not participate in updates.
    const double heldOutLoss = network.evaluate(heldOutInputs, heldOutTargets);
    std::cout << "Training MSE: " << trained.initialLoss << " -> " << trained.finalLoss
              << " on " << trained.selection.device.name << '\n';
    // Index shared by input, expected target and predicted output rows.
    for (std::size_t sample = 0; sample < heldOutInputs.size(); ++sample) {
        std::cout << "x=" << heldOutInputs[sample][0] << " expected=" << heldOutTargets[sample][0]
                  << " predicted=" << predictions.values[sample][0] << '\n';
    }
    std::cout << "Held-out MSE: " << heldOutLoss << '\n';
    return trained.finalLoss < trained.initialLoss * 0.02 && heldOutLoss < 0.002 ? 0 : 1;
}
