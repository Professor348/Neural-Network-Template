#include <nn/Network.h>
#include <iostream>

/* Demonstrate application-owned construction, training and inference. No command-line arguments or library logging. */
int main() {
    // Synthetic truth table, samples by input features.
    const nn::Dataset inputs{{0,0}, {0,1}, {1,0}, {1,1}};
    // XOR labels, one output per sample.
    const nn::Dataset targets{{0}, {1}, {1}, {0}};
    // Seeded architecture with automatic per-operation selection.
    nn::Network network(2, nn::DeviceType::AUTO, std::nullopt, 7);
    network.addDense(4, nn::Activation::Tanh);
    network.addDense(1, nn::Activation::Sigmoid);
    network.setLoss(nn::Loss::MeanSquaredError);
    // Settings belong to the application, not hardcoded library defaults.
    nn::TrainingOptions options;
    options.epochs = 6000;
    options.batchSize = 4;
    options.optimizer.learningRate = 0.5;
    options.seed = 123;
    // Returned summary lets the application decide how to present results.
    const auto trained = network.train(inputs, targets, options);
    std::cout << "Loss: " << trained.initialLoss << " -> " << trained.finalLoss
              << " on " << trained.selection.device.name << '\n';
    // Caller may inspect why higher-priority devices were skipped.
    for (const auto& reason : trained.selection.skipped) std::cout << reason << '\n';
    network.setDevice(nn::DeviceType::CPU, 0);
    // AUTO applies to only this prediction; configured preference remains CPU 0.
    const auto prediction = network.predict(inputs, nn::DevicePreference{nn::DeviceType::AUTO, std::nullopt});
    // Sample index for printing the truth table.
    for (std::size_t sample = 0; sample < inputs.size(); ++sample)
        std::cout << inputs[sample][0] << " XOR " << inputs[sample][1] << " = " << prediction.values[sample][0] << '\n';
    return trained.finalLoss < 0.005 ? 0 : 1;
}
