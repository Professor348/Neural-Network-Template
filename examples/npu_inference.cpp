#include <nn/Network.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

/* Train synthetic XOR on CPU, explicitly opt into reduced precision and compare real NPU inference.
 * Inputs: none. Outputs: CPU training loss and CPU/NPU predictions; returns 0 on agreement,
 * 77 when NPU inference is unavailable, 1 on failed checks. Backend exceptions propagate. */
int main() {
    // Synthetic binary truth table, generated directly in source.
    const nn::Dataset inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    // XOR labels aligned with the input rows.
    const nn::Dataset targets{{0}, {1}, {1}, {0}};
    // CPU owns training; selecting NPU later never changes this model preference.
    nn::Network network(2, nn::DeviceType::CPU, 0, 7);
    network.addDense(4, nn::Activation::Tanh);
    network.addDense(1, nn::Activation::Sigmoid);
    network.setLoss(nn::Loss::MeanSquaredError);
    // Reproducible CPU SGD settings for the four-row dataset.
    nn::TrainingOptions options;
    options.epochs = 6000;
    options.batchSize = inputs.size();
    options.optimizer.learningRate = 0.5;
    options.seed = 123;
    // CPU training result verifies learning before any accelerator comparison.
    const auto trained = network.train(inputs, targets, options);
    std::cout << std::fixed << std::setprecision(6)
              << "CPU training MSE: " << trained.initialLoss << " -> " << trained.finalLoss
              << " on " << trained.selection.device.name << '\n';
    if (trained.selection.device.type != nn::DeviceType::CPU || trained.finalLoss >= 0.005) {
        std::cerr << "CPU XOR training did not meet the accuracy requirement.\n";
        return 1;
    }
    if (!nn::isDeviceAvailable({nn::DeviceType::NPU, 0}, nn::Operation::Inference)) {
        std::cout << "Skipped: no executable NPU inference device is available. "
                  << "An NPU-enabled build and supported NPU hardware are required.\n";
        return 77;
    }
    // Full-precision CPU reference from the trained, host-owned parameters.
    const auto cpu = network.predict(inputs);
    // Explicit permission allows reduced-precision arithmetic only for NPU inference.
    network.setNpuOptions({true});
    // An explicit NPU request either executes on NPU or reports an error; it cannot silently use CPU.
    const auto npu = network.predict(inputs, nn::DevicePreference{nn::DeviceType::NPU, 0});
    if (npu.selection.device.type != nn::DeviceType::NPU) {
        std::cerr << "The explicit NPU call did not report NPU execution.\n";
        return 1;
    }
    // Largest absolute difference between full-precision CPU and reduced-precision NPU outputs.
    double maximumDifference = 0.0;
    // True only when both devices classify every XOR row correctly.
    bool labelsMatch = true;
    std::cout << "NPU inference executed on " << npu.selection.device.name << '\n';
    // Shared index into inputs, labels and both prediction results.
    for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
        maximumDifference = std::max(maximumDifference, std::abs(cpu.values[sample][0] - npu.values[sample][0]));
        labelsMatch = labelsMatch && (cpu.values[sample][0] >= 0.5) == (targets[sample][0] == 1.0)
            && (npu.values[sample][0] >= 0.5) == (targets[sample][0] == 1.0);
        std::cout << inputs[sample][0] << " XOR " << inputs[sample][1]
                  << ": CPU=" << cpu.values[sample][0] << ", NPU=" << npu.values[sample][0] << '\n';
    }
    std::cout << "Maximum absolute CPU/NPU difference: " << maximumDifference
              << " (allowed: 0.05 for reduced precision)\n";
    return labelsMatch && maximumDifference <= 0.05 ? 0 : 1;
}
