#include <nn/Network.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
namespace {
// Observable checksum prevents timed results from becoming dead work.
volatile double checksum = 0;
/* Report median milliseconds for label; operation runs repeats times after two untimed warmups. */
template<class Operation>
void measure(const char* label, std::size_t repeats, Operation operation) {
    operation();
    operation();
    // Independent elapsed samples for a median resistant to scheduling interruptions.
    std::vector<double> milliseconds;
    milliseconds.reserve(repeats);
    // Timed repetition index.
    for (std::size_t repetition = 0; repetition < repeats; ++repetition) {
        // Monotonic start of this operation.
        const auto start = std::chrono::steady_clock::now();
        operation();
        // Elapsed milliseconds for the completed operation.
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        milliseconds.push_back(elapsed);
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    std::cout << label << ": " << milliseconds[milliseconds.size() / 2] << " ms\n";
}
}
/* Benchmark deterministic real GPU dense work; skip with code 77 when no executable GPU is available. */
int main() {
    if (!nn::isDeviceAvailable({nn::DeviceType::GPU, 0})) {
        std::cout << "SKIP: no executable GPU; GPU performance is unmeasured\n";
        return 77;
    }
    // Deterministic synthetic medium network with 53,600 parameters.
    nn::Network network(128, nn::DeviceType::GPU, 0, 42);
    network.addDense(192, nn::Activation::Tanh);
    network.addDense(128, nn::Activation::Relu);
    network.addDense(32, nn::Activation::Linear);
    // Synthetic feature matrix with moderate finite magnitudes.
    nn::Dataset inputs(128, std::vector<double>(128));
    // Synthetic target matrix aligned with the 32 output features.
    nn::Dataset targets(128, std::vector<double>(32));
    // Sample coordinate in each deterministic dataset.
    for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
        // Input feature coordinate.
        for (std::size_t feature = 0; feature < inputs[sample].size(); ++feature)
            inputs[sample][feature] = std::sin(static_cast<double>(sample * 131 + feature) * 0.01);
        // Target feature coordinate.
        for (std::size_t feature = 0; feature < targets[sample].size(); ++feature)
            targets[sample][feature] = std::cos(static_cast<double>(sample * 37 + feature) * 0.02);
    }
    // Original parameters restored before every training measurement.
    const auto original = network.parameters();
    // Fixed SGD workload with two complete passes through the dataset.
    nn::TrainingOptions options;
    options.epochs = 2;
    options.batchSize = 32;
    options.shuffle = false;
    options.optimizer.learningRate = 0.001;
    std::cout << std::fixed << std::setprecision(3);
    measure("predict 128 samples", 15,
        /* Infer captured inputs on network and consume one result; no arguments or return value. */
        [&]() { checksum += network.predict(inputs).values.back().front(); });
    measure("evaluate 128 samples", 15,
        /* Evaluate captured inputs/targets on network and consume loss; no arguments or return value. */
        [&]() { checksum += network.evaluate(inputs, targets); });
    measure("gradient 128 samples", 15,
        /* Differentiate captured inputs/targets and consume one derivative; no arguments or return value. */
        [&]() { checksum += network.gradients(inputs, targets).front(); });
    measure("train 2 epochs, batch 32", 9,
        /* Restore original network, train with captured options/data, and consume loss; no arguments or return value. */
        [&]() {
            network.setParameters(original);
            checksum += network.train(inputs, targets, options).finalLoss;
        });
    measure("setParameters 1000 calls", 15,
        /* Replace network parameters repeatedly with captured original and consume one scalar; no arguments or return value. */
        [&]() {
            // Repetition coordinate amortizes clock overhead for parameter assignment.
            for (std::size_t call = 0; call < 1000; ++call) network.setParameters(original);
            checksum += network.parameters().front();
        });
    std::cout << "checksum: " << checksum << '\n';
}
