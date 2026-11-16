#include <nn/Network.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

// Exact production kernel compiled as host C++ solely to check indexing/algebra, not GPU execution.
namespace parallel {
using std::exp;
using std::tanh;
using std::fabs;
using std::sqrt;
using std::isfinite;
// Portable OpenCL ulong equivalent.
using index_type = std::uint64_t;
// Simulated global work-item coordinate for deterministic host-only tests.
static std::size_t workItem = 0;
/* Return simulated work-item index; dimension is unused because all launches are one-dimensional. */
std::size_t get_global_id(unsigned /* dimension */) { return workItem; }
/* Host-only atomic-or stand-in; address is error flag, value is failure bit. Serial test execution needs no atomics. */
int atomic_or(volatile int* address, int value) { const int previous /* prior failure bits */ = *address; *address |= value; return previous; }
#define ulong index_type
#define __global
#define __kernel
#include "../src/backends/opencl/kernels.cl"
#undef __kernel
#undef __global
#undef ulong
}
/* Require condition, described by message; throw runtime_error for failed numerical invariant. */
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
/* Compare left/right vectors with tolerance; throw on dimensions or nonfinite/numerical mismatch. */
void close(const std::vector<double>& left, const std::vector<double>& right, double tolerance) {
    require(left.size() == right.size(), "Parallel output dimension");
    // Output coordinate under comparison.
    for (std::size_t index = 0; index < left.size(); ++index)
        require(std::isfinite(left[index]) && std::isfinite(right[index]) && std::abs(left[index] - right[index]) < tolerance, "Parallel numerical mismatch");
}
/* Exercise full production parallel pipeline over uneven shapes and batches, entirely on host for algebra validation. */
int main() {
    // Batch sizes include single samples and non-workgroup multiples.
    for (const std::size_t samples : {1u, 3u, 17u, 257u}) {
        // Nonlinearity exercises each local derivative.
        for (const auto activation : {nn::Activation::Linear, nn::Activation::Sigmoid, nn::Activation::Tanh, nn::Activation::Relu}) {
            // CPU oracle independent of parallel indexing.
            nn::Network cpu(5, nn::DeviceType::CPU, 0, 42);
            cpu.addDense(7, activation);
            cpu.addDense(3, nn::Activation::Sigmoid);
            // Synthetic sample matrix.
            nn::Dataset inputs(samples, std::vector<double>(5));
            // Synthetic labels.
            nn::Dataset targets(samples, std::vector<double>(3));
            // Sample-major flattened inputs, then hidden activations, then outputs.
            std::vector<double> values(samples * 15);
            // Sample-major labels.
            std::vector<double> labels(samples * 3);
            // Sample index for deterministic data generation.
            for (std::size_t sample = 0; sample < samples; ++sample) {
                // Input feature index.
                for (std::size_t input = 0; input < 5; ++input)
                    values[sample * 5 + input] = inputs[sample][input] = std::sin(static_cast<double>(sample * 5 + input));
                // Label feature index.
                for (std::size_t output = 0; output < 3; ++output)
                    labels[sample * 3 + output] = targets[sample][output] = 0.1 * output;
            }
            // Model parameters in canonical order (42 first-layer, 24 second-layer).
            auto weights = cpu.parameters();
            // Saved original parameters for independent SGD comparison.
            const auto original = weights;
            // Loss derivatives at each activation boundary.
            std::vector<double> deltas(samples * 15);
            // Mean parameter gradients.
            std::vector<double> gradient(66);
            // Candidate SGD update.
            std::vector<double> candidate(66);
            // Numerical status, shared by all simulated work items.
            int error = 0;
            // Forward hidden neuron work item.
            for (parallel::workItem = 0; parallel::workItem < samples * 7; ++parallel::workItem)
                parallel::forwardLayer(weights.data(), values.data(), &error, samples, 5, 7, 0, 0, samples * 5, static_cast<std::uint64_t>(activation));
            // Forward output neuron work item.
            for (parallel::workItem = 0; parallel::workItem < samples * 3; ++parallel::workItem)
                parallel::forwardLayer(weights.data(), values.data(), &error, samples, 7, 3, 42, samples * 5, samples * 12, 1);
            // Independent CPU predictions.
            const auto predicted = cpu.predict(inputs).values;
            // Sample index for forward output comparison.
            for (std::size_t sample = 0; sample < samples; ++sample)
                close({values.begin() + samples * 12 + sample * 3, values.begin() + samples * 12 + (sample + 1) * 3}, predicted[sample], 1e-13);
            // Output derivative work item.
            for (parallel::workItem = 0; parallel::workItem < samples * 3; ++parallel::workItem)
                parallel::outputDelta(values.data(), labels.data(), deltas.data(), &error, samples, 3, samples * 12, 1);
            // Hidden derivative work item.
            for (parallel::workItem = 0; parallel::workItem < samples * 7; ++parallel::workItem)
                parallel::hiddenDelta(weights.data(), values.data(), deltas.data(), &error, samples, 7, 3, 42, samples * 5, samples * 12, static_cast<std::uint64_t>(activation));
            // First-layer parameter work item.
            for (parallel::workItem = 0; parallel::workItem < 42; ++parallel::workItem)
                parallel::parameterGradient(values.data(), deltas.data(), gradient.data(), &error, samples, 5, 7, 0, 0, samples * 5);
            // Second-layer parameter work item.
            for (parallel::workItem = 0; parallel::workItem < 24; ++parallel::workItem)
                parallel::parameterGradient(values.data(), deltas.data(), gradient.data(), &error, samples, 7, 3, 42, samples * 5, samples * 12);
            close(gradient, cpu.gradients(inputs, targets), 1e-12);
            // Independent scalar loss blocks, up to 256 outputs each.
            const std::size_t blocks = (samples * 3 + 255) / 256;
            // Scaled partial losses, pairs of magnitude scale and normalized sum of squares.
            std::vector<double> partial(blocks * 2);
            // Final scalar loss result.
            double loss = 0;
            // Loss block work item.
            for (parallel::workItem = 0; parallel::workItem < blocks; ++parallel::workItem)
                parallel::lossBlocks(values.data(), labels.data(), partial.data(), &error, samples * 3, samples * 12);
            parallel::finishLoss(partial.data(), &loss, &error, blocks, samples * 3);
            require(!error && std::abs(loss - cpu.evaluate(inputs, targets)) < 1e-13, "Parallel loss mismatch");
            // Parameter candidate work item.
            for (parallel::workItem = 0; parallel::workItem < 66; ++parallel::workItem)
                parallel::stageUpdate(weights.data(), gradient.data(), candidate.data(), &error, 66, 0.05);
            // Separate kernel boundary guarantees complete validation before any parameter is committed.
            for (parallel::workItem = 0; parallel::workItem < 66; ++parallel::workItem)
                parallel::commitUpdate(weights.data(), candidate.data(), &error, 66);
            // Independent SGD expected vector.
            auto expected = original;
            // Parameter coordinate for independent expectation.
            for (std::size_t index = 0; index < 66; ++index) expected[index] -= 0.05 * gradient[index];
            close(weights, expected, 1e-14);
            // Force overflow in one candidate while other candidates remain valid.
            gradient.back() = std::numeric_limits<double>::max();
            // Parameter candidate work item after deliberately overflowing proposal.
            for (parallel::workItem = 0; parallel::workItem < 66; ++parallel::workItem)
                parallel::stageUpdate(weights.data(), gradient.data(), candidate.data(), &error, 66, 100);
            // No work item may commit when any candidate failed validation.
            for (parallel::workItem = 0; parallel::workItem < 66; ++parallel::workItem)
                parallel::commitUpdate(weights.data(), candidate.data(), &error, 66);
            require(error != 0, "Parallel overflow not detected");
            close(weights, expected, 1e-14);
        }
    }
    // Large residual spans two loss blocks, with a mean square that remains finite.
    std::vector<double> largeValues(257, 0);
    // Labels align with one output per sample.
    std::vector<double> largeTargets(257, 0);
    largeTargets[0] = 2e154;
    // Two independent scaled partials.
    double partials[4]{};
    // Final stable mean loss.
    double largeLoss = 0;
    // Independent numerical failure status.
    int numericalError = 0;
    // Parallel loss block coordinate.
    for (parallel::workItem = 0; parallel::workItem < 2; ++parallel::workItem)
        parallel::lossBlocks(largeValues.data(), largeTargets.data(), partials, &numericalError, 257, 0);
    parallel::finishLoss(partials, &largeLoss, &numericalError, 2, 257);
    require(!numericalError && std::isfinite(largeLoss) && std::abs(largeLoss / (4 * (1e308 / 257)) - 1) < 1e-13, "Parallel scaled large loss");
    std::cout << "Host-only parallel kernel checks passed (not GPU execution)\n";
}
