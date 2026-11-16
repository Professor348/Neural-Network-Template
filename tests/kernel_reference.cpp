#include <nn/Network.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

// Historical serial kernel oracle compiled as host C++; never included in production or labeled as GPU execution.
namespace host_kernel {
using std::exp;
using std::tanh;
using std::fabs;
using std::sqrt;
using std::isfinite;
// Fixed-width index alias matches the OpenCL ulong ABI.
using index_type = std::uint64_t;
#define ulong index_type
#define __global
#define __kernel
#include "serial_kernel_reference.cl"
#undef __kernel
#undef __global
#undef ulong
}

/* Assert scalar/vector condition, with message identifying the invariant; throws runtime_error on mismatch. */
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

/* Verify historical serial reference algebra in a host-only test; does not validate current OpenCL runtime or GPU execution. */
int main() {
    // Test model input sample matrix.
    const nn::Dataset inputs{{0.2,-0.7}, {0.8,0.3}};
    // Test model two-output target matrix.
    const nn::Dataset targets{{0.1,0.9}, {-0.2,0.4}};
    // Flat input storage matching kernel ABI.
    const double flatInputs[]{0.2,-0.7,0.8,0.3};
    // Flat output target storage matching kernel ABI.
    const double flatTargets[]{0.1,0.9,-0.2,0.4};
    // Activation tested against CPU reference.
    for (const auto activation : {nn::Activation::Linear, nn::Activation::Sigmoid, nn::Activation::Tanh, nn::Activation::Relu}) {
        // Independent CPU reference model, never mislabeled as GPU.
        nn::Network reference(2, nn::DeviceType::CPU, 0, 42);
        reference.addDense(3, activation);
        reference.addDense(2, nn::Activation::Linear);
        // ABI metadata: widths, activation, parameter offset, incoming/outgoing offsets.
        const std::uint64_t architecture[]{2,3,static_cast<std::uint64_t>(activation),0,0,2, 3,2,0,9,2,5};
        // Mutable kernel parameter storage initialized identically to reference.
        auto parameters = reference.parameters();
        // Original vector for atomicity and SGD comparisons.
        const auto original = parameters;
        // Scratch: two activation arrays (7 each), two parameter arrays (17 each).
        std::vector<double> scratch(48);
        // Output storage sized for largest operation result.
        std::vector<double> output(17);
        // Kernel numerical failure flag.
        int error = 0;
        host_kernel::execute(architecture, parameters.data(), flatInputs, flatTargets, output.data(), scratch.data(), &error, 2,2,17,7,0,0);
        // Independent CPU inference result.
        const auto predicted = reference.predict(inputs).values;
        // Flattened output coordinate.
        for (std::size_t index = 0; index < 4; ++index)
            require(std::abs(output[index] - predicted[index / 2][index % 2]) < 1e-14, "Kernel forward mismatch");
        host_kernel::execute(architecture, parameters.data(), flatInputs, flatTargets, output.data(), scratch.data(), &error, 2,2,17,7,1,0);
        require(!error && std::abs(output[0] - reference.evaluate(inputs, targets)) < 1e-14, "Kernel loss mismatch");
        host_kernel::execute(architecture, parameters.data(), flatInputs, flatTargets, output.data(), scratch.data(), &error, 2,2,17,7,2,0);
        // CPU analytic derivatives independently checked by central-difference tests.
        const auto gradient = reference.gradients(inputs, targets);
        // Parameter coordinate.
        for (std::size_t index = 0; index < 17; ++index)
            require(std::abs(output[index] - gradient[index]) < 1e-14, "Kernel gradient mismatch");
        host_kernel::execute(architecture, parameters.data(), flatInputs, flatTargets, output.data(), scratch.data(), &error, 2,2,17,7,3,0.05);
        // Parameter coordinate after complete SGD step.
        for (std::size_t index = 0; index < 17; ++index)
            require(std::abs(parameters[index] - (original[index] - 0.05 * gradient[index])) < 1e-14, "Kernel SGD mismatch");
    }
    // Linear model metadata for numerical overflow/atomicity checks.
    const std::uint64_t linear[]{2,1,0,0,0,2};
    // Finite linear weights and bias before overflowing SGD proposal.
    double weights[]{1,1,0};
    // One two-feature sample.
    const double oneInput[]{1,1};
    // One regression target.
    const double oneTarget[]{0};
    // Linear model workspace (two activation arrays plus two parameter arrays).
    double workspace[12]{};
    // Output buffer for candidate parameters.
    double output[3]{};
    // Numerical status returned by kernel.
    int error = 0;
    host_kernel::execute(linear, weights, oneInput, oneTarget, output, workspace, &error, 1,1,3,3,3,std::numeric_limits<double>::max());
    require(error && weights[0] == 1 && weights[1] == 1 && weights[2] == 0, "Kernel update atomicity");
    std::cout << "Host-only historical serial kernel checks passed; this is not GPU execution\n";
}
