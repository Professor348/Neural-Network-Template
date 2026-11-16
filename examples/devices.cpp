#include <nn/Network.h>
#include <iostream>

/* Inspect available execution capabilities, then demonstrate AUTO and a call-only CPU override.
 * Inputs: none. Outputs: device inventory and selections; returns 0 when CPU override preserves AUTO,
 * 1 otherwise. Public API validation/backend exceptions propagate to the caller. */
int main() {
    // Executable devices reported by the library, including operation capabilities.
    const auto devices = nn::getAvailableDevices();
    std::cout << std::boolalpha << "Available devices:\n";
    // One discovered device, whose index is local to its processor category.
    for (const auto& device : devices) {
        std::cout << "  " << device.name << " (index " << device.index << ", " << device.backend
                  << "): training=" << device.training << ", inference=" << device.inference << '\n';
    }
    // Small seeded model with an AUTO preference resolved separately for each operation.
    nn::Network network(2, nn::DeviceType::AUTO, std::nullopt, 7);
    network.addDense(3, nn::Activation::Tanh);
    network.addDense(1, nn::Activation::Sigmoid);
    // Two generated input rows; this example demonstrates routing rather than learning.
    const nn::Dataset inputs{{0.25, -0.5}, {-0.75, 0.5}};
    // Configured preference before either operation, independent of the selected execution device.
    const auto preferenceBefore = network.getDevice();
    std::cout << "Configured preference: " << network.getDeviceName() << '\n';
    // Actual device selected automatically for this complete inference operation.
    const auto automatic = network.predict(inputs);
    std::cout << "AUTO inference selected: " << automatic.selection.device.name
              << " (" << automatic.selection.device.backend << ")\n";
    // CPU applies to this inference call without altering the configured AUTO preference.
    const auto onCpu = network.predict(inputs, nn::DevicePreference{nn::DeviceType::CPU, 0});
    // Configured preference read again after the per-call override.
    const auto preferenceAfter = network.getDevice();
    std::cout << "Per-call CPU inference selected: " << onCpu.selection.device.name
              << "\nConfigured preference afterward: " << network.getDeviceName() << '\n';
    return onCpu.selection.device.type == nn::DeviceType::CPU
        && preferenceBefore.type == nn::DeviceType::AUTO
        && preferenceAfter.type == preferenceBefore.type
        && preferenceAfter.index == preferenceBefore.index ? 0 : 1;
}
