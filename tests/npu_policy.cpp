#include <nn/Network.h>
#include "devices/Selection.h"
#include <stdexcept>

/* Require condition, described by message; throws runtime_error on a failed policy/API invariant. */
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

/* Check operation/precision/whole-graph policy with synthetic discovery only; no fake numerical execution. */
int main() {
    // Synthetic inventory with inference-only NPU and complete CPU, never used for numerical execution.
    const std::vector<nn::DeviceInfo> devices{
        {nn::DeviceType::NPU, 0, "Test NPU", "test-discovery-only", false, true},
        {nn::DeviceType::CPU, 0, "Test CPU", "test-discovery-only", true, true}};
    // Number of complete-model compatibility checks, not inference calls.
    std::size_t checks = 0;
    /* Reject NPU float64 compatibility and accept CPU; device is the candidate before any execution begins. */
    const auto fullPrecision = [&checks](const nn::DeviceInfo& device) -> std::optional<std::string> {
        ++checks;
        if (device.type == nn::DeviceType::NPU) return "NPU requires explicit reduced-precision opt-in";
        return std::nullopt;
    };
    // AUTO skips NPU only before execution and makes its reason available.
    const auto automatic = nn::detail::selectDevice({}, nn::Operation::Inference, devices, {}, fullPrecision);
    require(automatic.device.type == nn::DeviceType::CPU && !automatic.skipped.empty() && checks == 2, "NPU precision selection");
    // An explicit incompatible device must fail instead of silently selecting CPU.
    bool rejected = false;
    try { nn::detail::selectDevice({nn::DeviceType::NPU, 0}, nn::Operation::Inference, devices, {}, fullPrecision); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Explicit incompatible NPU fell back");
    checks = 0;
    nn::detail::selectDevice({}, nn::Operation::Training, devices, {}, fullPrecision);
    require(checks == 1, "Inference-only NPU considered for training");
    /* Simulate a successful compile-only suitability check; candidate identifies usable hardware, no inference performed. */
    const auto supported = [](const nn::DeviceInfo& /* candidate */) -> std::optional<std::string> { return std::nullopt; };
    require(nn::detail::selectDevice({}, nn::Operation::Inference, devices, {}, supported).device.type == nn::DeviceType::NPU, "NPU inference priority");
    // An actual network preserves full precision by default.
    nn::Network network(2, nn::DeviceType::CPU);
    network.addDense(1);
    require(!network.getNpuOptions().allowReducedPrecision, "Implicit precision demotion enabled");
    network.setNpuOptions({true});
    require(network.getNpuOptions().allowReducedPrecision, "NPU option not retained");
    // Canonical CPU weights do not change when inference permission changes.
    const auto weights = network.parameters();
    network.setNpuOptions({false});
    require(network.parameters() == weights, "NPU precision option changed canonical weights");
}
