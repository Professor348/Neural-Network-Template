#include <nn/Network.h>
#include "devices/Selection.h"
#include <stdexcept>
#include <limits>

/* Assert condition, labeled by message; throws runtime_error on policy failure. */
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
/* Expect runtime device error from action, a device request; throw if no error or diagnostic is empty. */
void unavailable(const std::function<void()>& action) {
    try { action(); } catch (const std::runtime_error& error /* backend diagnostic */) {
        check(std::string(error.what()).size() > 10, "Missing diagnostic");
        return;
    }
    throw std::runtime_error("Unavailable device accepted");
}
/* Verify policy on synthetic inventories separately from real CPU execution and override behavior. */
int main() {
    // Synthetic discovery only: these descriptors are never passed to numerical execution.
    const std::vector<nn::DeviceInfo> inventory{
        {nn::DeviceType::NPU, 0, "NPU0", "test-discovery", false, true},
        {nn::DeviceType::GPU, 0, "GPU0", "test-discovery", false, true},
        {nn::DeviceType::GPU, 1, "GPU1", "test-discovery", true, true},
        {nn::DeviceType::CPU, 0, "CPU0", "test-discovery", true, true}};
    // Automatic training resolution skips inference-only devices.
    const auto training = nn::detail::selectDevice({}, nn::Operation::Training, inventory, {});
    check(training.device.index == 1 && training.device.type == nn::DeviceType::GPU && training.skipped.size() == 2, "Training selection");
    check(nn::detail::selectDevice({}, nn::Operation::Inference, inventory, {}).device.type == nn::DeviceType::NPU, "Inference priority");
    /* Available inference-only NPU must reject explicit training without fallback. */
    unavailable([&inventory] { nn::detail::selectDevice({nn::DeviceType::NPU, 0}, nn::Operation::Training, inventory, {}); });
    /* Empty inventories must fail automatic selection. */
    unavailable([] { nn::detail::selectDevice({}, nn::Operation::Inference, {}, {}); });
    // Actual executable model, using real backend discovery only.
    nn::Network network(2);
    network.addDense(1);
    check(network.getDeviceName() == "Auto", "Configured name");
    check(nn::isDeviceAvailable({nn::DeviceType::CPU, 0}, nn::Operation::Training), "CPU availability");
    check(!nn::isDeviceAvailable({nn::DeviceType::CPU, 1}), "Invalid CPU index");
    check(!nn::isDeviceAvailable({nn::DeviceType::AUTO, 0}), "AUTO index accepted");
    /* Explicit missing GPU must fail at construction. */
    unavailable(
        /* Request an unavailable GPU through the constructor. */
        [] {
            // Model whose explicit unsupported device must reject construction.
            nn::Network model(2, nn::DeviceType::GPU, std::numeric_limits<std::size_t>::max());
        });
    /* Explicit NPU setter must fail without changing preference. */
    unavailable([&network] { network.setDevice(nn::DeviceType::NPU, std::numeric_limits<std::size_t>::max()); });
    /* Per-call GPU request must fail rather than execute on CPU. */
    unavailable([&network] { network.predict({{0,0}}, nn::DevicePreference{nn::DeviceType::GPU, std::numeric_limits<std::size_t>::max()}); });
    // Inference override chooses CPU without mutating AUTO configuration.
    const auto output = network.predict({{0,0}}, nn::DevicePreference{nn::DeviceType::CPU, 0});
    check(output.selection.device.type == nn::DeviceType::CPU && network.getDevice().type == nn::DeviceType::AUTO, "Inference override persisted");
    // One training epoch using an explicit CPU call-only override.
    nn::TrainingOptions options;
    options.epochs = 1;
    network.train({{0,0}}, {{1}}, options, {}, nn::DevicePreference{nn::DeviceType::CPU, 0});
    check(network.getDevice().type == nn::DeviceType::AUTO, "Training override persisted");
    check(!network.predict({{0,0}}).selection.skipped.empty(), "NPU skip reason missing");
    network.setDevice(nn::DeviceType::CPU, 0);
    check(network.getDeviceName() == "Host CPU", "CPU configured name");
    // Updated weights survive switching configured preference.
    const auto weights = network.parameters();
    network.setDevice(nn::DeviceType::AUTO);
    check(network.parameters() == weights, "Switch lost parameters");
    // Whether AUTO with explicit index raised the expected argument error.
    bool rejected = false;
    try { network.setDevice(nn::DeviceType::AUTO, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "AUTO index not rejected");
}
