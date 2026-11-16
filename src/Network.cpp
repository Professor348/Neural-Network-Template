#include <nn/Network.h>
#include "Validation.h"
#include "devices/Selection.h"
#include "backends/Execution.h"
#include "numerics/Functions.h"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace nn {
/* Initialize input width, seeded RNG and preference; inputs/seed/device configure the empty model; throws on invalid configuration. */
Network::Network(std::size_t inputs, std::uint32_t seed, DevicePreference device)
    : inputs_(inputs), generator_(seed), device_(device) {
    if (!inputs) throw std::invalid_argument("Network input width must be positive");
    setDevice(device.type, device.index);
}
/* Delegate construction; inputs/seed configure model, type/index configure preference; same exceptions as primary constructor. */
Network::Network(std::size_t inputs, DeviceType type, std::optional<std::size_t> index, std::uint32_t seed)
    : Network(inputs, seed, DevicePreference{type, index}) {}
/* Append initialized dense parameters; outputs is width, activation chooses nonlinearity; throws on invalid values/allocation failure. */
void Network::addDense(std::size_t outputs, Activation activation) {
    detail::activate(0, activation);
    // Input dimension is previous output width or network input width.
    const std::size_t width = layers_.empty() ? inputs_ : layers_.back().outputs;
    if (!outputs || outputs > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("Dense dimensions must be positive and fit in size_t");
    // Xavier uniform bound preserves scale across layers.
    const double limit = std::sqrt(6.0 / (static_cast<double>(width) + static_cast<double>(outputs)));
    // Independent weight distribution, zero biases break symmetry through random weights.
    std::uniform_real_distribution<double> distribution(-limit, limit);
    // Fully initialized layer staged before architecture mutation.
    Dense layer{width, outputs, activation, std::vector<double>(width * outputs), std::vector<double>(outputs, 0)};
    // Weight scalar drawn from the seeded generator.
    for (auto& weight : layer.weights) weight = distribution(generator_);
    layers_.push_back(std::move(layer));
}
/* Select supported loss; loss is enum to validate; throws invalid_argument for unknown loss. */
void Network::setLoss(Loss loss) {
    if (loss != Loss::MeanSquaredError) throw std::invalid_argument("Only MeanSquaredError is implemented");
}
/* Replace preference only after validation; type/index specify processor; throws without mutating preference on failure. */
void Network::setDevice(DeviceType type, std::optional<std::size_t> index) {
    // Candidate configured preference.
    const DevicePreference preference{type, index};
    detail::validatePreference(preference);
    if (type != DeviceType::AUTO) detail::resolve(preference, Operation::Inference);
    device_ = preference;
}
/* Configure explicit NPU reduced-precision permission; options affects future inference only, with no parameter changes. */
void Network::setNpuOptions(NpuOptions options) { npuOptions_ = options; }
/* Return current NPU precision permission without resolving or executing devices. */
NpuOptions Network::getNpuOptions() const { return npuOptions_; }
/* Return the persistent preference without resolving AUTO. */
DevicePreference Network::getDevice() const { return device_; }
/* Return configured device name or Auto; explicit configurations were validated by the setter. */
std::string Network::getDeviceName() const {
    return device_.type == DeviceType::AUTO ? "Auto" : detail::resolve(device_, Operation::Inference).device.name;
}
/* Execute inputs on resolved call-only device override; returns values and selection; throws for invalid data/device/numerics. */
Prediction Network::predict(const Dataset& inputs, std::optional<DevicePreference> device) const {
    if (layers_.empty()) throw std::invalid_argument("Add at least one dense layer before inference");
    detail::validateData(inputs, inputs_, "Inputs");
    // Model-aware selection and compilation complete before numerical inference begins.
    detail::Execution execution(device.value_or(device_), Operation::Inference, layers_, inputs.size(), false, npuOptions_);
    // Outputs include the actual resolved device and every pre-execution skip reason.
    Prediction result{execution.predict(inputs), execution.selection()};
    return result;
}
/* Evaluate MSE on inputs/targets using optional call-only device override; throws on validation or computation failures. */
double Network::evaluate(const Dataset& inputs, const Dataset& targets, std::optional<DevicePreference> device) const {
    detail::validateSupervised(layers_, inputs, targets, inputs_);
    // Resolved backend for the complete loss operation.
    detail::Execution execution(device.value_or(device_), Operation::Inference, layers_, inputs.size(), true, npuOptions_);
    return execution.loss(inputs, targets);
}
/* Return analytic mean gradients for inputs/targets on optional training device; throws before returning nonfinite derivatives. */
std::vector<double> Network::gradients(const Dataset& inputs, const Dataset& targets, std::optional<DevicePreference> device) const {
    detail::validateSupervised(layers_, inputs, targets, inputs_);
    // Resolved training backend for the complete differentiation operation.
    detail::Execution execution(device.value_or(device_), Operation::Training, layers_, inputs.size(), false, npuOptions_);
    return execution.gradient(inputs, targets);
}
/* Flatten all parameters in documented layer, weight, bias order; returns an independent copy. */
std::vector<double> Network::parameters() const {
    // Exact flattened count avoids repeated vector growth while copying parameters.
    std::size_t count = 0;
    // Layer whose weight and bias sizes contribute to the output allocation.
    for (const auto& layer : layers_) count += layer.weights.size() + layer.biases.size();
    // Flattened parameter copy.
    std::vector<double> result;
    result.reserve(count);
    // Dense layer whose parameters are copied.
    for (const auto& layer : layers_) {
        result.insert(result.end(), layer.weights.begin(), layer.weights.end());
        result.insert(result.end(), layer.biases.begin(), layer.biases.end());
    }
    return result;
}
/* Validate and atomically replace parameters; values is exact flattened vector; throws invalid_argument without partial mutation. */
void Network::setParameters(const std::vector<double>& values) {
    // Exact expected count, computed without allocating or copying the existing parameters.
    std::size_t count = 0;
    // Layer whose parameter sizes contribute to validation.
    for (const auto& layer : layers_) count += layer.weights.size() + layer.biases.size();
    if (values.size() != count) throw std::invalid_argument("Parameter count mismatch");
    // Candidate scalar checked before any assignment.
    for (const double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("Parameters must be finite");
    // Next flattened parameter index.
    std::size_t index = 0;
    // Layer receiving validated parameter values.
    for (auto& layer : layers_) {
        // Weight reference in output-major order.
        for (auto& weight : layer.weights) weight = values[index++];
        // Bias reference in output order.
        for (auto& bias : layer.biases) bias = values[index++];
    }
}
}
