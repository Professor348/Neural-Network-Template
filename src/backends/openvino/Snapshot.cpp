#include "Snapshot.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace nn::detail {
/* Convert finite canonical value to float32 for name; returns the conversion or throws before inference on overflow. */
float npuFloat(double value, const char* name) {
    if (!std::isfinite(value) || std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::runtime_error(std::string("NPU reduced-precision capability: ") + name + " is not finite float32 data");
    return static_cast<float>(value);
}
/* Validate count times width for a float32 tensor; name labels errors; throws on zero/overflow; no output or inference. */
static void checkTensorSize(std::size_t count, std::size_t width, const char* name) {
    if (!count || !width || count > std::numeric_limits<std::size_t>::max() / width / sizeof(float) ||
        count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        width > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::runtime_error(std::string("NPU static shape unsupported/overflow: ") + name);
}
/* Validate layers and batch, returning nothing; throws runtime_error for malformed shapes/parameters/activations; no inference. */
void npuValidateGraph(const std::vector<Dense>& layers, std::size_t samples) {
    if (layers.empty()) throw std::runtime_error("NPU unsupported graph: at least one dense layer is required");
    // Expected input width is updated after each validated layer.
    std::size_t width = layers.front().inputs;
    // Current dense layer is checked before any graph storage is allocated.
    for (const auto& layer : layers) {
        checkTensorSize(samples, layer.inputs, "input/activation");
        checkTensorSize(samples, layer.outputs, "output/activation");
        checkTensorSize(layer.outputs, layer.inputs, "weights");
        if (layer.inputs != width || layer.weights.size() != layer.inputs * layer.outputs ||
            layer.biases.size() != layer.outputs)
            throw std::runtime_error("NPU unsupported graph: inconsistent dense architecture or parameter sizes");
        if (layer.activation != Activation::Linear && layer.activation != Activation::Sigmoid &&
            layer.activation != Activation::Tanh && layer.activation != Activation::Relu)
            throw std::runtime_error("NPU unsupported graph: unknown dense activation");
        // Canonical parameter under validation; conversion never changes the source model.
        for (const double value : layer.weights) npuFloat(value, "weight");
        // Canonical bias under validation.
        for (const double value : layer.biases) npuFloat(value, "bias");
        width = layer.outputs;
    }
}
/* Append exact canonical double bits from values to key; mutates key, returns nothing, allocation may throw. */
static void appendParameters(std::vector<std::uint64_t>& key, const std::vector<double>& values) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "Exact NPU snapshot keys require 64-bit double");
    key.push_back(values.size());
    // Original canonical value, including its signed-zero bit pattern.
    for (const double value : values) {
        // Raw bits avoid stale cache hits caused by float32 rounding or hash collisions.
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        key.push_back(bits);
    }
}
/* Encode validated device/layers/samples/loss as a full collision-free snapshot key; throws before any inference on bad data. */
std::vector<std::uint64_t> npuGraphKey(const std::string& device, const std::vector<Dense>& layers,
    std::size_t samples, bool loss) {
    npuValidateGraph(layers, samples);
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t), "NPU key dimensions require at most 64-bit size_t");
    // Length-delimited words encode all properties affecting graph compilation and parameter storage.
    std::vector<std::uint64_t> key{device.size()};
    // Runtime ID byte preserved without depending on host char signedness.
    for (const unsigned char character : device) key.push_back(character);
    key.push_back(samples);
    key.push_back(loss ? 1 : 0);
    key.push_back(layers.size());
    // Dense topology and all current canonical parameter bits form the immutable snapshot identity.
    for (const auto& layer : layers) {
        key.push_back(layer.inputs);
        key.push_back(layer.outputs);
        key.push_back(static_cast<std::uint64_t>(layer.activation));
        appendParameters(key, layer.weights);
        appendParameters(key, layer.biases);
    }
    return key;
}
}
