#pragma once
#include "Dense.h"
#include "Devices.h"
#include "TrainingOptions.h"
#include "TrainingResult.h"
#include "Prediction.h"
#include "Progress.h"
#include "NpuOptions.h"
#include <functional>
#include <random>
namespace nn {
class Network {
public:
    /* Create an empty network. inputs: positive input width; seed: initialization seed;
     * device: configured preference; throws invalid_argument for bad dimensions/preferences,
     * runtime_error for unavailable explicit devices. Does not execute numerical work. */
    explicit Network(std::size_t inputs, std::uint32_t seed = 0, DevicePreference device = {});
    /* Construct with type/index convenience arguments. inputs and seed as above;
     * type: device category; index: optional zero-based index. Same exceptions as primary constructor. */
    Network(std::size_t inputs, DeviceType type, std::optional<std::size_t> index = std::nullopt, std::uint32_t seed = 0);
    /* Append a Xavier-initialized dense layer. outputs: positive width; activation: nonlinearity.
     * Mutates architecture and initialization RNG; throws invalid_argument for bad values. */
    void addDense(std::size_t outputs, Activation activation = Activation::Linear);
    /* Set mean loss function. loss: supported loss enum; throws invalid_argument for unknown values. */
    void setLoss(Loss loss);
    /* Set configured preference without changing weights. type/index: device request;
     * throws invalid_argument for AUTO/index or invalid enum; runtime_error for unavailable device. */
    void setDevice(DeviceType type, std::optional<std::size_t> index = std::nullopt);
    /* Set NPU inference options. options: explicit reduced-precision permission, disabled by default.
     * Changes only device suitability for later inference; never changes weights or training precision. */
    void setNpuOptions(NpuOptions options);
    /* Return configured NPU precision permission; no numerical computation or mutation. */
    NpuOptions getNpuOptions() const;
    /* Return configured preference, which may differ from the actual device selected per call. */
    DevicePreference getDevice() const;
    /* Return configured explicit device's name, or Auto. No computation occurs. */
    std::string getDeviceName() const;
    /* Infer all samples. inputs: nonempty finite samples; device: call-only override.
     * Returns outputs and actual selection; throws invalid_argument for data, runtime_error for
     * unavailable/unsupported execution. Does not mutate preference or parameters. */
    Prediction predict(const Dataset& inputs, std::optional<DevicePreference> device = std::nullopt) const;
    /* Compute mean loss. inputs/targets: matching finite samples; device: call-only override.
     * Returns scalar MSE; same validation/device exceptions as predict, plus target shape checks. */
    double evaluate(const Dataset& inputs, const Dataset& targets, std::optional<DevicePreference> device = std::nullopt) const;
    /* Compute mean-loss derivatives without mutation. inputs/targets: matching dataset;
     * device: call-only training override. Returns flattened gradient in parameters() order.
     * Throws for invalid data or unsupported training. */
    std::vector<double> gradients(const Dataset& inputs, const Dataset& targets, std::optional<DevicePreference> device = std::nullopt) const;
    /* Train using minibatch SGD. inputs/targets: finite matching dataset; options: epochs/batch/SGD/seed;
     * callback: optional synchronous completed-epoch observer; device: call-only override.
     * Returns loss summary and selection. Mutates weights; callback exceptions propagate with
     * completed updates retained. Invalid options/data throw before updates. Numerical failures throw
     * without retrying another device; earlier successful batch updates remain. Not thread-safe. */
    TrainingResult train(const Dataset& inputs, const Dataset& targets, const TrainingOptions& options = {},
        const std::function<void(const Progress&)>& callback = {}, std::optional<DevicePreference> device = std::nullopt);
    /* Return weights then biases for each layer, weights in output-major order. No mutation. */
    std::vector<double> parameters() const;
    /* Replace parameters atomically after validation. values: exact flattened count, all finite.
     * Throws invalid_argument on bad count/nonfinite values. Cached immutable NPU graphs are keyed by all current parameter bits. */
    void setParameters(const std::vector<double>& values);
private:
    // Network input width in features.
    std::size_t inputs_;
    // Initialization generator, independent of training shuffle generator.
    std::mt19937 generator_;
    // Persistent caller preference, never replaced by automatic resolution.
    DevicePreference device_;
    // Explicit permission for reduced-precision NPU inference; disabled unless caller opts in.
    NpuOptions npuOptions_;
    // Ordered feedforward architecture and latest host-owned parameters.
    std::vector<Dense> layers_;
};
}
