#include <nn/Network.h>
#include "Validation.h"
#include "devices/Selection.h"
#include "backends/Execution.h"
#include <algorithm>
#include <numeric>
namespace nn {
/* Train inputs/targets with options and optional synchronous callback on call-only device override.
 * Returns full-dataset loss summary; updates parameters batchwise; exceptions preserve completed updates. */
TrainingResult Network::train(const Dataset& inputs, const Dataset& targets, const TrainingOptions& options,
    const std::function<void(const Progress&)>& callback, std::optional<DevicePreference> device) {
    detail::validateSupervised(layers_, inputs, targets, inputs_);
    detail::validateOptions(options);
    // One model-aware backend session executes all epochs; inference-only NPUs are skipped before computation.
    detail::Execution execution(device.value_or(device_), Operation::Training, layers_, inputs.size(), false, npuOptions_);
    // Original full-dataset mean loss.
    const double initial = execution.loss(inputs, targets);
    // Current completed-epoch mean loss.
    double loss = initial;
    // Local shuffle state resets for reproducibility on each training call.
    std::mt19937 generator(options.seed);
    // Permutation of sample indices for minibatch formation.
    std::vector<std::size_t> order(inputs.size());
    std::iota(order.begin(), order.end(), 0);
    // Zero-based epoch index, bounded to avoid overflow at SIZE_MAX epochs.
    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
        if (options.shuffle) std::shuffle(order.begin(), order.end(), generator);
        // First sample index in the next minibatch.
        for (std::size_t start = 0; start < order.size();) {
            // Exclusive batch end computed without overflow.
            const std::size_t end = start + std::min(options.batchSize, order.size() - start);
            // Current batch inputs.
            Dataset batchInputs;
            // Current batch targets.
            Dataset batchTargets;
            // Permutation position for the current sample.
            for (std::size_t position = start; position < end; ++position) {
                batchInputs.push_back(inputs[order[position]]);
                batchTargets.push_back(targets[order[position]]);
            }
            // Complete SGD step runs on selected device; synchronize its latest weights to canonical host storage.
            setParameters(execution.step(batchInputs, batchTargets, options.optimizer.learningRate));
            start = end;
        }
        loss = execution.loss(inputs, targets);
        if (callback) callback(Progress{epoch + 1, loss});
    }
    return {initial, loss, options.epochs, execution.selection()};
}
}
