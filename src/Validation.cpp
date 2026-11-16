#include "Validation.h"
#include <cmath>
#include <stdexcept>
namespace nn::detail {
/* Validate data rows against width and finite values; name labels exceptions; no mutation. */
void validateData(const Dataset& data, std::size_t width, const char* name) {
    if (data.empty()) throw std::invalid_argument(std::string(name) + " must not be empty");
    // Dataset row containing one sample.
    for (const auto& row : data) {
        if (row.size() != width) throw std::invalid_argument(std::string(name) + " feature dimension mismatch");
        // Scalar feature or label, required to be finite.
        for (const double value : row)
            if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " contains nonfinite values");
    }
}
/* Validate model and paired datasets; layers describes model, inputs/targets hold samples, width is input dimension. */
void validateSupervised(const std::vector<Dense>& layers, const Dataset& inputs, const Dataset& targets, std::size_t width) {
    if (layers.empty()) throw std::invalid_argument("Add at least one dense layer before executing");
    validateData(inputs, width, "Inputs");
    validateData(targets, layers.back().outputs, "Targets");
    if (inputs.size() != targets.size()) throw std::invalid_argument("Input and target sample counts differ");
}
/* Reject invalid options before training mutates parameters; options contains epochs, batch size and SGD step. */
void validateOptions(const TrainingOptions& options) {
    if (!options.epochs || !options.batchSize) throw std::invalid_argument("Epochs and batchSize must be positive");
    if (!std::isfinite(options.optimizer.learningRate) || options.optimizer.learningRate <= 0)
        throw std::invalid_argument("SGD learningRate must be finite and positive");
}
}
