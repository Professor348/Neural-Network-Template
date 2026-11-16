#pragma once
#include <nn/Network.h>
namespace nn::detail {
/* Validate rectangular finite samples. data: sample matrix; width: required features; name: error label.
 * Throws invalid_argument for empty, ragged, wrong-sized or nonfinite data. */
void validateData(const Dataset& data, std::size_t width, const char* name);
/* Validate matching supervised data. layers: nonempty architecture; inputs/targets: sample matrices;
 * width: input features. Throws invalid_argument for data or missing architecture. */
void validateSupervised(const std::vector<Dense>& layers, const Dataset& inputs, const Dataset& targets, std::size_t width);
/* Validate SGD configuration. options: per-call training settings; throws invalid_argument for invalid values. */
void validateOptions(const TrainingOptions& options);
}
