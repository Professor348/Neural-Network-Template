#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace nn {
// One vector of features per sample; all rows must have the same width.
using Dataset = std::vector<std::vector<double>>;
}
