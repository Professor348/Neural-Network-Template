#pragma once
#include "Types.h"
#include "Enums.h"
namespace nn {
struct Dense {
    // Number of input features.
    std::size_t inputs;
    // Number of output features.
    std::size_t outputs;
    // Elementwise nonlinearity.
    Activation activation;
    // Output-major matrix, outputs times inputs scalar weights.
    std::vector<double> weights;
    // One additive offset per output feature.
    std::vector<double> biases;
};
}
