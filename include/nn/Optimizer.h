#pragma once
#include "Types.h"
#include "Enums.h"
namespace nn {
struct Optimizer {
    // SGD step size; finite and strictly positive.
    double learningRate = 0.1;
};
}
