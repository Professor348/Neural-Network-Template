#pragma once
#include "Types.h"
#include "Enums.h"
#include "Optimizer.h"
namespace nn {
struct TrainingOptions {
    // Number of complete dataset passes; strictly positive.
    std::size_t epochs = 1000;
    // Samples per SGD update; final batch may be smaller.
    std::size_t batchSize = 4;
    // Shuffle sample indices at the start of each epoch.
    bool shuffle = true;
    // Seed for per-call sample ordering; initialization uses the constructor seed.
    std::uint32_t seed = 0;
    // Optimizer configuration for this call.
    Optimizer optimizer;
};
}
