#pragma once
#include "Types.h"
#include "Enums.h"
namespace nn {
struct Progress {
    // Completed epoch count, starting at one.
    std::size_t epoch;
    // Mean full-dataset loss after this epoch's updates.
    double loss;
};
}
