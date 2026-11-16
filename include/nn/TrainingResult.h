#pragma once
#include "Types.h"
#include "Enums.h"
#include "Selection.h"
namespace nn {
struct TrainingResult {
    // Mean full-dataset loss before any updates.
    double initialLoss;
    // Mean full-dataset loss after the final epoch.
    double finalLoss;
    // Number of successfully completed epochs.
    std::size_t epochs;
    // Device resolution and reasons for skips.
    Selection selection;
};
}
