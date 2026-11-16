#pragma once
#include "Types.h"
#include "Enums.h"
#include "Selection.h"
namespace nn {
struct Prediction {
    // Samples by output features.
    Dataset values;
    // Device that executed this inference.
    Selection selection;
};
}
