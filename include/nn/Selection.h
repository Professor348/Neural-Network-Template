#pragma once
#include "Types.h"
#include "Enums.h"
#include "DeviceInfo.h"
namespace nn {
struct Selection {
    // Actual processor used for the complete operation.
    DeviceInfo device;
    // Reasons higher priority devices were not selected.
    std::vector<std::string> skipped;
};
}
