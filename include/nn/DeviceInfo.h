#pragma once
#include "Types.h"
#include "Enums.h"
namespace nn {
struct DeviceInfo {
    // Processor category, never AUTO.
    DeviceType type = DeviceType::CPU;
    // Zero-based index within type, in backend-documented deterministic discovery order.
    std::size_t index = 0;
    // Human-readable device label.
    std::string name;
    // Execution implementation name.
    std::string backend;
    // Whether training is implemented for the supported model family.
    bool training = false;
    // Whether inference is implemented for the supported model family.
    bool inference = false;
};
}
