#pragma once
#include "Types.h"
#include "Enums.h"
namespace nn {
struct DevicePreference {
    // Requested processor type; AUTO chooses NPU, then GPU, then CPU.
    DeviceType type = DeviceType::AUTO;
    // Zero-based index within type; absent means index zero for explicit types.
    std::optional<std::size_t> index;
};
}
