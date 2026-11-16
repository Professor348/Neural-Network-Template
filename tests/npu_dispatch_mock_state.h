#pragma once
#include <nn/Dense.h>

namespace nn::detail {
// Test-only session snapshot; this class contains no OpenVINO or NPU hardware implementation.
class NpuState {
public:
    // Independent canonical parameter snapshot captured when this fake graph is prepared.
    std::vector<Dense> layers;
    // Exact sample count accepted by the fake prepared graph.
    std::size_t samples = 0;
    // Whether preparation requested a complete device loss graph.
    bool withLoss = false;
};
}
