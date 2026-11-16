#pragma once
namespace nn {
struct NpuOptions {
    // Explicit permission for f32 NPU input/output and FP16 internal arithmetic; CPU/GPU remain float64.
    bool allowReducedPrecision = false;
};
}
