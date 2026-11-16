#pragma once
#include <nn/Types.h>
#include <openvino/openvino.hpp>
#include <mutex>
namespace nn::detail {
class NpuState {
public:
    /* Create a private infer request from a copied handle to fully checked compiled model; samples/inputWidth/outputWidth fix static I/O,
     * loss selects the prepared graph kind. No inference runs; allocation/request creation failures propagate. */
    NpuState(ov::CompiledModel model, std::size_t samples, std::size_t inputWidth,
        std::size_t outputWidth, bool loss);
    /* Upload inputs, execute prediction entirely on the chosen NPU and return widened float32 results.
     * Serializes this request; throws on data/runtime/nonfinite errors; never retries another device. */
    Dataset predict(const Dataset& inputs);
    /* Upload inputs/targets and execute the complete MSE graph entirely on the chosen NPU; return its scalar output.
     * Serializes this request; throws on data/runtime/nonfinite errors; never computes loss on the host or retries. */
    double loss(const Dataset& inputs, const Dataset& targets);
private:
    /* Check exact prepared shape and finite float32 range of data, then fill input port of expected width.
     * name identifies diagnostics; mutates only private input storage, never infers, throws runtime_error on bad data. */
    void upload(const Dataset& data, std::size_t port, std::size_t width, const char* name);
    /* Execute once and return float32 output tensor; caller holds mutex_ and keeps the tensor private until copied.
     * Throws runtime_error on SDK failures without retry/fallback; performs no host graph operations. */
    ov::Tensor execute();
    // Independent mutable infer request; compiled model snapshots are shared but never their request state.
    ov::InferRequest request_;
    // Exact static batch for which this graph was checked and compiled.
    std::size_t samples_;
    // Per-sample input feature count.
    std::size_t inputWidth_;
    // Per-sample output/target feature count.
    std::size_t outputWidth_;
    // True only for the prepared two-input, fully device-reduced MSE graph.
    bool loss_;
    // Serializes uploads, execution and output reads when callers share this one session.
    std::mutex mutex_;
};
}
