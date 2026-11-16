#pragma once
#include <nn/Dense.h>
#include <nn/DeviceInfo.h>
#include <openvino/openvino.hpp>
#include <list>
#include <mutex>
namespace nn::detail {
class NpuRuntime {
public:
    /* Initialize one immutable-configuration Core and snapshot physical NPU discovery without inference.
     * Captures plugin/hardware discovery failures in diagnostics; Core initialization/allocation may throw. */
    NpuRuntime();
    /* Return discovered descriptors by value; no inference or mutations; allocation may throw. */
    std::vector<DeviceInfo> devices() const;
    /* Return discovery/index diagnostic by value; no inference or mutations; allocation may throw. */
    std::string diagnostic() const;
    /* Obtain compiled snapshot for device index/layers/static samples/operation loss.
     * Fully checks support, precision and exclusive NPU execution before returning; never infers.
     * Serializes bounded eight-entry cache mutation, comparing every key word; throws runtime_error on rejection. */
    ov::CompiledModel compiled(std::size_t index, const std::vector<Dense>& layers, std::size_t samples, bool loss);
private:
    /* Discover and sort physical IDs; updates ids_/info_/diagnostic_ only at construction; never infers.
     * Throws runtime error when the plugin/runtime cannot be queried; caller retains diagnostics. */
    void discover();
    // Core configuration is never changed with set_property; OpenVINO's compile_model API itself is non-const.
    // Per-model properties cannot leak between calls or threads, and compilation is serialized by cacheMutex_.
    ov::Core core_;
    // Exact runtime identifiers, aligned with info_ and sorted lexicographically.
    std::vector<std::string> ids_;
    // Discovery descriptors explicitly advertise inference and exclude training.
    std::vector<DeviceInfo> info_;
    // Generic NPU/NPU.0 aliases are unambiguous only if the raw runtime exposes exactly one physical NPU.
    bool soleDevice_ = false;
    // Retains honest plugin/hardware/unknown-driver and rejected-device explanations.
    std::string diagnostic_;
    // Serializes cache lookup, compilation and LRU updates; requests are session-private.
    std::mutex cacheMutex_;
    // Full key words and immutable compiled models; most recently used snapshot is first; at most eight entries.
    std::list<std::pair<std::vector<std::uint64_t>, ov::CompiledModel>> cache_;
};
/* Return the process-wide thread-safely initialized runtime; creates no requests and performs no inference.
 * Core initialization exceptions propagate so discovery or session callers can attach appropriate context. */
NpuRuntime& npuRuntime();
}
