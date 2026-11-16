#include "NpuRuntime.h"
#include "Graph.h"
#include "Snapshot.h"
#include <algorithm>
#include <new>
#include <stdexcept>
namespace nn::detail {
/* Test id for an exclusive NPU runtime name, rejecting composite devices and CPU/GPU; returns bool without side effects. */
static bool isNpu(const std::string& id) {
    if (id == "NPU") return true;
    if (id.size() <= 4 || id.compare(0, 4, "NPU.") != 0) return false;
    return id.find_first_of(":,() ") == std::string::npos;
}
/* Check reported placement matches selected physical ID; aliases permits only sole-device NPU/NPU.0 equivalence.
 * Returns bool without side effects; explicit mismatched indices and ambiguous generic NPU placements are rejected. */
static bool sameNpu(const std::string& reported, const std::string& selected, bool aliases) {
    if (!isNpu(reported) || !isNpu(selected)) return false;
    if (reported == selected) return true;
    return aliases && ((reported == "NPU" && selected == "NPU.0") ||
        (reported == "NPU.0" && selected == "NPU"));
}
/* Construct runtime/discovery snapshot without inference; capture discovery errors while Core initialization may throw. */
NpuRuntime::NpuRuntime() {
    try { discover(); }
    catch (const std::bad_alloc& error /* host memory exhaustion is not unsupported hardware */) {
        (void)error;
        throw;
    }
    catch (const std::exception& error /* plugin initialization or runtime enumeration failure */) {
        ids_.clear();
        info_.clear();
        diagnostic_ = std::string("NPU OpenVINO discovery failed; hardware/driver status unknown: ") + error.what();
    }
}
/* Discover exclusively NPU runtime IDs; update ordered descriptors/diagnostics; no compile/inference; runtime errors propagate. */
void NpuRuntime::discover() {
    try {
        // Loading the NPU plugin separately distinguishes an unavailable plugin from an empty hardware inventory.
        const auto versions = core_.get_versions("NPU");
        if (versions.empty()) {
            diagnostic_ = "NPU OpenVINO plugin unavailable: no plugin version reported; hardware/driver status unknown";
            return;
        }
    } catch (const std::bad_alloc& error /* preserve host memory exhaustion for the operation caller */) {
        (void)error;
        throw;
    } catch (const std::exception& error /* plugin load failure may also originate in driver initialization */) {
        diagnostic_ = std::string("NPU OpenVINO plugin unavailable or failed to initialize; hardware/driver status unknown: ") + error.what();
        return;
    }
    // OpenVINO's actual runtime names are the only names permitted at query/compile time.
    auto candidates = core_.get_available_devices();
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        /* Return true for id belonging to another device category; no side effects or exceptions. */
        [](const std::string& id) { return !isNpu(id); }), candidates.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    soleDevice_ = candidates.size() == 1;
    // Enumerated NPU runtime ID; the ordered accepted descriptors determine stable public indices.
    for (const auto& id : candidates) {
        try {
            // Hardware label is queried from this exact NPU, without compiling any model.
            const auto name = core_.get_property(id, ov::device::full_name);
            // Published capabilities must include the explicitly permitted FP16 internal arithmetic.
            const auto capabilities = core_.get_property(id, ov::device::capabilities);
            if (std::find(capabilities.begin(), capabilities.end(), "FP16") == capabilities.end())
                throw std::runtime_error("required FP16 inference capability was not advertised");
            info_.push_back({DeviceType::NPU, ids_.size(), name + " (" + id + ")",
                "openvino-npu-float32-io-float16", false, true});
            ids_.push_back(id);
        } catch (const std::bad_alloc& error /* allocation failure must not silently change device indices */) {
            (void)error;
            throw;
        } catch (const std::exception& error /* rejected NPU remains visible through diagnostics */) {
            diagnostic_ += id + ": unavailable or unsupported capability: " + error.what() + "; ";
        }
    }
    if (candidates.empty())
        diagnostic_ = "NPU OpenVINO plugin is present but exposes no NPU hardware; hardware may be absent or its driver unavailable (driver status unknown)";
    if (!ids_.empty())
        diagnostic_ += "NPU indices use lexicographic OpenVINO runtime-ID order [0," + std::to_string(ids_.size()) +
            "); inference requires explicit reduced-precision opt-in and per-model query/compilation support; training unsupported";
}
/* Return immutable discovery descriptors by value; no runtime calls or inference; allocation may throw. */
std::vector<DeviceInfo> NpuRuntime::devices() const { return info_; }
/* Return discovery/index explanation by value; no runtime calls or inference; allocation may throw. */
std::string NpuRuntime::diagnostic() const { return diagnostic_; }
/* Verify graph's complete query support on exact device with properties; core remains configuration-immutable.
 * aliases permits only unambiguous sole-device NPU/NPU.0 equivalence. Returns nothing and never infers;
 * throws runtime_error for missing nodes or placement differing from the selected NPU. */
static void requireCompleteSupport(const ov::Core& core, const std::shared_ptr<ov::Model>& graph,
    const std::string& device, const ov::AnyMap& properties, bool aliases) {
    // Query coverage uses unique graph node names, including inputs/constants/results and the entire loss subgraph.
    const auto supported = core.query_model(graph, device, properties);
    // Original graph node whose assignment must be reported by the selected physical NPU plugin.
    for (const auto& node : graph->get_ordered_ops()) {
        // Query result for this exact unique friendly name; absent entries prohibit compilation/execution.
        const auto found = supported.find(node->get_friendly_name());
        if (found == supported.end() || !sameNpu(found->second, device, aliases))
            throw std::runtime_error("NPU unsupported graph on " + device + ": query_model did not assign " +
                node->get_friendly_name() + " (" + node->get_type_name() + ") exclusively to selected " + device);
    }
}
/* Verify compiled snapshot has exact f32 static I/O and exclusive NPU/f16 configuration for graph/device/loss.
 * aliases permits only sole-device NPU/NPU.0 equivalence; returns nothing, never infers,
 * and throws runtime_error if the runtime cannot establish these requirements. */
static void requireCompiledContract(const ov::CompiledModel& compiled, const std::shared_ptr<ov::Model>& graph,
    const std::string& device, bool aliases) {
    // Plugin-reported execution placement is checked before creating any inference request.
    const auto execution = compiled.get_property(ov::execution_devices);
    if (execution.size() != 1 || !sameNpu(execution.front(), device, aliases))
        throw std::runtime_error("NPU compile rejected for " + device + ": execution_devices is not exactly the selected NPU");
    if (compiled.get_property(ov::hint::inference_precision) != ov::element::f16)
        throw std::runtime_error("NPU compile rejected for " + device + ": runtime did not accept explicit float16 inference precision");
    if (compiled.inputs().size() != graph->inputs().size() || compiled.outputs().size() != 1)
        throw std::runtime_error("NPU compile rejected: unexpected compiled graph interface");
    // Input port index whose exact static shape and element type must survive compilation.
    for (std::size_t index = 0; index < graph->inputs().size(); ++index) {
        if (compiled.input(index).get_element_type() != ov::element::f32 ||
            compiled.input(index).get_shape() != graph->input(index).get_shape())
            throw std::runtime_error("NPU compile rejected: float32 static input contract changed");
    }
    if (compiled.output().get_element_type() != ov::element::f32 ||
        compiled.output().get_shape() != graph->output().get_shape())
        throw std::runtime_error("NPU compile rejected: float32 static output contract changed");
}
/* Obtain the fully validated immutable snapshot for index/layers/samples/loss, updating eight-entry LRU; never infers. */
ov::CompiledModel NpuRuntime::compiled(std::size_t index, const std::vector<Dense>& layers, std::size_t samples, bool loss) {
    if (index >= ids_.size()) throw std::runtime_error("NPU index unavailable: " + diagnostic_);
    // Exact physical runtime identifier copied from immutable discovery; composite/CPU names cannot enter the cache.
    const auto& device = ids_[index];
    // Exact words include all canonical double bits, independent of graph float32 rounding.
    const auto key = npuGraphKey(device, layers, samples, loss);
    // Compilation and LRU mutation are serialized; inference remains independent for every private request.
    const std::lock_guard<std::mutex> lock(cacheMutex_);
    // Cached snapshot candidate whose entire key must compare equal to reuse it.
    for (auto found = cache_.begin(); found != cache_.end(); ++found) {
        if (found->first == key) {
            cache_.splice(cache_.begin(), cache_, found);
            return cache_.front().second;
        }
    }
    // The same local properties are supplied to both support query and compilation; Core defaults never change.
    const ov::AnyMap properties{ov::hint::inference_precision(ov::element::f16),
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
    // Entire static dense/prediction-or-loss graph, including fresh canonical weight snapshots.
    const auto graph = buildNpuGraph(layers, samples, loss);
    requireCompleteSupport(core_, graph, device, properties, soleDevice_);
    // Direct compilation never names AUTO/HETERO/MULTI/CPU; rejected models cannot trigger a fallback device.
    const auto model = core_.compile_model(graph, device, properties);
    requireCompiledContract(model, graph, device, soleDevice_);
    cache_.emplace_front(key, model);
    if (cache_.size() > 8) cache_.pop_back();
    return model;
}
/* Return one process-wide runtime; initialization is thread-safe, discovery performs no inference, exceptions propagate. */
NpuRuntime& npuRuntime() {
    // Immutable Core configuration and shared snapshot cache outlive all sessions created during normal program execution.
    static NpuRuntime runtime;
    return runtime;
}
}
