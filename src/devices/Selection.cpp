#include "Selection.h"
#include <stdexcept>
#ifdef NN_ENABLE_NPU
#include "backends/openvino/NpuSession.h"
#endif
#ifdef NN_ENABLE_GPU
#include "backends/opencl/OpenClSession.h"
#endif
namespace nn::detail {
/* Validate preference syntax; preference is the requested type/index; throws invalid_argument for invalid combinations. */
void validatePreference(DevicePreference preference) {
    if (preference.type != DeviceType::AUTO && preference.type != DeviceType::CPU &&
        preference.type != DeviceType::GPU && preference.type != DeviceType::NPU)
        throw std::invalid_argument("Unknown DeviceType");
    if (preference.type == DeviceType::AUTO && preference.index)
        throw std::invalid_argument("AUTO cannot specify a device index");
}
/* Resolve an inventory without execution; preference/operation specify request, devices is usable inventory,
 * absentReasons describes unavailable NPU/GPU/CPU; suitability optionally checks whole-model compilation.
 * Returns selection or throws on unsupported request; never executes numerical work. */
Selection selectDevice(DevicePreference preference, Operation operation, const std::vector<DeviceInfo>& devices,
    const std::vector<std::string>& absentReasons,
    const std::function<std::optional<std::string>(const DeviceInfo&)>& suitability) {
    validatePreference(preference);
    if (operation != Operation::Training && operation != Operation::Inference)
        throw std::invalid_argument("Unknown operation");
    // Resolution report accumulates skips before selecting a complete-operation device.
    Selection result;
    // Priority position also addresses absentReasons.
    std::size_t priority = 0;
    // Device category in fixed automatic priority order.
    for (const auto type : {DeviceType::NPU, DeviceType::GPU, DeviceType::CPU}) {
        // Whether inventory contains the requested type/index.
        bool found = false;
        // Inventory device in deterministic runtime order.
        for (const auto& device : devices) {
            if (device.type != type) continue;
            if (preference.type != DeviceType::AUTO &&
                (type != preference.type || device.index != preference.index.value_or(0))) continue;
            found = true;
            if ((operation == Operation::Training && device.training) ||
                (operation == Operation::Inference && device.inference)) {
                if (suitability) {
                    // Compile-only rejection occurs before any numerical execution and is safe for AUTO to skip.
                    const auto rejected = suitability(device);
                    if (rejected) { result.skipped.push_back(device.name + ": " + *rejected); continue; }
                }
                result.device = device;
                return result;
            }
            result.skipped.push_back(device.name + ": unsupported requested operation (training or inference)");
        }
        if (!found && (preference.type == DeviceType::AUTO || preference.type == type))
            result.skipped.push_back(priority < absentReasons.size() ? absentReasons[priority] : "Device missing or invalid index");
        ++priority;
    }
    // Error contains all known causes, including disabled backends.
    std::string message = "No device supports requested operation (explicit index may be invalid)";
    // Diagnostic for each rejected category/device.
    for (const auto& reason : result.skipped) message += "; " + reason;
    throw std::runtime_error(message);
}
/* Resolve against real inventory; preference/operation define request, suitability optionally checks model compilation; throws on unsupported selection. */
Selection resolve(DevicePreference preference, Operation operation,
    const std::function<std::optional<std::string>(const DeviceInfo&)>& suitability) {
    validatePreference(preference);
    // Explicit CPU calls avoid driver discovery and GPU program compilation entirely.
    if (preference.type == DeviceType::CPU)
        return selectDevice(preference, operation,
            {{DeviceType::CPU, 0, "Host CPU", "reference-cpu-float64", true, true}},
            {"NPU backend disabled", "GPU discovery not requested", "CPU index invalid: only CPU 0 is available"}, suitability);
    return selectDevice(preference, operation, getAvailableDevices(),
        {
#ifdef NN_ENABLE_NPU
         npuDiagnostic(),
#else
         "NPU backend disabled at build time: configure NN_ENABLE_NPU=ON with OpenVINO Runtime",
#endif
#ifdef NN_ENABLE_GPU
         openClDiagnostic(),
#else
         "GPU backend disabled at build time: configure NN_ENABLE_GPU=ON for OpenCL execution",
#endif
         "CPU index invalid: only CPU 0 is available"}, suitability);
}
}
namespace nn {
/* Return actual executable devices; this reference backend treats all host CPU cores as CPU index zero. */
std::vector<DeviceInfo> getAvailableDevices() {
    // Executable device inventory; GPU discovery includes runtime/precision/compiler probes.
    std::vector<DeviceInfo> devices;
#ifdef NN_ENABLE_NPU
    devices = detail::npuDevices();
#endif
#ifdef NN_ENABLE_GPU
    // Real GPU devices follow NPU candidates in the inventory.
    const auto gpu = detail::openClDevices();
    devices.insert(devices.end(), gpu.begin(), gpu.end());
#endif
    devices.push_back({DeviceType::CPU, 0, "Host CPU", "reference-cpu-float64", true, true});
    return devices;
}
/* Test preference usability; preference specifies type/index, operation optionally requires a particular capability. */
bool isDeviceAvailable(DevicePreference preference, std::optional<Operation> operation) {
    try {
        detail::resolve(preference, operation.value_or(Operation::Inference));
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::runtime_error&) {
        return false;
    }
}
}
