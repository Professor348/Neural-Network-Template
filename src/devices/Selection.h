#pragma once
#include <nn/Devices.h>
#include <functional>
namespace nn::detail {
/* Validate preference syntax. preference: request to inspect; throws invalid_argument for AUTO/index or invalid type. */
void validatePreference(DevicePreference preference);
/* Resolve a request against a usable inventory. preference: requested type/index; operation: required capability;
 * devices: backend-provided inventory with stable per-type indices; absentReasons: unavailable type diagnostics
 * in NPU/GPU/CPU order. Returns selection or throws runtime_error; never executes any backend.
 * suitability: optional compile-only complete-model check, returning a rejection reason or nullopt.
 * It must not execute numerical inference/training; exceptions propagate without retry.
 * This pure policy is testable with synthetic inventories without simulating numerical execution. */
Selection selectDevice(DevicePreference preference, Operation operation, const std::vector<DeviceInfo>& devices,
    const std::vector<std::string>& absentReasons,
    const std::function<std::optional<std::string>(const DeviceInfo&)>& suitability = {});
/* Resolve using real compiled backends. preference: type/index; operation: required capability.
 * suitability: optional compile-only model check; returns selection or throws; no numerical execution. */
Selection resolve(DevicePreference preference, Operation operation,
    const std::function<std::optional<std::string>(const DeviceInfo&)>& suitability = {});
}
