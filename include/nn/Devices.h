#pragma once
#include "DevicePreference.h"
#include "Selection.h"
namespace nn {
/* Return executable devices for this build. No accelerator detection is advertised as execution. */
std::vector<DeviceInfo> getAvailableDevices();
/* Test preference for basic usability, or operation capability when operation is supplied.
 * preference: processor type/index; operation: optional training/inference. Returns false on invalid usage. */
bool isDeviceAvailable(DevicePreference preference, std::optional<Operation> operation = std::nullopt);
}
