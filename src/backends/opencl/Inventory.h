#pragma once
#include "OpenClRuntime.h"
#include <nn/DeviceInfo.h>
#include <string>
#include <vector>
namespace nn::detail {
struct Inventory {
    // GPU handles in the same index order as info; all passed capability/build probes.
    std::vector<cl_device_id> handles;
    // Public descriptors corresponding to handles.
    std::vector<DeviceInfo> info;
    // Diagnostic for rejected devices or missing runtime/hardware.
    std::string diagnostic;
};
/* Return process-lifetime immutable GPU discovery snapshot; construction probes runtime/compiler, never executes numerics. */
const Inventory& inventory();
/* Check OpenCL code; operation describes API action, code is runtime status; throws runtime_error with code on failure. */
void checkCl(cl_int code, const char* operation);
/* Build embedded kernel for device within context; returns owned program or throws with compiler log, cleaning failed program. */
cl_program buildProgram(cl_context context, cl_device_id device);
}
