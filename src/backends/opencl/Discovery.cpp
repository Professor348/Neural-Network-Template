#include "Inventory.h"
#include "KernelSource.h"
#include <algorithm>
#include <stdexcept>
#include <tuple>
namespace nn::detail {
/* Throw descriptive runtime_error for non-success code; operation identifies failed API; no fallback. */
void checkCl(cl_int code, const char* operation) {
    if (code != CL_SUCCESS) throw std::runtime_error(std::string("OpenCL ") + operation + " failed (status " + std::to_string(code) + ")");
}
/* Compile production kernel for context/device; returns owned program or throws compiler log, releasing failed program. */
cl_program buildProgram(cl_context context, cl_device_id device) {
    // OpenCL status from program creation.
    cl_int status = CL_SUCCESS;
    // Program ownership transfers to caller only after successful build.
    // Mutable pointer variable required by the OpenCL 1.2 source-array signature.
    const char* source = openClSource;
    // Newly allocated program, released if compilation fails.
    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &status);
    checkCl(status, "create program");
    status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        // Build log bytes including terminator.
        std::size_t size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        // Compiler diagnostics, captured before releasing failed program.
        std::string log(size, '\0');
        if (size) clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL GPU runtime/compiler incompatible: production kernel build failed: " + log);
    }
    return program;
}
/* Read device string field; device is GPU handle, field is property enum; returns string or throws runtime error. */
static std::string deviceText(cl_device_id device, cl_device_info field) {
    // Required property size in bytes.
    std::size_t size = 0;
    checkCl(clGetDeviceInfo(device, field, 0, nullptr, &size), "query device string size");
    // Runtime-owned property copied into host memory.
    std::string value(size, '\0');
    checkCl(clGetDeviceInfo(device, field, size, value.data(), nullptr), "query device string");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}
/* Probe GPU device for float64, cached production compilation, every kernel entry and queue creation;
 * retains the shared runtime, releases probe resources, and throws specific unsupported/runtime diagnostics. */
static void probe(cl_device_id device) {
    // Double precision feature bitfield.
    cl_device_fp_config precision = 0;
    checkCl(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(precision), &precision, nullptr), "query fp64");
    // Availability flag from runtime.
    cl_bool available = CL_FALSE;
    checkCl(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr), "query availability");
    // Runtime source compiler availability.
    cl_bool compiler = CL_FALSE;
    checkCl(clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, nullptr), "query compiler");
    if (!available) throw std::runtime_error("GPU hardware reported unavailable by its driver/runtime");
    if (!compiler) throw std::runtime_error("GPU runtime has no OpenCL source compiler");
    if (!precision || deviceText(device, CL_DEVICE_EXTENSIONS).find("cl_khr_fp64") == std::string::npos)
        throw std::runtime_error("GPU unsupported operation: float64 with cl_khr_fp64 is required");
    if (deviceText(device, CL_DEVICE_VERSION).find("OpenCL 1.0") != std::string::npos ||
        deviceText(device, CL_DEVICE_VERSION).find("OpenCL 1.1") != std::string::npos)
        throw std::runtime_error("GPU runtime incompatible: OpenCL 1.2 or newer required");
    // Shared GPU-only context and compiled program, retained across discovery and model sessions.
    const auto runtime = runtimeFor(device);
    // Temporary queue/kernel creation status.
    cl_int status = CL_SUCCESS;
    // Queue used only for the capability probe; execution sessions own separate queues.
    cl_command_queue queue = nullptr;
    try {
        queue = clCreateCommandQueue(runtime->context(), device, 0, &status);
        checkCl(status, "probe queue");
        // All production entry points must be executable before publishing this device.
        const char* names[] = {"forwardLayer", "outputDelta", "hiddenDelta", "parameterGradient",
            "stageUpdate", "commitUpdate", "lossBlocks", "finishLoss"};
        // Name of the next entry point checked against the cached compiled program.
        for (const char* name : names) {
            // Probe-only kernel, released immediately before examining its creation status.
            cl_kernel kernel = clCreateKernel(runtime->program(), name, &status);
            if (kernel) clReleaseKernel(kernel);
            // Operation name preserves the missing or incompatible entry in diagnostics.
            const std::string operation = std::string("probe kernel ") + name;
            checkCl(status, operation.c_str());
        }
    } catch (...) {
        if (queue) clReleaseCommandQueue(queue);
        throw;
    }
    clReleaseCommandQueue(queue);
}
/* Discover GPU hardware, check full model support, sort deterministically; returns snapshot and diagnostic for every rejection. */
static Inventory discover() {
    // Discovery snapshot under construction.
    Inventory result;
    // Platform count exposed by the ICD loader.
    cl_uint count = 0;
    // First API status distinguishes missing platforms from other runtime errors.
    const cl_int status = clGetPlatformIDs(0, nullptr, &count);
    if (status == -1001 || (status == CL_SUCCESS && count == 0)) {
        result.diagnostic = "GPU runtime unavailable: OpenCL loader found no platforms; install/repair a vendor GPU ICD and driver";
        return result;
    }
    if (status != CL_SUCCESS) {
        result.diagnostic = "GPU runtime incompatible/unavailable: clGetPlatformIDs status " + std::to_string(status);
        return result;
    }
    // Runtime platforms in stable enumeration order.
    std::vector<cl_platform_id> platforms(count);
    checkCl(clGetPlatformIDs(count, platforms.data(), nullptr), "enumerate platforms");
    // Key, original runtime ordinal and GPU handle for deterministic ordering.
    std::vector<std::tuple<std::string, std::size_t, cl_device_id>> candidates;
    // Platform index used as final tie-breaker through insertion ordinal.
    for (const auto platform : platforms) {
        // GPU count on this platform.
        cl_uint deviceCount = 0;
        // GPU-only enumeration status; CPU OpenCL implementations are never selected.
        const cl_int enumerated = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (enumerated == CL_DEVICE_NOT_FOUND) continue;
        if (enumerated != CL_SUCCESS) {
            result.diagnostic += "GPU runtime enumeration error " + std::to_string(enumerated) + "; ";
            continue;
        }
        // GPU handles for this platform.
        std::vector<cl_device_id> devices(deviceCount);
        checkCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr), "enumerate GPUs");
        // GPU candidate handle being inspected.
        for (const auto device : devices) {
            // Sort key includes vendor/name/driver/version; runtime ordinal breaks identical-device ties.
            const std::string key = deviceText(device, CL_DEVICE_VENDOR) + "/" + deviceText(device, CL_DEVICE_NAME) +
                "/" + deviceText(device, CL_DRIVER_VERSION) + "/" + deviceText(device, CL_DEVICE_VERSION);
            candidates.emplace_back(key, candidates.size(), device);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    // Sorted candidate tuple being probed.
    for (const auto& candidate : candidates) {
        // Candidate GPU handle.
        const auto device = std::get<2>(candidate);
        try {
            probe(device);
            result.info.push_back({DeviceType::GPU, result.handles.size(), deviceText(device, CL_DEVICE_NAME), "opencl-parallel-float64", true, true});
            result.handles.push_back(device);
        } catch (const std::runtime_error& error /* unsupported GPU/runtime/build reason */) {
            result.diagnostic += std::get<0>(candidate) + ": " + error.what() + "; ";
        }
    }
    if (candidates.empty() && result.diagnostic.empty()) result.diagnostic = "GPU hardware missing: OpenCL platforms expose no GPU devices";
    if (!result.info.empty()) result.diagnostic += "GPU index invalid if outside executable inventory [0," + std::to_string(result.info.size()) + ")";
    return result;
}
/* Convert runtime discovery failures into an unavailable-GPU diagnostic so AUTO can still select CPU before execution. */
static Inventory safeDiscover() {
    try { return discover(); }
    catch (const std::runtime_error& error /* runtime inventory failure */) {
        // Empty inventory retains the concrete runtime failure for explicit GPU requests and AUTO skips.
        Inventory result;
        result.diagnostic = std::string("GPU runtime discovery failed: ") + error.what();
        return result;
    }
}
/* Return immutable discovery snapshot, initialized once per process; no numerical computation is performed. */
const Inventory& inventory() {
    // Thread-safe snapshot retains stable index-to-runtime mapping throughout the process.
    static const Inventory found = safeDiscover();
    return found;
}
/* Return executable GPU descriptors; resource and production build probes have already succeeded. */
std::vector<DeviceInfo> openClDevices() { return inventory().info; }
/* Return actionable diagnostic for unavailable, unsupported or out-of-range GPU request. */
std::string openClDiagnostic() { return inventory().diagnostic; }
}
