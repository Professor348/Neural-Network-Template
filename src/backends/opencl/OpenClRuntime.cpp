#include "OpenClRuntime.h"
#include "Inventory.h"
#include <map>
#include <mutex>
#include <stdexcept>
namespace nn::detail {
/* Create the GPU-only context and compile production kernels for device; retain resources on success,
 * release partial construction on failure, and throw runtime_error for unsupported devices or OpenCL errors. */
OpenClRuntime::OpenClRuntime(cl_device_id device) {
    // Runtime-reported type prevents CPU devices from entering the shared GPU cache.
    cl_device_type type = 0;
    checkCl(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr), "query runtime device type");
    if (!(type & CL_DEVICE_TYPE_GPU)) throw std::runtime_error("OpenCL runtime requires a GPU device");
    // Context creation status, checked before starting program compilation.
    cl_int status = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    checkCl(status, "create shared GPU context");
    try {
        program_ = buildProgram(context_, device);
    } catch (...) {
        clReleaseContext(context_);
        context_ = nullptr;
        throw;
    }
}
/* Release owned program before context; no parameters or exceptions, including during process shutdown. */
OpenClRuntime::~OpenClRuntime() noexcept {
    if (program_) clReleaseProgram(program_);
    if (context_) clReleaseContext(context_);
}
/* Return the borrowed GPU context without transferring ownership; no parameters, mutation or exceptions. */
cl_context OpenClRuntime::context() const noexcept { return context_; }
/* Return the borrowed compiled program without transferring ownership; no parameters, mutation or exceptions. */
cl_program OpenClRuntime::program() const noexcept { return program_; }
/* Return a retained runtime for GPU device; lock cache construction so successful devices compile once.
 * Mutates only the process cache, leaves failed construction uncached, and propagates runtime/allocation exceptions. */
std::shared_ptr<OpenClRuntime> runtimeFor(cl_device_id device) {
    // Mutex serializes cache lookup and first compilation, including concurrent first use.
    static std::mutex mutex;
    // Strong references preserve successful contexts/programs until process shutdown.
    static std::map<cl_device_id, std::shared_ptr<OpenClRuntime>> runtimes;
    // Scope-bound ownership releases the cache mutex on success and every exception path.
    const std::lock_guard<std::mutex> lock(mutex);
    // Existing immutable runtime, if this device was already probed or used by another session.
    const auto found = runtimes.find(device);
    if (found != runtimes.end()) return found->second;
    // First successful construction is shared by discovery and every later GPU session.
    auto runtime = std::make_shared<OpenClRuntime>(device);
    runtimes.emplace(device, runtime);
    return runtime;
}
}
