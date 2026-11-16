#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#include <memory>
namespace nn::detail {
class OpenClRuntime {
public:
    /* Create one GPU-only context and compile its production program. device: GPU handle;
     * owns both resources on success; throws runtime_error on device/context/compiler failure. */
    explicit OpenClRuntime(cl_device_id device);
    /* Release the compiled program and its context; has no parameters and never throws. */
    ~OpenClRuntime() noexcept;
    /* Prevent copying resource ownership. other: runtime that cannot be copied; no side effects or exceptions. */
    OpenClRuntime(const OpenClRuntime& other) = delete;
    /* Prevent assigning resource ownership. other: runtime that cannot be assigned; no side effects or exceptions. */
    OpenClRuntime& operator=(const OpenClRuntime& other) = delete;
    /* Return the borrowed GPU-only context; no parameters, ownership transfer, side effects or exceptions. */
    cl_context context() const noexcept;
    /* Return the borrowed immutable compiled program; no parameters, ownership transfer, side effects or exceptions. */
    cl_program program() const noexcept;
private:
    // Shared GPU-only context, released after the compiled program.
    cl_context context_ = nullptr;
    // Immutable production program; sessions create their own kernels and queues from it.
    cl_program program_ = nullptr;
};
/* Return the process-lifetime shared runtime for device, a GPU handle. Serializes first construction
 * and caches its context/program; throws runtime_error for runtime/compiler errors or allocation exceptions. */
std::shared_ptr<OpenClRuntime> runtimeFor(cl_device_id device);
}
