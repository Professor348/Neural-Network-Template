#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <nn/Dense.h>
#include <nn/DeviceInfo.h>
#include <array>
#include <memory>
namespace nn::detail {
class OpenClRuntime;
/* Return successfully initialized GPU descriptors, in deterministic runtime ordering; never includes CPU devices. */
std::vector<DeviceInfo> openClDevices();
/* Describe disabled/missing runtime/hardware/precision/build failure, or invalid GPU index; no numerical execution. */
std::string openClDiagnostic();
class OpenClSession {
public:
    /* Construct GPU operation session. index: discovery index; layers: validated model;
     * reuses cached immutable runtime, owns mutable queue/weights/workspace; throws runtime_error on failure. */
    OpenClSession(std::size_t index, const std::vector<Dense>& layers);
    /* Release private GPU queue, kernels and buffers; shared runtime remains cached; no exceptions. */
    ~OpenClSession();
    /* Prevent resource ownership duplication. other: session that cannot be copied. */
    OpenClSession(const OpenClSession& other) = delete;
    /* Prevent resource reassignment. other: session that cannot be copied; no output. */
    OpenClSession& operator=(const OpenClSession& other) = delete;
    /* Run complete operation on selected GPU. inputs/targets: validated matrices (empty targets for mode 0);
     * mode: 0 prediction, 1 MSE, 2 gradient, 3 SGD; rate: positive SGD step for mode 3.
     * Returns flattened GPU results; reuses capacities between batches; throws without retry/fallback.
     * Only a completely validated SGD proposal mutates parameters. */
    std::vector<double> run(const Dataset& inputs, const Dataset& targets, int mode, double rate = 0);
private:
    // Kernel entry slots, matching the names in initialize().
    enum Kernel { Forward, OutputDelta, HiddenDelta, Gradient, Stage, Commit, LossBlocks, FinishLoss, KernelCount };
    // Private buffer slots; capacities are measured in bytes.
    enum Buffer { Parameters, Values, Targets, Deltas, Derivatives, Candidate, Error, Partials, LossResult, BufferCount };
    /* Bind cached runtime for device and create a private queue/kernel handles; throws on runtime failure. */
    void initialize(cl_device_id device);
    /* Ensure slot has at least bytes capacity; copies data only on new allocation if supplied.
     * Returns owned buffer; allocation errors preserve existing buffer and throw runtime_error. */
    cl_mem ensure(Buffer slot, std::size_t bytes, const void* data = nullptr);
    /* Prepare reusable data/workspaces. samples: batch size; mode: operation kind; throws on overflow/allocation error. */
    void prepare(std::size_t samples, int mode);
    /* Launch all forward layers for samples; values contain uploaded input row matrix; flags nonfinite arithmetic on GPU. */
    void forward(cl_ulong samples);
    /* Launch complete reverse pass for samples using resident targets; writes canonical gradient array on GPU. */
    void differentiate(cl_ulong samples);
    /* Release all private runtime resources in dependency order; safe after incomplete construction; no exceptions. */
    void release() noexcept;
    // Shared immutable GPU context and compiled program, cached per physical device.
    std::shared_ptr<OpenClRuntime> runtime_;
    // Session-private in-order queue; kernel boundaries synchronize dependent stages.
    cl_command_queue queue_ = nullptr;
    // Private kernel handles allow argument binding without cross-session races.
    std::array<cl_kernel, KernelCount> kernels_{};
    // Session-owned buffers, retained and grown across minibatches.
    std::array<cl_mem, BufferCount> buffers_{};
    // Allocated byte capacity for each buffer slot.
    std::array<std::size_t, BufferCount> capacities_{};
    // Host metadata per layer: input/output width, activation, parameter/input/output offsets.
    std::vector<std::array<cl_ulong, 6>> layers_;
    // Reusable packed upload storage for sample-major input data.
    std::vector<double> input_;
    // Reusable packed upload storage for labels.
    std::vector<double> target_;
    // Total scalar parameter count.
    cl_ulong parameterCount_ = 0;
    // Sum of all boundary feature widths; batch activation size multiplies this by samples.
    cl_ulong activationCount_ = 0;
    // Final output feature width.
    std::size_t outputWidth_ = 0;
};
}
