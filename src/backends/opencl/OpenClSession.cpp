#include "OpenClSession.h"
#include "OpenClRuntime.h"
#include "Inventory.h"
#include <limits>
#include <stdexcept>
namespace nn::detail {
/* Return left+right count; left/right are counts; throw runtime_error on overflow before allocation. */
static std::size_t sumSize(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) throw std::runtime_error("OpenCL workspace size overflow");
    return left + right;
}
/* Return count*width elements/bytes; count/width are dimensions; throw on overflow before allocation. */
static std::size_t product(std::size_t count, std::size_t width) {
    if (width && count > std::numeric_limits<std::size_t>::max() / width) throw std::runtime_error("OpenCL buffer size overflow");
    return count * width;
}
/* Bind kernel argument. kernel: entry point; index: ABI slot; value: typed argument; throws on API error. */
template<class T>
static void argument(cl_kernel kernel, cl_uint index, const T& value) {
    checkCl(clSetKernelArg(kernel, index, sizeof(T), &value), "set kernel argument");
}
/* Bind arguments and launch count independent work items on queue/kernel; args: ABI arguments; throws without retry. */
template<class... Args>
static void launch(cl_command_queue queue, cl_kernel kernel, std::size_t count, const Args&... args) {
    // Next argument position, evaluated left-to-right by comma fold.
    cl_uint index = 0;
    (argument(kernel, index++, args), ...);
    checkCl(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &count, nullptr, 0, nullptr, nullptr), "enqueue parallel GPU kernel");
}
/* Pack data sample-major into reusable result; preserves scalar values without numerical computation. */
static void flattenInto(const Dataset& data, std::vector<double>& result) {
    result.clear();
    if (!data.empty()) result.reserve(product(data.size(), data.front().size()));
    // Sample row copied into upload buffer.
    for (const auto& row : data) result.insert(result.end(), row.begin(), row.end());
}
/* Acquire runtime for device, create private queue and entry handles; throws with cleanup delegated to constructor. */
void OpenClSession::initialize(cl_device_id device) {
    runtime_ = runtimeFor(device);
    // Runtime creation status for queue and kernel handles.
    cl_int status = CL_SUCCESS;
    queue_ = clCreateCommandQueue(runtime_->context(), device, 0, &status);
    checkCl(status, "create GPU queue");
    // Entry names in the same order as the private Kernel enum.
    const char* names[]{"forwardLayer", "outputDelta", "hiddenDelta", "parameterGradient", "stageUpdate", "commitUpdate", "lossBlocks", "finishLoss"};
    // Kernel slot being initialized; every handle is private to this session.
    for (std::size_t index = 0; index < KernelCount; ++index) {
        kernels_[index] = clCreateKernel(runtime_->program(), names[index], &status);
        checkCl(status, "create parallel GPU kernel");
    }
}
/* Grow slot to bytes if needed, optionally copying data on allocation; returns owned buffer or throws preserving old slot. */
cl_mem OpenClSession::ensure(Buffer slot, std::size_t bytes, const void* data) {
    if (capacities_[slot] >= bytes) return buffers_[slot];
    // Runtime buffer allocation status.
    cl_int status = CL_SUCCESS;
    // Replacement allocated before old storage is released, retaining valid ownership on failure.
    cl_mem replacement = clCreateBuffer(runtime_->context(), CL_MEM_READ_WRITE | (data ? CL_MEM_COPY_HOST_PTR : 0),
        bytes, const_cast<void*>(data), &status);
    checkCl(status, "allocate GPU workspace");
    if (buffers_[slot]) clReleaseMemObject(buffers_[slot]);
    buffers_[slot] = replacement;
    capacities_[slot] = bytes;
    return replacement;
}
/* Upload latest canonical model to GPU index; layers defines architecture/parameters; throws after cleaning partial resources. */
OpenClSession::OpenClSession(std::size_t index, const std::vector<Dense>& layers) {
    static_assert(static_cast<int>(Activation::Linear) == 0 && static_cast<int>(Activation::Sigmoid) == 1 &&
        static_cast<int>(Activation::Tanh) == 2 && static_cast<int>(Activation::Relu) == 3, "OpenCL activation ABI changed");
    if (index >= inventory().handles.size()) throw std::runtime_error(openClDiagnostic());
    if (layers.empty()) throw std::invalid_argument("OpenCL requires a nonempty model");
    try {
        initialize(inventory().handles[index]);
        // Initial parameter vector uploaded once per public operation/session.
        std::vector<double> parameters;
        // Incoming activation boundary offset in feature-width units.
        std::size_t offset = 0;
        activationCount_ = layers.front().inputs;
        layers_.reserve(layers.size());
        // Dense layer packed into session metadata.
        for (const auto& layer : layers) {
            layers_.push_back({layer.inputs, layer.outputs, static_cast<cl_ulong>(layer.activation), parameters.size(), offset, activationCount_});
            offset = activationCount_;
            activationCount_ = sumSize(activationCount_, layer.outputs);
            parameters.insert(parameters.end(), layer.weights.begin(), layer.weights.end());
            parameters.insert(parameters.end(), layer.biases.begin(), layer.biases.end());
        }
        parameterCount_ = parameters.size();
        outputWidth_ = layers.back().outputs;
        ensure(Parameters, product(parameters.size(), sizeof(double)), parameters.data());
        ensure(Error, sizeof(cl_int));
    } catch (...) { release(); throw; }
}
/* Grow workspaces for samples/mode, retaining allocations between batches; throws on invalid size or allocation failure. */
void OpenClSession::prepare(std::size_t samples, int mode) {
    ensure(Values, product(product(samples, activationCount_), sizeof(double)));
    if (mode != 0) ensure(Targets, product(product(samples, outputWidth_), sizeof(double)));
    if (mode >= 2) {
        ensure(Deltas, product(product(samples, activationCount_), sizeof(double)));
        ensure(Derivatives, product(parameterCount_, sizeof(double)));
    }
    if (mode == 3) ensure(Candidate, product(parameterCount_, sizeof(double)));
    if (mode == 1) {
        // Number of loss blocks, computed without count+255 overflow.
        const std::size_t blocks = (product(samples, outputWidth_) - 1) / 256 + 1;
        ensure(Partials, product(product(blocks, 2), sizeof(double)));
        ensure(LossResult, sizeof(double));
    }
}
/* Execute batched forward layers for samples; all activation arithmetic stays on GPU. */
void OpenClSession::forward(cl_ulong samples) {
    // Layer metadata; boundary feature offsets scale by actual current batch size.
    for (const auto& layer : layers_)
        launch(queue_, kernels_[Forward], product(samples, layer[1]), buffers_[Parameters], buffers_[Values], buffers_[Error],
            samples, layer[0], layer[1], layer[3], layer[4] * samples, layer[5] * samples, layer[2]);
}
/* Compute all preactivation deltas and parameter gradients for samples; each scalar has a unique writer per launch. */
void OpenClSession::differentiate(cl_ulong samples) {
    // Last layer metadata initializes the final preactivation derivatives.
    const auto& last = layers_.back();
    launch(queue_, kernels_[OutputDelta], product(samples, last[1]), buffers_[Values], buffers_[Targets], buffers_[Deltas], buffers_[Error],
        samples, last[1], last[5] * samples, last[2]);
    // Reverse layer index; avoid unused derivatives with respect to original inputs.
    for (std::size_t index = layers_.size(); index-- > 0;) {
        // Current layer's widths and offsets.
        const auto& layer = layers_[index];
        launch(queue_, kernels_[Gradient], product(layer[1], sumSize(layer[0], 1)), buffers_[Values], buffers_[Deltas], buffers_[Derivatives], buffers_[Error],
            samples, layer[0], layer[1], layer[3], layer[4] * samples, layer[5] * samples);
        if (index != 0)
            launch(queue_, kernels_[HiddenDelta], product(samples, layer[0]), buffers_[Parameters], buffers_[Values], buffers_[Deltas], buffers_[Error],
                samples, layer[0], layer[1], layer[3], layer[4] * samples, layer[5] * samples, layers_[index - 1][2]);
    }
}
/* Execute inputs/targets with mode and rate on one GPU; return results, reuse buffers and propagate errors without fallback. */
std::vector<double> OpenClSession::run(const Dataset& inputs, const Dataset& targets, int mode, double rate) {
    if (mode < 0 || mode > 3) throw std::invalid_argument("Unknown OpenCL operation");
    prepare(inputs.size(), mode);
    flattenInto(inputs, input_);
    if (mode != 0) flattenInto(targets, target_);
    // Stable kernel sample-count ABI type.
    const cl_ulong samples = inputs.size();
    // Reset source stays unchanged until the asynchronous write and pipeline complete.
    const cl_int zero = 0;
    // Distinct readback storage avoids reusing an in-flight host upload pointer.
    cl_int error = 0;
    try {
        checkCl(clEnqueueWriteBuffer(queue_, buffers_[Error], CL_FALSE, 0, sizeof(zero), &zero, 0, nullptr, nullptr), "reset GPU error status");
        checkCl(clEnqueueWriteBuffer(queue_, buffers_[Values], CL_FALSE, 0, product(input_.size(), sizeof(double)), input_.data(), 0, nullptr, nullptr), "upload GPU inputs");
        if (mode != 0)
            checkCl(clEnqueueWriteBuffer(queue_, buffers_[Targets], CL_FALSE, 0, product(target_.size(), sizeof(double)), target_.data(), 0, nullptr, nullptr), "upload GPU labels");
        forward(samples);
        // GPU buffer holding final requested result.
        cl_mem output = buffers_[Values];
        // Output offset for prediction skips the input/hidden activation matrices.
        std::size_t offset = product(product(layers_.back()[5], inputs.size()), sizeof(double));
        // Number of returned scalar outputs.
        std::size_t count = product(inputs.size(), outputWidth_);
        if (mode == 1) {
            // Parallel reduction count in fixed-width kernel ABI.
            const cl_ulong blocks = (count - 1) / 256 + 1;
            // Scalar residual count in fixed-width kernel ABI.
            const cl_ulong residuals = count;
            launch(queue_, kernels_[LossBlocks], blocks, buffers_[Values], buffers_[Targets], buffers_[Partials], buffers_[Error], residuals, layers_.back()[5] * samples);
            launch(queue_, kernels_[FinishLoss], 1, buffers_[Partials], buffers_[LossResult], buffers_[Error], blocks, residuals);
            output = buffers_[LossResult]; offset = 0; count = 1;
        }
        if (mode >= 2) {
            differentiate(samples);
            output = buffers_[Derivatives]; offset = 0; count = parameterCount_;
        }
        if (mode == 3) {
            launch(queue_, kernels_[Stage], parameterCount_, buffers_[Parameters], buffers_[Derivatives], buffers_[Candidate], buffers_[Error], parameterCount_, rate);
            launch(queue_, kernels_[Commit], parameterCount_, buffers_[Parameters], buffers_[Candidate], buffers_[Error], parameterCount_);
            output = buffers_[Parameters];
        }
        checkCl(clEnqueueReadBuffer(queue_, buffers_[Error], CL_TRUE, 0, sizeof(error), &error, 0, nullptr, nullptr), "read GPU numerical status");
        if (error) throw std::runtime_error("Nonfinite GPU arithmetic: operation rejected without device fallback");
        // Caller-owned result copied after every kernel and validation completes.
        std::vector<double> result(count);
        checkCl(clEnqueueReadBuffer(queue_, output, CL_TRUE, offset, product(count, sizeof(double)), result.data(), 0, nullptr, nullptr), "read GPU results");
        return result;
    } catch (...) {
        // Ensure queued transfers no longer reference local host storage before unwinding.
        if (queue_) clFinish(queue_);
        throw;
    }
}
/* Release buffers, entry handles and queue; cached immutable context/program live through shared runtime ownership. */
void OpenClSession::release() noexcept {
    if (queue_) clFinish(queue_);
    // Private device memory handle, possibly absent after partial initialization.
    for (const auto memory : buffers_) if (memory) clReleaseMemObject(memory);
    // Private entry point handle.
    for (const auto kernel : kernels_) if (kernel) clReleaseKernel(kernel);
    if (queue_) clReleaseCommandQueue(queue_);
}
/* Destroy operation/session resources without throwing. */
OpenClSession::~OpenClSession() { release(); }
}
