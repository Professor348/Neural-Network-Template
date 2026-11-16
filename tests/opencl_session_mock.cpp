#include "backends/opencl/OpenClSession.h"
#include "backends/opencl/Inventory.h"
#include "backends/Cpu.h"
#include "opencl_mock/BoundArgument.h"
#include "opencl_mock/Memory.h"
#include "opencl_mock/Kernel.h"
#include "opencl_mock/Transfer.h"
#include "opencl_mock/Upload.h"
#include "opencl_mock/Device.h"
#include "opencl_mock/Context.h"
#include "opencl_mock/Program.h"
#include "opencl_mock/Queue.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// This test executes the real session orchestration with fake OpenCL handles.
// Exact production kernels run serially on the host: this is NOT physical GPU execution.
namespace host {
using std::exp;
using std::tanh;
using std::fabs;
using std::sqrt;
using std::isfinite;
// Portable kernel ulong ABI and currently emulated global coordinate.
using index_type = std::uint64_t;
// Work item selected by the mocked enqueue operation.
static std::size_t workItem = 0;
/* Return host-emulated coordinate; dimension is unused for one-dimensional kernels. */
std::size_t get_global_id(unsigned /* dimension */) { return workItem; }
/* Serial host equivalent of atomic OR; returns the previous error bits. */
int atomic_or(volatile int* address, int value) {
    // Previously recorded numerical error.
    const int previous = *address;
    *address |= value;
    return previous;
}
#define ulong index_type
#define __global
#define __kernel
#include "../src/backends/opencl/kernels.cl"
#undef __kernel
#undef __global
#undef ulong
}

namespace mock {
// One intentionally synthetic device handle, never reported as discovered physical hardware.
static _cl_device_id device;
// Currently live resources detect stale handles and destruction omissions.
static std::set<cl_mem> memories;
// Currently live private kernel handles.
static std::set<cl_kernel> kernels;
// Currently live private command queues.
static std::set<cl_command_queue> queues;
// Number of successful memory allocations, used to prove workspace reuse.
static std::size_t allocations = 0;
// Number of constructed immutable runtime contexts.
static std::size_t contexts = 0;
// Number of compiled mock programs.
static std::size_t programs = 0;
// Number of completed kernel enqueues by production entry point.
static std::map<std::string, std::size_t> launches;
// Last public result readback, excluding the integer numerical status.
static Transfer resultRead{0, 0};
// In-order queued uploads, flushed at the next kernel/read/finish boundary.
static std::vector<Upload> uploads;

/* Require condition with diagnostic message; throws to fail the standalone integration test. */
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
/* Return correctly aligned mock backing storage for memory. */
void* data(cl_mem memory) {
    return memory->bytes == sizeof(cl_int) ? static_cast<void*>(&memory->error) : memory->values.data();
}
/* Validate byte offset/count against a live buffer, including overflow-safe range checking. */
void bounds(cl_mem memory, std::size_t offset, std::size_t bytes) {
    require(memories.count(memory) != 0, "Unknown or released OpenCL memory handle");
    require(offset <= memory->bytes && bytes <= memory->bytes - offset, "OpenCL buffer range exceeded");
}
/* Execute pending nonblocking uploads in order and check their host sources were kept unchanged. */
void flush() {
    // Upload whose host pointer remains borrowed until this command boundary.
    for (const auto& upload : uploads) {
        require(std::memcmp(upload.source, upload.snapshot.data(), upload.snapshot.size()) == 0,
            "Nonblocking upload source changed before completion");
        std::memcpy(static_cast<unsigned char*>(data(upload.memory)) + upload.offset,
            upload.source, upload.snapshot.size());
    }
    uploads.clear();
}
/* Return expected typed argument ABI for the named production kernel; m=buffer, u=cl_ulong, d=double. */
std::string signature(const std::string& name) {
    if (name == "forwardLayer") return "mmmuuuuuuu";
    if (name == "outputDelta") return "mmmmuuuu";
    if (name == "hiddenDelta") return "mmmmuuuuuuu";
    if (name == "parameterGradient") return "mmmmuuuuuu";
    if (name == "stageUpdate") return "mmmmud";
    if (name == "commitUpdate") return "mmmu";
    if (name == "lossBlocks") return "mmmmuu";
    if (name == "finishLoss") return "mmmuu";
    throw std::runtime_error("Unknown production kernel name");
}
/* Decode a copied kernel ABI argument at index; rejects wrong size or missing binding. */
template<class T>
T argument(cl_kernel kernel, std::size_t index) {
    require(index < kernel->arguments.size() && kernel->arguments[index].bytes.size() == sizeof(T),
        "Missing or incorrectly typed OpenCL kernel argument");
    // Typed value reconstructed without unaligned access.
    T value{};
    std::memcpy(&value, kernel->arguments[index].bytes.data(), sizeof(value));
    return value;
}
/* Return a double buffer argument after proving the kernel's complete indexed range fits. */
double* buffer(cl_kernel kernel, std::size_t index, std::size_t elements) {
    // Buffer handle passed through the real session binding helper.
    const cl_mem memory = argument<cl_mem>(kernel, index);
    require(elements <= std::numeric_limits<std::size_t>::max() / sizeof(double), "Kernel extent overflow");
    bounds(memory, 0, elements * sizeof(double));
    require(memory->bytes != sizeof(cl_int), "Integer buffer bound as double workspace");
    return memory->values.data();
}
/* Return the checked shared numerical error buffer bound at index. */
cl_int* error(cl_kernel kernel, std::size_t index) {
    // Numerical status buffer handle in the kernel ABI.
    const cl_mem memory = argument<cl_mem>(kernel, index);
    bounds(memory, 0, sizeof(cl_int));
    require(memory->bytes == sizeof(cl_int), "Numerical status bound to wrong buffer");
    return &memory->error;
}
/* Launch one production kernel on host after validating dimensions, typed bindings and every indexed buffer extent. */
void execute(cl_kernel kernel, std::size_t count) {
    // Kernel's metadata scalars after its initial buffer arguments.
    const std::size_t firstScalar = signature(kernel->name).find('u');
    // Kernel dimensions/offsets in original positional order.
    std::vector<cl_ulong> u;
    // ABI slot scanned for unsigned metadata scalars.
    for (std::size_t index = firstScalar; index < kernel->arguments.size(); ++index) {
        if (signature(kernel->name)[index] == 'u') u.push_back(argument<cl_ulong>(kernel, index));
    }
    if (kernel->name == "forwardLayer" || kernel->name == "hiddenDelta") {
        // Input sample count, input/output widths and parameter/activation offsets.
        const auto n = u[0], in = u[1], out = u[2], weights = u[3], incoming = u[4], outgoing = u[5], kind = u[6];
        require(count == n * (kernel->name == "forwardLayer" ? out : in), "Wrong forward/hidden launch extent");
        // Bound canonical model and activation storage with all source/destination regions checked.
        double* parameters = buffer(kernel, 0, weights + in * out + out);
        // Activations must include both participating boundaries.
        double* values = buffer(kernel, 1, std::max(incoming + n * in, outgoing + n * out));
        if (kernel->name == "forwardLayer") {
            // Forward numerical error flag.
            cl_int* failure = error(kernel, 2);
            for (host::workItem = 0; host::workItem < count; ++host::workItem)
                host::forwardLayer(parameters, values, failure, n, in, out, weights, incoming, outgoing, kind);
        } else {
            // Backpropagation destinations and upstream source both fit this delta buffer.
            double* deltas = buffer(kernel, 2, std::max(incoming + n * in, outgoing + n * out));
            // Hidden numerical error flag.
            cl_int* failure = error(kernel, 3);
            for (host::workItem = 0; host::workItem < count; ++host::workItem)
                host::hiddenDelta(parameters, values, deltas, failure, n, in, out, weights, incoming, outgoing, kind);
        }
    } else if (kernel->name == "outputDelta") {
        // Final boundary shape, activation offset and activation kind.
        const auto n = u[0], out = u[1], outgoing = u[2], kind = u[3];
        require(count == n * out, "Wrong output delta launch extent");
        // Prediction boundary, complete target matrix and final delta boundary.
        double* values = buffer(kernel, 0, outgoing + count);
        // Uploaded labels in sample-major order.
        double* targets = buffer(kernel, 1, count);
        // Final layer derivatives.
        double* deltas = buffer(kernel, 2, outgoing + count);
        // Output numerical status.
        cl_int* failure = error(kernel, 3);
        for (host::workItem = 0; host::workItem < count; ++host::workItem)
            host::outputDelta(values, targets, deltas, failure, n, out, outgoing, kind);
    } else if (kernel->name == "parameterGradient") {
        // Batch shape and canonical gradient/input/output offsets.
        const auto n = u[0], in = u[1], out = u[2], weights = u[3], incoming = u[4], outgoing = u[5];
        require(count == out * (in + 1), "Wrong parameter gradient launch extent");
        // Input activation matrix used in weight derivatives.
        double* values = buffer(kernel, 0, incoming + n * in);
        // Output preactivation derivative matrix.
        double* deltas = buffer(kernel, 1, outgoing + n * out);
        // Canonical model-order gradient destination.
        double* gradient = buffer(kernel, 2, weights + count);
        // Gradient numerical status.
        cl_int* failure = error(kernel, 3);
        for (host::workItem = 0; host::workItem < count; ++host::workItem)
            host::parameterGradient(values, deltas, gradient, failure, n, in, out, weights, incoming, outgoing);
    } else if (kernel->name == "stageUpdate") {
        require(count == u[0], "Wrong staged update launch extent");
        // Current model, derivatives and isolated candidate buffers.
        double* parameters = buffer(kernel, 0, count);
        // Canonical mean-loss gradients.
        double* gradient = buffer(kernel, 1, count);
        // Staged proposal destination.
        double* candidate = buffer(kernel, 2, count);
        // Shared failure flag checked only by the following commit launch.
        cl_int* failure = error(kernel, 3);
        require(parameters != candidate && parameters != gradient && gradient != candidate, "SGD workspaces alias");
        for (host::workItem = 0; host::workItem < count; ++host::workItem)
            host::stageUpdate(parameters, gradient, candidate, failure, u[0], argument<double>(kernel, 5));
    } else if (kernel->name == "commitUpdate") {
        require(count == u[0], "Wrong commit launch extent");
        // Canonical model destination.
        double* parameters = buffer(kernel, 0, count);
        // Completed and validated proposal from the preceding enqueue.
        double* candidate = buffer(kernel, 1, count);
        // Stable failure flag after all proposal work items complete.
        cl_int* failure = error(kernel, 2);
        for (host::workItem = 0; host::workItem < count; ++host::workItem)
            host::commitUpdate(parameters, candidate, failure, u[0]);
    } else if (kernel->name == "lossBlocks") {
        require(count == (u[0] - 1) / 256 + 1, "Wrong loss block launch extent");
        // Final predictions including their layer boundary offset.
        double* values = buffer(kernel, 0, u[1] + u[0]);
        // Full uploaded target matrix.
        double* targets = buffer(kernel, 1, u[0]);
        // Two scaled-reduction scalars per block.
        double* partials = buffer(kernel, 2, count * 2);
        // Loss numerical status.
        cl_int* failure = error(kernel, 3);
        for (host::workItem = 0; host::workItem < count; ++host::workItem)
            host::lossBlocks(values, targets, partials, failure, u[0], u[1]);
    } else if (kernel->name == "finishLoss") {
        require(count == 1, "Wrong final loss launch extent");
        // Completed scaled loss blocks.
        double* partials = buffer(kernel, 0, u[0] * 2);
        // Scalar mean-loss destination.
        double* result = buffer(kernel, 1, 1);
        // Final loss numerical status.
        cl_int* failure = error(kernel, 2);
        host::finishLoss(partials, result, failure, u[0], u[1]);
    }
    ++launches[kernel->name];
}
}

extern "C" {
/* Advertise only the synthetic GPU type needed by the real runtime constructor. */
CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info name,
    std::size_t bytes, void* value, std::size_t* actual) {
    mock::require(device == &mock::device && name == CL_DEVICE_TYPE && bytes == sizeof(cl_device_type), "Unexpected device query");
    // Synthetic type enables production GPU-only runtime validation without hardware.
    const cl_device_type type = CL_DEVICE_TYPE_GPU;
    std::memcpy(value, &type, sizeof(type));
    if (actual) *actual = sizeof(type);
    return CL_SUCCESS;
}
/* Construct a mock context while checking the real runtime requests exactly its selected device. */
CL_API_ENTRY cl_context CL_API_CALL clCreateContext(const cl_context_properties*, cl_uint count,
    const cl_device_id* devices, void (CL_CALLBACK*)(const char*, const void*, std::size_t, void*), void*, cl_int* status) {
    mock::require(count == 1 && devices[0] == &mock::device, "Unexpected context device");
    ++mock::contexts;
    *status = CL_SUCCESS;
    return new _cl_context;
}
/* Release the mock context owned by the real shared runtime. */
CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context context) { delete context; return CL_SUCCESS; }
/* Release the mock program owned by the real shared runtime. */
CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program) { delete program; return CL_SUCCESS; }
/* Create one private in-order command queue for the selected runtime. */
CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context, cl_device_id device,
    cl_command_queue_properties properties, cl_int* status) {
    mock::require(device == &mock::device && properties == 0, "Session requires a private in-order queue");
    // New queue handle tracked until release.
    const auto queue = new _cl_command_queue;
    mock::queues.insert(queue);
    *status = CL_SUCCESS;
    return queue;
}
/* Finish delayed transfers and release the private queue. */
CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue queue) {
    mock::flush();
    mock::require(mock::queues.erase(queue) == 1, "Unknown queue release");
    delete queue;
    return CL_SUCCESS;
}
/* Construct a private production kernel handle and verify the entry name. */
CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program, const char* name, cl_int* status) {
    // Kernel ABI determines the exact number of bound slots required for each launch.
    const auto abi = mock::signature(name);
    // New kernel handle with initially unbound arguments.
    const auto kernel = new _cl_kernel{name, std::vector<BoundArgument>(abi.size())};
    mock::kernels.insert(kernel);
    *status = CL_SUCCESS;
    return kernel;
}
/* Release one private kernel handle and reject duplicate destruction. */
CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
    mock::require(mock::kernels.erase(kernel) == 1, "Unknown kernel release");
    delete kernel;
    return CL_SUCCESS;
}
/* Allocate checked host backing storage; preserve the exact initial model upload. */
CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context, cl_mem_flags flags, std::size_t bytes, void* source, cl_int* status) {
    mock::require(bytes != 0 && (flags & CL_MEM_READ_WRITE), "Invalid workspace allocation");
    // Newly allocated buffer filled with NaNs so missing uploads are visible numerically.
    const auto memory = new _cl_mem{bytes, std::vector<double>((bytes + sizeof(double) - 1) / sizeof(double),
        std::numeric_limits<double>::quiet_NaN()), 0};
    mock::memories.insert(memory);
    if (flags & CL_MEM_COPY_HOST_PTR) {
        mock::require(source != nullptr, "Missing initial upload pointer");
        std::memcpy(mock::data(memory), source, bytes);
    }
    ++mock::allocations;
    *status = CL_SUCCESS;
    return memory;
}
/* Release a live workspace after all borrowed transfers have completed. */
CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem memory) {
    mock::require(mock::uploads.empty(), "Workspace freed with in-flight upload");
    mock::require(mock::memories.erase(memory) == 1, "Unknown memory release");
    delete memory;
    return CL_SUCCESS;
}
/* Queue nonblocking uploads with their original host source pointers and byte ranges. */
CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue queue, cl_mem memory, cl_bool blocking,
    std::size_t offset, std::size_t bytes, const void* source, cl_uint waits, const cl_event*, cl_event*) {
    mock::require(mock::queues.count(queue) && blocking == CL_FALSE && waits == 0, "Unexpected upload queue/ordering");
    mock::bounds(memory, offset, bytes);
    // Snapshot records exact bytes that must survive until the next execution boundary.
    std::vector<unsigned char> snapshot(bytes);
    std::memcpy(snapshot.data(), source, bytes);
    mock::uploads.push_back({memory, offset, source, std::move(snapshot)});
    return CL_SUCCESS;
}
/* Read only valid buffer ranges after completing all preceding in-order commands. */
CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue queue, cl_mem memory, cl_bool blocking,
    std::size_t offset, std::size_t bytes, void* destination, cl_uint waits, const cl_event*, cl_event*) {
    mock::require(mock::queues.count(queue) && blocking == CL_TRUE && waits == 0, "Unexpected readback queue/ordering");
    mock::flush();
    mock::bounds(memory, offset, bytes);
    std::memcpy(destination, static_cast<unsigned char*>(mock::data(memory)) + offset, bytes);
    if (memory->bytes != sizeof(cl_int)) mock::resultRead = {offset, bytes};
    return CL_SUCCESS;
}
/* Copy each bound argument and enforce its production fixed-width ABI before launch. */
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint index, std::size_t bytes, const void* value) {
    mock::require(mock::kernels.count(kernel) && index < kernel->arguments.size(), "Invalid kernel argument index");
    // Expected argument category in the named kernel's production signature.
    const char kind = mock::signature(kernel->name)[index];
    // Exact ABI size; size_t/int substitutions fail on platforms with differing widths.
    const std::size_t expected = kind == 'm' ? sizeof(cl_mem) : kind == 'u' ? sizeof(cl_ulong) : sizeof(double);
    mock::require(bytes == expected && value != nullptr, "Wrong kernel argument ABI size");
    kernel->arguments[index].bytes.resize(bytes);
    std::memcpy(kernel->arguments[index].bytes.data(), value, bytes);
    return CL_SUCCESS;
}
/* Execute the exact production kernel bodies serially on host after ABI/range checks. */
CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel,
    cl_uint dimensions, const std::size_t* offset, const std::size_t* global, const std::size_t* local,
    cl_uint waits, const cl_event*, cl_event*) {
    mock::require(mock::queues.count(queue) && mock::kernels.count(kernel), "Invalid kernel enqueue handles");
    mock::require(dimensions == 1 && !offset && !local && waits == 0 && global && *global,
        "Unexpected NDRange launch geometry");
    mock::flush();
    mock::execute(kernel, *global);
    return CL_SUCCESS;
}
/* Complete all outstanding host-pointer borrows at the production cleanup boundary. */
CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue queue) {
    mock::require(mock::queues.count(queue) != 0, "Unknown queue finish");
    mock::flush();
    return CL_SUCCESS;
}
}

namespace nn::detail {
/* Supply an explicitly synthetic discovery snapshot only to this isolated executable. */
const Inventory& inventory() {
    // Stable synthetic device inventory; real Discovery.cpp is deliberately not linked.
    static const Inventory result{{&mock::device}, {}, "Host-only OpenCL session mock"};
    return result;
}
/* Supply the constructor's invalid-index diagnostic without hardware discovery. */
std::string openClDiagnostic() { return "Host-only OpenCL session mock"; }
/* Retain production status-check semantics for mocked entrypoints. */
void checkCl(cl_int code, const char* operation) {
    if (code != CL_SUCCESS) throw std::runtime_error(operation);
}
/* Stand in for compiler setup; numerical bodies above include kernels.cl directly. */
cl_program buildProgram(cl_context, cl_device_id) {
    ++mock::programs;
    return new _cl_program;
}
}

/* Compare finite vectors with mixed absolute/relative tolerance against independent CPU calculations. */
void close(const std::vector<double>& actual, const std::vector<double>& expected) {
    mock::require(actual.size() == expected.size(), "Session result shape mismatch");
    // Canonical result coordinate under comparison.
    for (std::size_t index = 0; index < actual.size(); ++index)
        mock::require(std::isfinite(actual[index]) && std::isfinite(expected[index]) &&
            std::abs(actual[index] - expected[index]) <= 2e-12 * std::max(1.0, std::abs(expected[index])),
            "Session result differs from CPU reference");
}
/* Flatten matrix rows for comparison with the real session's public result representation. */
std::vector<double> flatten(const nn::Dataset& rows) {
    // Packed sample-major values.
    std::vector<double> result;
    // Matrix row appended without numerical conversion.
    for (const auto& row : rows) result.insert(result.end(), row.begin(), row.end());
    return result;
}
/* Return canonical weights-then-biases parameter storage for the CPU reference layers. */
std::vector<double> parameters(const std::vector<nn::Dense>& layers) {
    // Canonical model vector.
    std::vector<double> result;
    // Layer parameters appended in the same public order.
    for (const auto& layer : layers) {
        result.insert(result.end(), layer.weights.begin(), layer.weights.end());
        result.insert(result.end(), layer.biases.begin(), layer.biases.end());
    }
    return result;
}
/* Apply independent CPU SGD using gradient in canonical order. */
void update(std::vector<nn::Dense>& layers, const std::vector<double>& gradient, double rate) {
    // Canonical parameter coordinate shared across layers.
    std::size_t index = 0;
    // Layer updated independently from resident session buffers.
    for (auto& layer : layers) {
        // Weight scalar updated with its matching derivative.
        for (double& weight : layer.weights) weight -= rate * gradient[index++];
        // Bias scalar updated with its matching derivative.
        for (double& bias : layer.biases) bias -= rate * gradient[index++];
    }
}
/* Exercise real session calls over changing batches, modes and repeated resident SGD. */
void checkSession(nn::Activation activation) {
    // Uneven two-layer CPU reference; deterministic values exercise nonzero offsets and every derivative.
    std::vector<nn::Dense> layers{{3, 5, activation, std::vector<double>(15), {0.1, -0.2, 0.3, -0.4, 0.5}},
        {5, 2, nn::Activation::Sigmoid, std::vector<double>(10), {-0.3, 0.2}}};
    // Initial weight coordinate across both layers.
    std::size_t coordinate = 0;
    // Dense layer receiving deterministic bounded weights.
    for (auto& layer : layers) {
        // Individual weight in output-major order.
        for (double& weight : layer.weights) weight = 0.2 * std::sin(static_cast<double>(++coordinate));
    }
    // Real production session with private resources and shared real runtime cache.
    nn::detail::OpenClSession session(0, layers);
    // Allocation high-water mark after warming all modes for the largest batch.
    std::size_t warmAllocations = 0;
    // Growing then shrinking batches force offsets to depend on current samples, never buffer capacities.
    for (const std::size_t samples : {1u, 17u, 257u, 3u, 257u, 1u}) {
        // Input matrix with changed values on every batch size.
        nn::Dataset inputs(samples, std::vector<double>(3));
        // Two-output labels cross a loss reduction block boundary at 257 samples.
        nn::Dataset targets(samples, std::vector<double>(2));
        // Sample coordinate for deterministic dataset construction.
        for (std::size_t sample = 0; sample < samples; ++sample) {
            // Input feature coordinate.
            for (std::size_t feature = 0; feature < 3; ++feature)
                inputs[sample][feature] = std::sin(static_cast<double>(sample * 3 + feature + samples));
            targets[sample] = {0.15 + 0.01 * (sample % 7), 0.75 - 0.02 * (sample % 5)};
        }
        close(session.run(inputs, {}, 0), flatten(nn::detail::predictBatch(layers, inputs)));
        mock::require(mock::resultRead.offset == samples * 8 * sizeof(double) &&
            mock::resultRead.bytes == samples * 2 * sizeof(double), "Prediction readback boundary offset/size mismatch");
        close(session.run(inputs, targets, 1), {nn::detail::meanLoss(layers, inputs, targets)});
        // Independent CPU gradient before the next SGD update.
        const auto gradient = nn::detail::batchGradient(layers, inputs, targets);
        close(session.run(inputs, targets, 2), gradient);
        update(layers, gradient, 0.04);
        close(session.run(inputs, targets, 3, 0.04), parameters(layers));
        close(session.run(inputs, {}, 0), flatten(nn::detail::predictBatch(layers, inputs)));
        close(session.run(inputs, targets, 1), {nn::detail::meanLoss(layers, inputs, targets)});
        close(session.run(inputs, targets, 2), nn::detail::batchGradient(layers, inputs, targets));
        if (!warmAllocations && samples == 257) warmAllocations = mock::allocations;
        else if (warmAllocations) mock::require(mock::allocations == warmAllocations, "Workspace reallocated after smaller/equal batch");
    }
}
/* Force a late nonfinite SGD candidate and prove no earlier finite candidate was committed. */
void checkRejectedUpdate() {
    // First bias has a finite candidate; the last bias overflows only during the proposed update.
    const std::vector<nn::Dense> layers{{1, 2, nn::Activation::Linear, {0, 0}, {0, 0}}};
    // Production resident session whose parameter vector must remain intact after rejection.
    nn::detail::OpenClSession session(0, layers);
    // Zero input leaves weights unchanged while final bias derivative is finite and nonzero.
    const nn::Dataset inputs{{0}};
    // First bias proposes a finite changed value, final bias proposal overflows.
    const nn::Dataset targets{{-0.25, -2}};
    // Pre-rejection predictions independently record resident parameters' observable behavior.
    const auto before = session.run(inputs, {}, 0);
    // Whether production numerical status propagated the staged overflow.
    bool rejected = false;
    try { session.run(inputs, targets, 3, std::numeric_limits<double>::max()); }
    catch (const std::runtime_error& failure /* expected production numerical rejection */) {
        rejected = std::string(failure.what()).find("Nonfinite GPU arithmetic") != std::string::npos;
    }
    mock::require(rejected, "Nonfinite candidate was accepted");
    close(session.run(inputs, {}, 0), before);
    close(session.run(inputs, targets, 2), nn::detail::batchGradient(layers, inputs, targets));
    close(session.run(inputs, targets, 3, 0.1), {0, 0, -0.025, -0.2});
}
/* Run host-only OpenCL orchestration integration checks; never claim physical accelerator execution. */
int main() {
    try {
        // Every supported hidden nonlinearity exercises the real forward/reverse metadata and bindings.
        for (const auto activation : {nn::Activation::Linear, nn::Activation::Sigmoid, nn::Activation::Tanh, nn::Activation::Relu})
            checkSession(activation);
        checkRejectedUpdate();
        mock::require(mock::contexts == 1 && mock::programs == 1, "Immutable runtime was rebuilt between sessions");
        mock::require(mock::memories.empty() && mock::kernels.empty() && mock::queues.empty() && mock::uploads.empty(),
            "Session leaked private OpenCL resources");
        // Every production kernel entry must have traversed the real session enqueue path.
        for (const char* name : {"forwardLayer", "outputDelta", "hiddenDelta", "parameterGradient", "stageUpdate", "commitUpdate", "lossBlocks", "finishLoss"})
            mock::require(mock::launches[name] != 0, "Production kernel was not exercised");
        std::cout << "Host-only real OpenClSession integration checks passed (NOT physical GPU execution)\n";
        return 0;
    } catch (const std::exception& error /* failed invariant or production operation */) {
        std::cerr << "OpenClSession host mock failed: " << error.what() << '\n';
        return 1;
    }
}
