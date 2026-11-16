#include "NpuState.h"
#include "Snapshot.h"
#include <cmath>
#include <stdexcept>
namespace nn::detail {
/* Create a session-private request for model and record its samples/inputWidth/outputWidth/loss contract; never infer. */
NpuState::NpuState(ov::CompiledModel model, std::size_t samples, std::size_t inputWidth,
    std::size_t outputWidth, bool loss)
    : request_(model.create_infer_request()), samples_(samples), inputWidth_(inputWidth),
      outputWidth_(outputWidth), loss_(loss) {}
/* Check and upload data of width into private port; name labels errors; never infer, runtime_error rejects bad input. */
void NpuState::upload(const Dataset& data, std::size_t port, std::size_t width, const char* name) {
    if (data.size() != samples_)
        throw std::runtime_error(std::string("NPU static batch mismatch for ") + name);
    // Input tensor is allocated for this private request and has the prevalidated static float32 shape.
    auto tensor = request_.get_input_tensor(port);
    if (tensor.get_element_type() != ov::element::f32 || tensor.get_shape() != ov::Shape{samples_, width})
        throw std::runtime_error("NPU runtime changed the prepared input tensor contract");
    // Writable storage is not shared with any other inference request.
    float* destination = tensor.data<float>();
    // Next flattened float32 input location.
    std::size_t offset = 0;
    // Dataset row being checked and transferred, with no host dense/activation/loss computation.
    for (const auto& row : data) {
        if (row.size() != width)
            throw std::runtime_error(std::string("NPU static feature width mismatch for ") + name);
        // Canonical scalar is explicitly narrowed only under the session's prior precision opt-in.
        for (const double value : row) destination[offset++] = npuFloat(value, name);
    }
}
/* Run this request once and return float32 output; caller owns mutex_; runtime failures become descriptive runtime_error. */
ov::Tensor NpuState::execute() {
    try {
        request_.infer();
        // Output remains valid while this private request is not executed again.
        auto output = request_.get_output_tensor();
        if (output.get_element_type() != ov::element::f32)
            throw std::runtime_error("runtime changed the prepared float32 output contract");
        return output;
    } catch (const std::exception& error /* committed NPU operation cannot be retried on CPU or any other device */) {
        throw std::runtime_error(std::string("NPU inference failed after device selection; no fallback: ") + error.what());
    }
}
/* Execute prepared prediction with inputs; serialize private mutable request, return widened device data, throw without fallback. */
Dataset NpuState::predict(const Dataset& inputs) {
    // One session can be called from multiple threads; no shared request storage is accessed without the lock.
    const std::lock_guard<std::mutex> lock(mutex_);
    if (loss_) throw std::runtime_error("NPU session was prepared for loss, not prediction");
    upload(inputs, 0, inputWidth_, "inputs");
    // Final prediction matrix is obtained directly from the selected NPU; host only checks and widens values.
    const auto output = execute();
    if (output.get_shape() != ov::Shape{samples_, outputWidth_})
        throw std::runtime_error("NPU runtime changed the prepared prediction output shape; no fallback");
    // Read-only pointer is valid until the next execution, prevented by mutex_ for the duration of the copy.
    const float* source = output.data<const float>();
    // Public sample-major double result widens the explicitly permitted reduced-precision device outputs.
    Dataset result(samples_, std::vector<double>(outputWidth_));
    // Next flattened output position.
    std::size_t offset = 0;
    // Output sample receiving its device-computed coordinates.
    for (auto& row : result) {
        // Public result scalar; host processing only checks/widens its corresponding device output.
        for (double& value : row) {
            value = source[offset++];
            if (!std::isfinite(value))
                throw std::runtime_error("NPU nonfinite prediction result; no fallback");
        }
    }
    return result;
}
/* Execute prepared complete MSE graph with inputs/targets; return only its device-computed scalar, throwing without fallback. */
double NpuState::loss(const Dataset& inputs, const Dataset& targets) {
    // Serializes both uploads and the scalar read from this session-private request.
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!loss_) throw std::runtime_error("NPU session was prepared for prediction, not loss");
    upload(inputs, 0, inputWidth_, "inputs");
    upload(targets, 1, outputWidth_, "targets");
    // All residuals, squares and averaging were executed as part of this single NPU graph.
    const auto output = execute();
    if (output.get_shape() != ov::Shape{1, 1} || output.get_size() != 1)
        throw std::runtime_error("NPU runtime changed the prepared loss output shape; no fallback");
    // The only host arithmetic is widening the already-reduced float32 scalar to the public double type.
    const double value = output.data<const float>()[0];
    if (!std::isfinite(value) || value < 0)
        throw std::runtime_error("NPU nonfinite or negative loss result; no fallback");
    return value;
}
}
