#include <nn/Network.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
/* Check condition; message identifies the failed hardware contract, including context. */
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

/* Check action throws Exception; message identifies an unexpectedly accepted operation. */
template<class Exception, class Action>
void requireRejected(const Action& action, const std::string& message) {
    try { action(); }
    catch (const Exception& error /* expected rejection, retained for useful diagnostics */) {
        require(std::string(error.what()).size() != 0, message + ": empty rejection diagnostic");
        return;
    }
    throw std::runtime_error(message);
}

/* Compare actual to expected using absolute/relative error limits; context identifies the scalar. */
void near(double actual, double expected, double absolute, double relative, const std::string& context) {
    require(std::isfinite(actual) && std::isfinite(expected), context + ": nonfinite value");
    require(std::abs(actual - expected) <= absolute + relative * std::abs(expected),
        context + ": actual=" + std::to_string(actual) + ", CPU=" + std::to_string(expected));
}

/* Return deterministic, nonsymmetric three-feature samples; count is the requested batch size. */
nn::Dataset inputsFor(std::size_t count) {
    // Samples span both activation signs without large FP16 intermediates.
    nn::Dataset inputs;
    // Sample index controls independent feature sequences.
    for (std::size_t sample = 0; sample < count; ++sample) {
        inputs.push_back({0.17 * static_cast<double>(sample % 7) - 0.51,
            0.13 * static_cast<double>((sample * 3 + 1) % 9) - 0.39,
            0.11 * static_cast<double>((sample * 5 + 2) % 11) - 0.44});
    }
    return inputs;
}

/* Return deterministic two-output targets; count is the matching sample count. */
nn::Dataset targetsFor(std::size_t count) {
    // Different coordinates expose erroneous batch/output MSE normalization.
    nn::Dataset targets;
    // Sample index provides distinct values across coordinates and rows.
    for (std::size_t sample = 0; sample < count; ++sample) {
        targets.push_back({0.19 * static_cast<double>(sample % 5) - 0.3,
            0.23 * static_cast<double>((sample + 2) % 7) - 0.8});
    }
    return targets;
}

/* Build a CPU-configured 3->5->2 model; activation applies to both asymmetric dense layers. */
nn::Network hiddenModel(nn::Activation activation) {
    // CPU initialization avoids AUTO selection or any accelerator numerical work.
    nn::Network network(3, nn::DeviceType::CPU);
    network.addDense(5, activation);
    network.addDense(2, activation);
    // Explicit output-major weights and biases expose transpose/layout mistakes.
    auto parameters = network.parameters();
    // Flattened coordinate makes each weight and bias deterministic and asymmetric.
    for (std::size_t coordinate = 0; coordinate < parameters.size(); ++coordinate) {
        parameters[coordinate] = 0.09 * static_cast<double>((coordinate * 7 + 3) % 13) - 0.49;
    }
    network.setParameters(parameters);
    network.setNpuOptions({true});
    return network;
}

/* Return a known 3->2 linear CPU model, with explicit NPU reduced-precision permission. */
nn::Network linearModel() {
    // Exact known parameters make synchronization changes visible above FP16 tolerances.
    nn::Network network(3, nn::DeviceType::CPU);
    network.addDense(2, nn::Activation::Linear);
    network.setParameters({0.5, -0.25, 0.125, -0.375, 0.625, 0.25, 0.125, -0.25});
    network.setNpuOptions({true});
    return network;
}

/* Compare full actual/expected prediction tensors; device is the requested real NPU and context labels the case. */
void compare(const nn::Prediction& actual, const nn::Prediction& expected,
    const nn::DeviceInfo& device, const std::string& context) {
    require(actual.selection.device.type == nn::DeviceType::NPU && actual.selection.device.index == device.index,
        context + ": did not execute on requested NPU index " + std::to_string(device.index));
    require(expected.selection.device.type == nn::DeviceType::CPU, context + ": oracle did not execute on CPU");
    require(actual.values.size() == expected.values.size(), context + ": batch size mismatch");
    // Every sample must have all expected finite output coordinates.
    for (std::size_t sample = 0; sample < expected.values.size(); ++sample) {
        require(actual.values[sample].size() == expected.values[sample].size(), context + ": output width mismatch");
        // Every coordinate is checked; mixed tolerance accommodates FP16 rounding and near-zero values.
        for (std::size_t output = 0; output < expected.values[sample].size(); ++output) {
            // 0.004 absolute + 2% relative permits FP16 dense/nonlinear execution without hiding layout errors.
            near(actual.values[sample][output], expected.values[sample][output], 0.004, 0.02,
                context + " sample " + std::to_string(sample) + " output " + std::to_string(output));
        }
    }
}

/* Exercise all supported nonlinearities and static batches on device, using CPU double results as oracle. */
void inferenceCase(const nn::DeviceInfo& device) {
    // Explicit override guarantees the selected device cannot silently become CPU or GPU.
    const nn::DevicePreference npu{nn::DeviceType::NPU, device.index};
    // Each supported activation must compile and execute as a complete graph on this NPU.
    for (const auto activation : {nn::Activation::Linear, nn::Activation::Sigmoid, nn::Activation::Tanh, nn::Activation::Relu}) {
        // Each architecture has unequal input/hidden/output widths and multiple outputs.
        auto network = hiddenModel(activation);
        // Single sample and nontrivial static batches exercise compilation/cache shape keys.
        for (const std::size_t batch : {std::size_t{1}, std::size_t{3}, std::size_t{17}}) {
            // Identical data is supplied to the CPU oracle and physical NPU graph.
            const auto inputs = inputsFor(batch);
            compare(network.predict(inputs, npu), network.predict(inputs), device,
                "activation " + std::to_string(static_cast<int>(activation)) + " batch " + std::to_string(batch));
        }
    }
}

/* Verify entire-graph MSE on device; unsupported loss compilation is a failure, never a skip. */
void lossCase(const nn::DeviceInfo& device) {
    // Explicit preference requires physical NPU execution for prediction and scalar loss.
    const nn::DevicePreference npu{nn::DeviceType::NPU, device.index};
    // Nonlinear graph verifies loss is attached to all network layers, not a separate fake execution path.
    auto network = hiddenModel(nn::Activation::Tanh);
    // Unequal batch/output dimensions expose mean normalization errors.
    for (const std::size_t batch : {std::size_t{1}, std::size_t{3}, std::size_t{17}}) {
        // Shared input tensor for graph prediction and MSE.
        const auto inputs = inputsFor(batch);
        // Shared target tensor with multiple distinct output coordinates.
        const auto targets = targetsFor(batch);
        // Full-precision CPU prediction validates the library's mean-loss convention independently.
        const auto reference = network.predict(inputs);
        compare(network.predict(inputs, npu), reference, device, "MSE forward batch " + std::to_string(batch));
        // Independent sum of all coordinate errors before division by sample and output counts.
        double manual = 0;
        // Sample index into the reference/target tensors.
        for (std::size_t sample = 0; sample < batch; ++sample) {
            // Output index includes both coordinates in the normalization check.
            for (std::size_t output = 0; output < targets[sample].size(); ++output) {
                // Signed coordinate error is squared before reduction.
                const double difference = reference.values[sample][output] - targets[sample][output];
                manual += difference * difference;
            }
        }
        manual /= static_cast<double>(batch * targets.front().size());
        near(network.evaluate(inputs, targets), manual, 1e-13, 1e-13, "CPU MSE normalization");
        // FP16 subtraction/square/reduction add error beyond forward inference: 0.006 absolute + 3% relative.
        near(network.evaluate(inputs, targets, npu), manual, 0.006, 0.03,
            "whole-graph NPU MSE batch " + std::to_string(batch));
    }
}

/* Check cached graphs observe CPU updates and explicit parameter replacement on device. */
void synchronizationCase(const nn::DeviceInfo& device) {
    // Call-only NPU override is intentionally applied to a persistent CPU preference.
    const nn::DevicePreference npu{nn::DeviceType::NPU, device.index};
    // Known linear parameters make stale cached graphs easy to detect.
    auto network = linearModel();
    // Batch of three is reused before and after parameter updates to exercise the same cache entry.
    const auto inputs = inputsFor(3);
    // Targets differ from initial predictions and therefore produce a nonzero CPU update.
    const auto targets = targetsFor(3);
    // Initial host snapshot proves CPU training actually changes the canonical model.
    const auto before = network.parameters();
    compare(network.predict(inputs, npu), network.predict(inputs), device, "initial cached prediction");
    compare(network.predict(inputs, npu), network.predict(inputs), device, "repeated cached prediction");
    // Short deterministic SGD execution always uses CPU; this is not NPU training.
    nn::TrainingOptions options;
    options.epochs = 3;
    options.batchSize = 3;
    options.shuffle = false;
    options.optimizer.learningRate = 0.2;
    // Training uses an explicit CPU override even if future defaults change.
    const auto trained = network.train(inputs, targets, options, {}, nn::DevicePreference{nn::DeviceType::CPU, 0});
    require(trained.selection.device.type == nn::DeviceType::CPU, "synchronization training did not use CPU");
    require(network.parameters() != before, "CPU training did not change parameters");
    compare(network.predict(inputs, npu), network.predict(inputs), device, "after CPU training");
    // Canonical replacement alters both layers' output coordinates by much more than FP16 error bounds.
    network.setParameters({-0.75, 0.125, 0.5, 0.25, -0.625, -0.375, 0.625, -0.75});
    compare(network.predict(inputs, npu), network.predict(inputs), device, "after setParameters same batch");
    // Changed static batch requires a separate graph using the latest canonical parameters.
    const auto changedBatch = inputsFor(17);
    compare(network.predict(changedBatch, npu), network.predict(changedBatch), device, "after setParameters changed batch");
    compare(network.predict(inputs, npu), network.predict(inputs), device, "after returning to cached batch");
    require(network.getDevice().type == nn::DeviceType::CPU && !network.getDevice().index,
        "call-only NPU override changed persistent CPU preference");
    require(network.predict(inputs).selection.device.type == nn::DeviceType::CPU,
        "call-only NPU override persisted into the next prediction");
    // Permission is checked on every call, including graphs cached while permission was enabled.
    network.setNpuOptions({false});
    requireRejected<std::runtime_error>(
        /* Cached graph must not bypass revoked precision permission; captures current network/data/device. */
        [&network, &inputs, &npu] { network.predict(inputs, npu); }, "cached graph bypassed reduced-precision opt-out");
}

/* Verify precision, inference-only, invalid-index, and AUTO-training public contracts on device. */
void contractsCase(const nn::DeviceInfo& device) {
    // Explicit configured NPU is valid at construction even before precision permission is granted.
    nn::Network network(3, nn::DeviceType::NPU, device.index);
    network.addDense(2, nn::Activation::Linear);
    network.setParameters({0.5, -0.25, 0.125, -0.375, 0.625, 0.25, 0.125, -0.25});
    // Shared data is finite and valid, so any expected rejection concerns device policy.
    const auto inputs = inputsFor(3);
    // Matching targets also ensure unsupported training is not obscured by validation errors.
    const auto targets = targetsFor(3);
    // Exact double snapshot detects any mutation by rejected calls.
    const auto original = network.parameters();
    require(!network.getNpuOptions().allowReducedPrecision, "reduced precision was enabled by default");
    requireRejected<std::runtime_error>(
        /* Default explicit NPU inference must reject precision demotion without changing model state. */
        [&network, &inputs] { network.predict(inputs); }, "default NPU precision permission was bypassed");
    requireRejected<std::runtime_error>(
        /* Loss is inference too, and must require explicit reduced-precision permission. */
        [&network, &inputs, &targets] { network.evaluate(inputs, targets); }, "default NPU loss permission was bypassed");
    require(network.parameters() == original, "rejected default inference/loss changed parameters");
    network.setNpuOptions({true});
    compare(network.predict(inputs), network.predict(inputs, nn::DevicePreference{nn::DeviceType::CPU, 0}),
        device, "explicit opt-in inference");
    // Valid minimal training options distinguish unsupported operation from invalid-option rejection.
    nn::TrainingOptions options;
    options.epochs = 1;
    options.batchSize = 3;
    options.shuffle = false;
    requireRejected<std::runtime_error>(
        /* Explicit NPU training must reject before modifying any weight or bias. */
        [&network, &inputs, &targets, &options] { network.train(inputs, targets, options); }, "NPU training was accepted");
    require(network.parameters() == original, "rejected NPU training changed parameters");
    requireRejected<std::runtime_error>(
        /* Explicit NPU gradient requests must also reject instead of silently using CPU. */
        [&network, &inputs, &targets] { network.gradients(inputs, targets); }, "NPU gradients were accepted");
    require(network.parameters() == original, "rejected NPU gradients changed parameters");
    requireRejected<std::runtime_error>(
        /* An impossible explicit NPU index must fail; the model must retain its prior valid preference. */
        [&network] { network.setDevice(nn::DeviceType::NPU, std::numeric_limits<std::size_t>::max()); },
        "invalid NPU index was accepted");
    requireRejected<std::invalid_argument>(
        /* AUTO may not bind an index, independent of the available hardware count. */
        [&network] { network.setDevice(nn::DeviceType::AUTO, 0); }, "AUTO with an index was accepted");
    require(network.getDevice().type == nn::DeviceType::NPU && network.getDevice().index == device.index,
        "rejected preference mutation changed configured NPU");
    // CPU gradient oracle remains precise even though the model permits reduced-precision inference.
    const auto expectedGradient = network.gradients(inputs, targets, nn::DevicePreference{nn::DeviceType::CPU, 0});
    // AUTO must skip inference-only NPU before differentiation, and may choose a supported GPU or CPU.
    const auto automaticGradient = network.gradients(inputs, targets, nn::DevicePreference{});
    require(automaticGradient.size() == expectedGradient.size(), "AUTO gradient count mismatch");
    // Every derivative is checked against full-precision CPU; no FP16 gradient tolerance is used.
    for (std::size_t coordinate = 0; coordinate < expectedGradient.size(); ++coordinate) {
        near(automaticGradient[coordinate], expectedGradient[coordinate], 1e-10, 1e-10, "AUTO training gradient");
    }
    // AUTO training selection is returned publicly and must identify a non-NPU implementation.
    const auto trained = network.train(inputs, targets, options, {}, nn::DevicePreference{});
    require(trained.selection.device.type != nn::DeviceType::NPU, "AUTO training selected inference-only NPU");
    require(std::isfinite(trained.initialLoss) && std::isfinite(trained.finalLoss), "AUTO training returned nonfinite loss");
    require(network.getDevice().type == nn::DeviceType::NPU && network.getDevice().index == device.index,
        "AUTO training override changed persistent NPU preference");
    compare(network.predict(inputs), network.predict(inputs, nn::DevicePreference{nn::DeviceType::CPU, 0}),
        device, "NPU after AUTO training");
}

/* Report missing hardware with diagnostic; return failure when NN_REQUIRE_NPU_TESTS is exactly 1, else skip 77. */
int unavailable(const std::string& diagnostic) {
    // Environment opt-in prevents a hardware-required CI run from silently passing through skips.
    const char* required = std::getenv("NN_REQUIRE_NPU_TESTS");
    std::cerr << "NPU hardware tests unavailable: " << diagnostic << '\n';
    return required && std::string(required) == "1" ? 1 : 77;
}
}

/* Run one named public-API-only hardware case; argc/argv select inference, loss, synchronization, or contracts. */
int main(int argc, char** argv) {
    // Reject invalid selectors before hardware discovery so typos cannot become skipped tests.
    const std::string name = argc == 2 ? argv[1] : "";
    if (argc != 2 || (name != "inference" && name != "loss" && name != "synchronization" && name != "contracts")) {
        std::cerr << "Usage: nn_test_npu_hardware {inference|loss|synchronization|contracts}\n";
        return 1;
    }
    try {
        // Only genuine public discovery results are eligible; no synthetic device is ever executed here.
        std::vector<nn::DeviceInfo> devices;
        try {
            // Full inventory can include CPU/GPU; retain all actual NPU indices without assuming only NPU 0.
            const auto available = nn::getAvailableDevices();
            // Discovered candidate is selected solely by its public processor type.
            for (const auto& candidate : available) {
                if (candidate.type == nn::DeviceType::NPU) devices.push_back(candidate);
            }
        } catch (const std::exception& error /* discovery errors explain unavailable hardware, including missing runtimes */) {
            return unavailable(std::string("device discovery failed: ") + error.what());
        }
        if (devices.empty()) {
            try {
                // Explicit construction exposes the backend's missing-SDK/plugin/driver/hardware diagnostic.
                nn::Network probe(1, nn::DeviceType::NPU, 0);
                (void)probe;
            } catch (const std::exception& error /* expected unavailable explicit NPU diagnostic */) {
                return unavailable(error.what());
            }
            return unavailable("public discovery returned no NPU, despite successful explicit NPU construction");
        }
        // Every discovered physical NPU must satisfy the selected case; later execution failures are fatal.
        for (const auto& device : devices) {
            std::cout << "Running NPU " << name << " on index " << device.index << ": " << device.name << '\n';
            if (name == "inference") inferenceCase(device);
            else if (name == "loss") lossCase(device);
            else if (name == "synchronization") synchronizationCase(device);
            else contractsCase(device);
        }
        std::cout << "NPU hardware " << name << " passed on " << devices.size() << " real device(s)\n";
        return 0;
    } catch (const std::exception& error /* execution failures must remain failures, even when hardware is optional */) {
        std::cerr << "NPU hardware " << name << " failed: " << error.what() << '\n';
        return 1;
    }
}
