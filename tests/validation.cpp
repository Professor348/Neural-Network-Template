#include <nn/Network.h>
#include <limits>
#include <stdexcept>

/* Expect invalid_argument from action: operation under test; throws runtime_error when accepted incorrectly. */
void invalid(const std::function<void()>& action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid input accepted");
}

/* Exercise public validation, update atomicity and callback exception propagation; throws on failure. */
int main() {
    /* Construct invalid zero-width architecture; no inputs or outputs. */
    invalid(
        /* Construct an invalid model to verify input dimension validation. */
        [] {
            // Deliberately invalid zero-width network.
            nn::Network network(0);
        });
    // Valid model for dimension and value checks.
    nn::Network network(2, nn::DeviceType::CPU);
    /* Empty architecture must reject inference. */
    invalid([&network] { network.predict({{1, 2}}); });
    /* Dense output dimensions must be positive. */
    invalid([&network] { network.addDense(0); });
    /* Unknown activations must be rejected. */
    invalid([&network] { network.addDense(1, static_cast<nn::Activation>(999)); });
    network.addDense(1);
    /* Empty input datasets must be rejected. */
    invalid([&network] { network.predict({}); });
    /* Ragged inputs must be rejected. */
    invalid([&network] { network.predict({{1,2},{1}}); });
    /* Infinite inputs must be rejected. */
    invalid([&network] { network.predict({{1, std::numeric_limits<double>::infinity()}}); });
    /* Mismatched target counts must be rejected. */
    invalid([&network] { network.train({{1,2}}, {{1},{2}}); });
    /* Mismatched target feature widths must be rejected. */
    invalid([&network] { network.evaluate({{1,2}}, {{1,2}}); });
    /* Nonfinite targets must be rejected. */
    invalid([&network] { network.evaluate({{1,2}}, {{std::numeric_limits<double>::quiet_NaN()}}); });
    /* Invalid loss enums must be rejected. */
    invalid([&network] { network.setLoss(static_cast<nn::Loss>(999)); });
    // Initial parameter state used to check rejected mutation.
    const auto before = network.parameters();
    /* Wrong parameter count must be rejected atomically. */
    invalid([&network] { network.setParameters({}); });
    // Nonfinite candidate parameter vector.
    auto bad = before;
    bad.back() = std::numeric_limits<double>::quiet_NaN();
    /* Invalid scalar values must not partially change model state. */
    invalid([&network, &bad] { network.setParameters(bad); });
    // Training settings varied to exercise validation before any update.
    nn::TrainingOptions options;
    options.epochs = 0;
    /* Zero epochs must be rejected. */
    invalid([&network, &options] { network.train({{1,2}}, {{1}}, options); });
    options.epochs = 1;
    options.batchSize = 0;
    /* Zero batch size must be rejected. */
    invalid([&network, &options] { network.train({{1,2}}, {{1}}, options); });
    options.batchSize = 1;
    options.optimizer.learningRate = -1;
    /* Negative learning rates must be rejected. */
    invalid([&network, &options] { network.train({{1,2}}, {{1}}, options); });
    options.optimizer.learningRate = std::numeric_limits<double>::infinity();
    /* Infinite learning rates must be rejected. */
    invalid([&network, &options] { network.train({{1,2}}, {{1}}, options); });
    if (before != network.parameters()) throw std::runtime_error("Rejected operation mutated parameters");
    options.optimizer.learningRate = 0.1;
    // Whether caller-owned observer exception escaped unchanged.
    bool propagated = false;
    try {
        /* Deliberately fail the synchronous observer; progress is the completed epoch (unused). */
        network.train({{1,2}}, {{1}}, options, [](const nn::Progress& /* progress */) { throw 123; });
    } catch (int value /* caller exception sentinel */) { propagated = value == 123; }
    if (!propagated) throw std::runtime_error("Callback exception swallowed");
    // Finite parameters with a deliberately overflowing candidate SGD update.
    nn::Network overflow(2, nn::DeviceType::CPU);
    overflow.addDense(1);
    overflow.setParameters({1, 1, 0});
    // Copy used to verify atomic rejection of the entire batch update.
    const auto stable = overflow.parameters();
    options.optimizer.learningRate = std::numeric_limits<double>::max();
    // Numerical failure must propagate rather than falling back or partially updating.
    bool failed = false;
    try { overflow.train({{1,1}}, {{0}}, options); }
    catch (const std::runtime_error&) { failed = true; }
    if (!failed || overflow.parameters() != stable || overflow.getDevice().type != nn::DeviceType::CPU)
        throw std::runtime_error("Overflow update was not atomic");

}
