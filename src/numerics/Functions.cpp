#include "Functions.h"
#include <cmath>
#include <stdexcept>
namespace nn::detail {
/* Reject nonfinite computation; value is scalar result; throws runtime_error without fallback. */
void requireFinite(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite numerical result; execution stopped without device fallback");
}
/* Evaluate value through activation, returning scalar; throws for bad enum or nonfinite preactivation. */
double activate(double value, Activation activation) {
    requireFinite(value);
    switch (activation) {
        case Activation::Linear: return value;
        case Activation::Tanh: return std::tanh(value);
        case Activation::Relu: return value > 0 ? value : 0;
        case Activation::Sigmoid:
            if (value >= 0) return 1 / (1 + std::exp(-value));
            // Negative-branch exponential avoids overflow.
            const double exponential = std::exp(value);
            return exponential / (1 + exponential);
    }
    throw std::invalid_argument("Unknown activation");
}
/* Return local derivative from activated output and activation enum; throws for invalid activation. */
double derivative(double output, Activation activation) {
    switch (activation) {
        case Activation::Linear: return 1;
        case Activation::Tanh: return 1 - output * output;
        case Activation::Relu: return output > 0 ? 1 : 0;
        case Activation::Sigmoid: return output * (1 - output);
    }
    throw std::invalid_argument("Unknown activation");
}
}
