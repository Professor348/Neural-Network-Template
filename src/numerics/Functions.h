#pragma once
#include <nn/Enums.h>
namespace nn::detail {
/* Evaluate activation. value: preactivation scalar; activation: nonlinearity; returns finite output or throws. */
double activate(double value, Activation activation);
/* Return derivative using output: activated scalar and activation: nonlinearity. ReLU derivative at zero is zero. */
double derivative(double output, Activation activation);
/* Reject NaN/infinity. value: scalar to check; throws runtime_error on nonfinite numerical result. */
void requireFinite(double value);
}
