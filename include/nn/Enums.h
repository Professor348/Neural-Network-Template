#pragma once
#include "Types.h"
namespace nn {
// Supported device preferences; AUTO resolves separately for each operation.
enum class DeviceType { AUTO, CPU, GPU, NPU };
// Each operation must run entirely on one compatible device.
enum class Operation { Training, Inference };
// Elementwise dense layer activation choices.
enum class Activation { Linear, Sigmoid, Tanh, Relu };
// Mean over samples and output coordinates.
enum class Loss { MeanSquaredError };
}
