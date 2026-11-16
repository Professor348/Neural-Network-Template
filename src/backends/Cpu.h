#pragma once
#include <nn/Dense.h>
namespace nn::detail {
/* Return independent prediction rows, reusing scratch across inputs. layers: validated model;
 * inputs: validated sample rows; throws on nonfinite computation without mutating layers or inputs. */
Dataset predictBatch(const std::vector<Dense>& layers, const Dataset& inputs);
/* Return dataset MSE. layers: model; inputs/targets: validated samples; throws on numerical failure. */
double meanLoss(const std::vector<Dense>& layers, const Dataset& inputs, const Dataset& targets);
/* Return flattened mean-loss gradient. layers: model; inputs/targets: validated nonempty batch.
 * Does not update model; throws on nonfinite computation. */
std::vector<double> batchGradient(const std::vector<Dense>& layers, const Dataset& inputs, const Dataset& targets);
}
