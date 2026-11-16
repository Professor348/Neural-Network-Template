#pragma once
#include <nn/Dense.h>
#include <openvino/openvino.hpp>
#include <memory>
namespace nn::detail {
/* Build a static float32 dense graph from layers/current canonical weights for samples; loss adds targets,
 * subtraction, squaring and reduction over both axes. Return validated graph; throws on unsupported model/data.
 * No inference runs and the resulting constants own their converted parameter storage. */
std::shared_ptr<ov::Model> buildNpuGraph(const std::vector<Dense>& layers, std::size_t samples, bool loss);
}
