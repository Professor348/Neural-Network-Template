#pragma once
#include <nn/Dense.h>
namespace nn::detail {
/* Check finite value is representable as float32; value is canonical data, name describes its source.
 * Return its explicitly permitted reduced representation, or throw runtime_error; no numerical inference occurs. */
float npuFloat(double value, const char* name);
/* Validate layers and static samples, then encode device, batch, operation, full architecture and all canonical
 * double parameter bits into an exact key. Returns owned words; throws runtime_error on malformed/oversized data.
 * No hash or rounded parameter representation is used, and no inference runs. */
std::vector<std::uint64_t> npuGraphKey(const std::string& device, const std::vector<Dense>& layers,
    std::size_t samples, bool loss);
/* Validate dense layers and exact static samples; returns nothing and never infers.
 * Throws runtime_error for zero/overflowed shapes, inconsistent topology, bad activations or non-float32 parameters. */
void npuValidateGraph(const std::vector<Dense>& layers, std::size_t samples);
}
