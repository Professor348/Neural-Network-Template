#pragma once
#include <vector>

// Byte-preserving bound kernel argument, copied exactly as OpenCL requires.
struct BoundArgument {
    // ABI bytes supplied to clSetKernelArg.
    std::vector<unsigned char> bytes;
};
