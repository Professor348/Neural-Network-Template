#pragma once
#include <CL/cl.h>
#include <cstddef>
#include <vector>

// Mock memory with correctly aligned numerical storage and separate integer status.
struct _cl_mem {
    // Requested capacity in bytes, independent of rounded host allocation.
    std::size_t bytes;
    // Double-precision workspace backing every numerical allocation.
    std::vector<double> values;
    // Integer backing the shared numerical status allocation.
    cl_int error = 0;
};
