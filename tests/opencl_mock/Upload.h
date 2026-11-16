#pragma once
#include <CL/cl.h>
#include <cstddef>
#include <vector>

namespace mock {
// Delayed host upload record exercises nonblocking source-pointer lifetimes.
struct Upload {
    // Destination owned by the session.
    cl_mem memory;
    // Byte position in the destination.
    std::size_t offset;
    // Host address that must remain unchanged until execution reaches this transfer.
    const void* source;
    // Snapshot used to detect premature host storage reuse.
    std::vector<unsigned char> snapshot;
};
}
