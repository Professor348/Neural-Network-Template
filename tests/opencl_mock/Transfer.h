#pragma once
#include <cstddef>

namespace mock {
// Transfer metadata used to inspect production result readback offsets.
struct Transfer {
    // Original byte offset within the OpenCL buffer.
    std::size_t offset;
    // Original requested byte count.
    std::size_t bytes;
};
}
