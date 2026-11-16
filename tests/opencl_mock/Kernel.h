#pragma once
#include "BoundArgument.h"
#include <string>
#include <vector>

// Session-private kernel object retains its name and copied arguments.
struct _cl_kernel {
    // Production entry point selected at creation.
    std::string name;
    // Typed argument bytes in positional ABI order.
    std::vector<BoundArgument> arguments;
};
