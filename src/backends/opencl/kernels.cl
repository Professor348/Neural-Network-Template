#ifdef __OPENCL_C_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
#endif

/* Evaluate scalar x through activation kind (0 linear, 1 sigmoid, 2 tanh, 3 ReLU); returns scalar. */
double activation(double x, ulong kind) {
    if (kind == 0) return x;
    if (kind == 1) {
        if (x >= 0) return 1.0 / (1.0 + exp(-x));
        // Negative exponential branch avoids overflow.
        const double exponential = exp(x);
        return exponential / (1.0 + exponential);
    }
    if (kind == 2) return tanh(x);
    return x > 0 ? x : 0;
}
/* Return derivative from activated value y and activation kind; ReLU derivative at zero is zero. */
double activationDerivative(double y, ulong kind) {
    if (kind == 0) return 1;
    if (kind == 1) return y * (1 - y);
    if (kind == 2) return 1 - y * y;
    return y > 0 ? 1 : 0;
}
/* Forward one neuron per work item over sample-major data.
 * parameters: flattened model; values: all batch layer activations; error: atomic failure flag;
 * samples/inputs/outputs: dimensions; weights/incoming/outgoing: buffer offsets; kind: activation. */
__kernel void forwardLayer(__global const double* parameters, __global double* values,
    volatile __global int* error, ulong samples, ulong inputs, ulong outputs,
    ulong weights, ulong incoming, ulong outgoing, ulong kind) {
    // Flattened output coordinate, output feature varying fastest.
    const ulong index = get_global_id(0);
    if (index >= samples * outputs) return;
    // Output neuron within its sample.
    const ulong row = index % outputs;
    // Input row offset for this sample.
    const ulong input = incoming + (index / outputs) * inputs;
    // Float64 weighted sum for this neuron.
    double sum = parameters[weights + inputs * outputs + row];
    // Contiguous input feature coordinate.
    for (ulong column = 0; column < inputs; ++column)
        sum += parameters[weights + row * inputs + column] * values[input + column];
    if (!isfinite(sum)) atomic_or(error, 1);
    values[outgoing + index] = activation(sum, kind);
}
/* Output derivatives, one sample/output coordinate per work item.
 * values/targets: predictions and labels; deltas: preactivation derivatives; error: failure flag;
 * samples/outputs: dimensions; outgoing: final activations offset; kind: final activation. */
__kernel void outputDelta(__global const double* values, __global const double* targets,
    __global double* deltas, volatile __global int* error, ulong samples, ulong outputs, ulong outgoing, ulong kind) {
    // Final layer coordinate for this sample.
    const ulong index = get_global_id(0);
    if (index >= samples * outputs) return;
    // Final activated scalar value.
    const double value = values[outgoing + index];
    // Mean-loss derivative with respect to the output preactivation.
    const double delta = 2 * (value - targets[index]) / (double)samples / (double)outputs * activationDerivative(value, kind);
    if (!isfinite(delta)) atomic_or(error, 1);
    deltas[outgoing + index] = delta;
}
/* Propagate from one layer into its activated inputs; one input feature/sample per work item.
 * parameters/values: model/cache; deltas: preactivation derivatives; error: failure flag;
 * samples/inputs/outputs: layer dimensions; weights/incoming/outgoing: offsets; kind: input activation. */
__kernel void hiddenDelta(__global const double* parameters, __global const double* values,
    __global double* deltas, volatile __global int* error, ulong samples, ulong inputs, ulong outputs,
    ulong weights, ulong incoming, ulong outgoing, ulong kind) {
    // Flattened input coordinate at this layer boundary.
    const ulong index = get_global_id(0);
    if (index >= samples * inputs) return;
    // Feature coordinate to backpropagate.
    const ulong feature = index % inputs;
    // Next-layer derivatives for this sample.
    const ulong next = outgoing + (index / inputs) * outputs;
    // Chain-rule sum across outgoing neurons.
    double sum = 0;
    // Next-layer output neuron index.
    for (ulong row = 0; row < outputs; ++row)
        sum += parameters[weights + row * inputs + feature] * deltas[next + row];
    sum *= activationDerivative(values[incoming + index], kind);
    if (!isfinite(sum)) atomic_or(error, 1);
    deltas[incoming + index] = sum;
}
/* Reduce sample contributions for one parameter per work item; no floating-point atomics.
 * values/deltas: batch cache and preactivation derivatives; gradient: canonical parameter order;
 * error: failure flag; samples/inputs/outputs: dimensions; weights/incoming/outgoing: offsets. */
__kernel void parameterGradient(__global const double* values, __global const double* deltas,
    __global double* gradient, volatile __global int* error, ulong samples, ulong inputs, ulong outputs,
    ulong weights, ulong incoming, ulong outgoing) {
    // One work item per weight or bias in this layer.
    const ulong index = get_global_id(0);
    if (index >= outputs * (inputs + 1)) return;
    // Neuron output coordinate.
    const ulong row = index / (inputs + 1);
    // Feature coordinate; inputs denotes bias.
    const ulong column = index % (inputs + 1);
    // Mean gradient accumulated in deterministic sample order.
    double sum = 0;
    // Sample coordinate across batch.
    for (ulong sample = 0; sample < samples; ++sample)
        sum += deltas[outgoing + sample * outputs + row] * (column == inputs ? 1.0 : values[incoming + sample * inputs + column]);
    if (!isfinite(sum)) atomic_or(error, 1);
    gradient[weights + (column == inputs ? inputs * outputs + row : row * inputs + column)] = sum;
}
/* Stage SGD candidates in parallel; parameters/gradient: current model and derivatives; candidate: staged outputs;
 * error: atomic shared failure flag; count: parameter length; rate: SGD step. Does not mutate current model. */
__kernel void stageUpdate(__global const double* parameters, __global const double* gradient,
    __global double* candidate, volatile __global int* error, ulong count, double rate) {
    // Unique parameter coordinate.
    const ulong index = get_global_id(0);
    if (index >= count) return;
    // Proposed parameter, validated before the separate commit launch.
    const double value = parameters[index] - rate * gradient[index];
    if (!isfinite(value)) atomic_or(error, 1);
    candidate[index] = value;
}
/* Commit candidates only if every prior kernel succeeded. parameters: mutable model; candidate: validated proposal;
 * error: flag stable across this launch, count: parameter length. Must launch after stageUpdate on the in-order queue. */
__kernel void commitUpdate(__global double* parameters, __global const double* candidate,
    __global const int* error, ulong count) {
    // Unique parameter coordinate.
    const ulong index = get_global_id(0);
    if (index < count && error[0] == 0) parameters[index] = candidate[index];
}
/* Compute independent scaled loss blocks; values/targets: predictions/labels; partial: scale/squares pairs;
 * error: atomic failure flag; count: scalar output count; outgoing: prediction offset. Each work item reduces up to 256 terms. */
__kernel void lossBlocks(__global const double* values, __global const double* targets,
    __global double* partial, volatile __global int* error, ulong count, ulong outgoing) {
    // Independent loss block index.
    const ulong block = get_global_id(0);
    // First scalar residual in this block.
    const ulong start = block * 256;
    if (start >= count) return;
    // Maximum residual magnitude within this block.
    double scale = 0;
    // Sum of squared residuals relative to current scale.
    double squares = 0;
    // Scalar residual coordinate inside this block.
    for (ulong index = start; index < count && index - start < 256; ++index) {
        // Current absolute prediction error.
        const double magnitude = fabs(values[outgoing + index] - targets[index]);
        if (!isfinite(magnitude)) atomic_or(error, 1);
        if (magnitude > scale) {
            // Rescale old contributions to the new largest magnitude.
            const double ratio = scale / magnitude;
            squares = 1 + squares * ratio * ratio;
            scale = magnitude;
        } else if (magnitude != 0) {
            // Current residual's normalized contribution.
            const double ratio = magnitude / scale;
            squares += ratio * ratio;
        }
    }
    partial[2 * block] = scale;
    partial[2 * block + 1] = squares;
}
/* Combine scaled blocks into scalar MSE, one final reduction work item.
 * partial: scale/squares pairs; result: scalar loss; error: atomic failure flag; blocks: pair count; count: output scalar count. */
__kernel void finishLoss(__global const double* partial, __global double* result,
    volatile __global int* error, ulong blocks, ulong count) {
    // Largest partial scale seen so far.
    double scale = 0;
    // Combined sum of squared residuals relative to scale.
    double squares = 0;
    // Partial loss block index.
    for (ulong block = 0; block < blocks; ++block) {
        // Current block scale.
        const double magnitude = partial[2 * block];
        if (magnitude > scale) {
            // Old scale expressed in the new scale's units.
            const double ratio = scale / magnitude;
            squares = partial[2 * block + 1] + squares * ratio * ratio;
            scale = magnitude;
        } else if (magnitude != 0) {
            // Current block scale relative to the running maximum.
            const double ratio = magnitude / scale;
            squares += partial[2 * block + 1] * ratio * ratio;
        }
    }
    // Normalize before squaring to preserve finite large MSE values.
    const double rootMean = scale * sqrt(squares / (double)count);
    result[0] = rootMean * rootMean;
    if (!isfinite(result[0])) atomic_or(error, 1);
}
