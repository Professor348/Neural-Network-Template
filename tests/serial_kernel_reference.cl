// Historical serial GPU algebra, preserved only as a test oracle. Never built into the library.
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
/* Execute the complete operation in one GPU work item (correctness reference, not a parallel optimizer).
 * architecture: six ulong values per layer (input/output width, activation, weight/input/output offsets);
 * parameters: mutable flattened weights/biases; inputs/targets: sample-major data; output: predictions,
 * loss, gradients or updated parameters for mode 0/1/2/3; scratch: activations/deltas/gradient/candidate;
 * error: numerical error status; layers/samples/parameterCount/activationCount: dimensions;
 * rate: SGD step. Nonfinite results set error without committing a partial parameter update. */
__kernel void execute(__global const ulong* architecture, __global double* parameters,
    __global const double* inputs, __global const double* targets, __global double* output,
    __global double* scratch, __global int* error, ulong layers, ulong samples,
    ulong parameterCount, ulong activationCount, int mode, double rate) {
    // Activation workspace, one sample at a time.
    __global double* values = scratch;
    // Chain-rule workspace at each layer boundary.
    __global double* deltas = scratch + activationCount;
    // Batch-mean parameter derivatives.
    __global double* gradient = scratch + 2 * activationCount;
    // Proposed SGD parameters, committed only after complete validation.
    __global double* candidate = gradient + parameterCount;
    // Input feature width.
    const ulong inputWidth = architecture[0];
    // Final layer metadata start.
    const ulong last = 6 * (layers - 1);
    // Output feature width.
    const ulong outputWidth = architecture[last + 1];
    // Final activations offset.
    const ulong finalOffset = architecture[last + 5];
    // Scale for overflow-safe mean squared error.
    double scale = 0;
    // Sum of squares normalized by scale squared.
    double squares = 0;
    error[0] = 0;
    // Parameter index for clearing derivatives.
    for (ulong index = 0; index < parameterCount; ++index) gradient[index] = 0;
    // Dataset sample index.
    for (ulong sample = 0; sample < samples; ++sample) {
        // Input coordinate in this sample.
        for (ulong input = 0; input < inputWidth; ++input) values[input] = inputs[sample * inputWidth + input];
        // Current forward layer index.
        for (ulong layer = 0; layer < layers; ++layer) {
            // Metadata row offset for this layer.
            const ulong base = 6 * layer;
            // Input feature width.
            const ulong width = architecture[base];
            // Output feature width.
            const ulong count = architecture[base + 1];
            // Flattened parameter offset.
            const ulong weights = architecture[base + 3];
            // Output neuron index.
            for (ulong row = 0; row < count; ++row) {
                // Neuron preactivation in float64.
                double sum = parameters[weights + width * count + row];
                // Incoming feature coordinate.
                for (ulong column = 0; column < width; ++column)
                    sum += parameters[weights + row * width + column] * values[architecture[base + 4] + column];
                if (!isfinite(sum)) { error[0] = 1; return; }
                values[architecture[base + 5] + row] = activation(sum, architecture[base + 2]);
            }
        }
        // Final output coordinate.
        for (ulong feature = 0; feature < outputWidth; ++feature) {
            if (mode == 0) output[sample * outputWidth + feature] = values[finalOffset + feature];
            if (mode == 1) {
                // Absolute residual for scaled sum-of-squares loss.
                const double magnitude = fabs(values[finalOffset + feature] - targets[sample * outputWidth + feature]);
                if (!isfinite(magnitude)) { error[0] = 1; return; }
                if (magnitude > scale) {
                    // Rescale previous contributions to the new maximum.
                    const double ratio = scale / magnitude;
                    squares = 1 + squares * ratio * ratio;
                    scale = magnitude;
                } else if (magnitude != 0) {
                    // Contribution relative to the current maximum.
                    const double ratio = magnitude / scale;
                    squares += ratio * ratio;
                }
            }
            if (mode >= 2) deltas[finalOffset + feature] =
                2 * (values[finalOffset + feature] - targets[sample * outputWidth + feature]) / (double)samples / (double)outputWidth;
        }
        if (mode < 2) continue;
        // Reverse layer index for backpropagation.
        for (ulong layer = layers; layer-- > 0;) {
            // Metadata offset for this layer.
            const ulong base = 6 * layer;
            // Input width.
            const ulong width = architecture[base];
            // Output width.
            const ulong count = architecture[base + 1];
            // Flattened parameter offset.
            const ulong weights = architecture[base + 3];
            // Input activation offset.
            const ulong incoming = architecture[base + 4];
            // Output activation offset.
            const ulong outgoing = architecture[base + 5];
            // Input coordinate whose upstream derivative is reset.
            for (ulong input = 0; input < width; ++input) deltas[incoming + input] = 0;
            // Neuron output coordinate.
            for (ulong row = 0; row < count; ++row) {
                // Loss derivative with respect to preactivation.
                const double delta = deltas[outgoing + row] * activationDerivative(values[outgoing + row], architecture[base + 2]);
                gradient[weights + width * count + row] += delta;
                // Input feature coordinate for weight and upstream derivatives.
                for (ulong column = 0; column < width; ++column) {
                    gradient[weights + row * width + column] += delta * values[incoming + column];
                    deltas[incoming + column] += parameters[weights + row * width + column] * delta;
                }
            }
        }
    }
    if (mode == 1) {
        // Root mean square calculated before final square to avoid premature overflow.
        const double rootMean = scale * sqrt(squares / (double)samples / (double)outputWidth);
        output[0] = rootMean * rootMean;
        if (!isfinite(output[0])) error[0] = 1;
    }
    if (mode >= 2) {
        // Parameter coordinate for complete gradient/update validation.
        for (ulong index = 0; index < parameterCount; ++index) {
            if (!isfinite(gradient[index])) { error[0] = 1; return; }
            candidate[index] = parameters[index] - rate * gradient[index];
            if (mode == 3 && !isfinite(candidate[index])) { error[0] = 1; return; }
        }
        // Validated parameter coordinate for atomic batch commit or gradient return.
        for (ulong index = 0; index < parameterCount; ++index) {
            if (mode == 3) parameters[index] = candidate[index];
            output[index] = mode == 3 ? parameters[index] : gradient[index];
        }
    }
}
