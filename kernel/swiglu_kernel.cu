#include "cuda_runtime.h"
#include "swiglu_kernel.h"


__global__ void swiglu_kernel_from_gate_up(
    const float* gate,
    const float* up,
    float* output,
    size_t hidden_size,
    size_t total_elements
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    const float gate_val = gate[idx];
    const float up_val = up[idx];
    const float sigmoid_gate = 1.0f / (1.0f + __expf(-gate_val));
    output[idx] = (gate_val * sigmoid_gate) * up_val;
}


void launch_swiglu_kernel_from_gate_up(
    const float* gate,
    const float* up,
    float* output,
    size_t num_tokens,
    size_t hidden_size
) {
    const size_t total_elements = num_tokens * hidden_size;
    if (total_elements == 0) {
        return;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    swiglu_kernel_from_gate_up<<<blocks, threads>>>(
        gate,
        up,
        output,
        hidden_size,
        total_elements
    );
}