#include "SwiGLU.h"
#include "swiglu_kernel.h"

void SwiGLU::forward(
    const Tensor& gate,
    const Tensor& up,
    Tensor& output,
    ForwardContext& context
) {
    if (gate.data == nullptr 
        || up.data == nullptr 
        || output.data == nullptr 
        || context.batch == nullptr) {
        return;
    }

    const size_t num_tokens = context.batch->num_tokens;

    launch_swiglu_kernel_from_gate_up(
        static_cast<const float*>(gate.data),
        static_cast<const float*>(up.data),
        static_cast<float*>(output.data),
        num_tokens,
        hidden_size
    );
}