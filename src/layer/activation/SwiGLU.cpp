#include "SwiGLU.h"
#include "swiglu_kernel.h"

void SwiGLU::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    launch_swiglu_kernel(
        static_cast<const float*>(input.data),
        static_cast<float*>(output.data),
        context.batch->num_tokens,
        hidden_size
    );
}

void SwiGLU::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
        launch_swiglu_kernel(
        static_cast<const float*>(input.data),
        static_cast<float*>(output.data),
        context.batch->num_tokens,
        hidden_size
    );
} 