#include "Linear.h"
#include "cuda_runtime.h"
#include "linear_kernel.h"

void Linear::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    size_t batch_seq_len = context.batch->num_tokens;
    launch_mlp_linear_kernel(
        static_cast<const float*>(input.data),
        static_cast<const float*>(linear_weight.data),
        static_cast<float*>(output.data),
        batch_seq_len,
        config.in_features,
        config.out_features
    );
}

void Linear::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    size_t batch_seq_len = context.batch->num_tokens;
    launch_mlp_linear_kernel(
        static_cast<const float*>(input.data),
        static_cast<const float*>(linear_weight.data),
        static_cast<float*>(output.data),
        batch_seq_len,
        config.in_features,
        config.out_features
    );
}
