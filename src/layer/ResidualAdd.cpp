#include "ResidualAdd.h"
#include "residual_add_kernel.h"

void ResidualAdd::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    const size_t num_elements = input.numel();
    (void)context;
    launch_residual_add_kernel(
        static_cast<const float*>(output.data),
        static_cast<const float*>(input.data),
        static_cast<float*>(output.data),
        num_elements
    );
}

void ResidualAdd::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    prefill_forward(input, output, context);
}