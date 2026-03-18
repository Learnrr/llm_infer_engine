#include "TransformerLayer.h"
#include <cuda_runtime.h>

void TransformerLayer::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) {
    Tensor attn_output(input.numel(), nullptr, input.shape, input.dtype);
    attn_output.data = context.workspace->get_attn_output_workspace();

    Tensor attn_residual(input.numel(), nullptr, input.shape, input.dtype);
    attn_residual.data = context.workspace->get_temp_workspace();

    Tensor mlp_output(input.numel(), nullptr, input.shape, input.dtype);
    mlp_output.data = context.workspace->get_mlp_workspace();

    attention->prefill_forward(input, attn_output, context);
    cudaMemcpy(attn_residual.data, input.data, input.size, cudaMemcpyDeviceToDevice);
    residual_add->prefill_forward(attn_output, attn_residual, context);

    mlp->prefill_forward(attn_residual, mlp_output, context);
    cudaMemcpy(output.data, attn_residual.data, attn_residual.size, cudaMemcpyDeviceToDevice);
    residual_add->prefill_forward(mlp_output, output, context);
}

void TransformerLayer::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) {
    Tensor attn_output(input.numel(), nullptr, input.shape, input.dtype);
    attn_output.data = context.workspace->get_attn_output_workspace();

    Tensor attn_residual(input.numel(), nullptr, input.shape, input.dtype);
    attn_residual.data = context.workspace->get_temp_workspace();

    Tensor mlp_output(input.numel(), nullptr, input.shape, input.dtype);
    mlp_output.data = context.workspace->get_mlp_workspace();

    attention->decode_forward(input, attn_output, context);
    cudaMemcpy(attn_residual.data, input.data, input.size, cudaMemcpyDeviceToDevice);
    residual_add->decode_forward(attn_output, attn_residual, context);

    mlp->decode_forward(attn_residual, mlp_output, context);
    cudaMemcpy(output.data, attn_residual.data, attn_residual.size, cudaMemcpyDeviceToDevice);
    residual_add->decode_forward(mlp_output, output, context);
}