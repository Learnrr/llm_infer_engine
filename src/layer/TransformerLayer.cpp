#include "TransformerLayer.h"
#include <cuda_runtime.h>

void TransformerLayer::prefill_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) {
    if (!attention || !mlp || !residual_add || norm_layers.size() < 2) {
        LOG_ERROR("TransformerLayer prefill_forward called with incomplete initialization");
        return;
    }

    Tensor norm_output(input.numel(), nullptr, input.shape, input.dtype);
    norm_output.data = context.workspace->get_attn_norm_workspace();

    Tensor attn_output(norm_output.numel(), nullptr, input.shape, input.dtype);
    attn_output.data = context.workspace->get_attn_output_workspace();

    Tensor attn_residual(norm_output.numel(), nullptr, input.shape, input.dtype);
    attn_residual.data = context.workspace->get_temp_workspace();

    Tensor mlp_output(input.numel(), nullptr, input.shape, input.dtype);
    mlp_output.data = context.workspace->get_mlp_workspace();

    norm_layers[0]->prefill_forward(input, norm_output, context);
    attention->prefill_forward(norm_output, attn_output, context);
    cudaMemcpy(attn_residual.data, input.data, input.size, cudaMemcpyDeviceToDevice);
    residual_add->prefill_forward(attn_output, attn_residual, context);
    
    norm_layers[1]->prefill_forward(attn_residual, norm_output, context);
    mlp->prefill_forward(norm_output, mlp_output, context);
    cudaMemcpy(output.data, attn_residual.data, attn_residual.size, cudaMemcpyDeviceToDevice);
    residual_add->prefill_forward(mlp_output, output, context);
}

void TransformerLayer::decode_forward(
    const Tensor& input, 
    Tensor& output, 
    ForwardContext& context
) {
    if (!attention || !mlp || !residual_add || norm_layers.size() < 2) {
        LOG_ERROR("TransformerLayer decode_forward called with incomplete initialization");
        return;
    }

    Tensor norm_output(input.numel(), nullptr, input.shape, input.dtype);
    norm_output.data = context.workspace->get_attn_norm_workspace();

    Tensor attn_output(input.numel(), nullptr, input.shape, input.dtype);
    attn_output.data = context.workspace->get_attn_output_workspace();

    Tensor attn_residual(input.numel(), nullptr, input.shape, input.dtype);
    attn_residual.data = context.workspace->get_temp_workspace();

    Tensor mlp_output(input.numel(), nullptr, input.shape, input.dtype);
    mlp_output.data = context.workspace->get_mlp_workspace();

    norm_layers[0]->decode_forward(input, norm_output, context);
    attention->decode_forward(norm_output, attn_output, context);
    cudaMemcpy(attn_residual.data, input.data, input.size, cudaMemcpyDeviceToDevice);
    residual_add->decode_forward(attn_output, attn_residual, context);

    norm_layers[1]->decode_forward(attn_residual, norm_output, context);
    mlp->decode_forward(norm_output, mlp_output, context);
    cudaMemcpy(output.data, attn_residual.data, attn_residual.size, cudaMemcpyDeviceToDevice);
    residual_add->decode_forward(mlp_output, output, context);
}