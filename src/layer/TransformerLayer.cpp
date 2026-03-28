#include "layer/TransformerLayer.h"
#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include "utils/tensor_debug.h"


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
    mlp_output.data = context.workspace->get_mlp_out_workspace();
    size_t layer_id = context.layer_id;
    auto make_desc = [&](const char* stage) {
        return std::string("layer=") + std::to_string(layer_id) + " stage=" + stage;
    };
    norm_layers[0]->prefill_forward(input, norm_output, context);
    //const std::string prefill_norm1 = make_desc("prefill_after_norm1");
    //log_tensor_anomaly(norm_output, prefill_norm1);
    attention->prefill_forward(norm_output, attn_output, context);
    //const std::string prefill_attn = make_desc("prefill_after_attention");
   // log_tensor_anomaly(attn_output, prefill_attn);
    cudaMemcpy(attn_residual.data, input.data, input.size, cudaMemcpyDeviceToDevice);
    residual_add->prefill_forward(attn_output, attn_residual, context);
   // const std::string prefill_attn_residual = make_desc("prefill_after_attn_residual");
    //log_tensor_anomaly(attn_residual, prefill_attn_residual);

    norm_layers[1]->prefill_forward(attn_residual, norm_output, context);
   // const std::string prefill_norm2 = make_desc("prefill_after_norm2");
    //log_tensor_anomaly(norm_output, prefill_norm2);
    mlp->prefill_forward(norm_output, mlp_output, context);
   // const std::string prefill_mlp = make_desc("prefill_after_mlp");
    //log_tensor_anomaly(mlp_output, prefill_mlp);
    cudaMemcpy(output.data, attn_residual.data, attn_residual.size, cudaMemcpyDeviceToDevice);
    residual_add->prefill_forward(mlp_output, output, context);
   // const std::string prefill_out = make_desc("prefill_after_output_residual");
    //log_tensor_anomaly(output, prefill_out);
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
    mlp_output.data = context.workspace->get_mlp_out_workspace();

    const size_t layer_id = context.layer_id;
    auto make_desc = [&](const char* stage) {
        return std::string("layer=") + std::to_string(layer_id) + " stage=" + stage;
    };

    norm_layers[0]->decode_forward(input, norm_output, context);
   // const std::string decode_norm1 = make_desc("decode_after_norm1");
    //log_tensor_anomaly(norm_output, decode_norm1);

    attention->decode_forward(norm_output, attn_output, context);
    //const std::string decode_attn = make_desc("decode_after_attention");
   // log_tensor_anomaly(attn_output, decode_attn);

    cudaMemcpy(attn_residual.data, input.data, input.size, cudaMemcpyDeviceToDevice);
    residual_add->decode_forward(attn_output, attn_residual, context);
   // const std::string decode_attn_residual = make_desc("decode_after_attn_residual");
    //log_tensor_anomaly(attn_residual, decode_attn_residual);

    norm_layers[1]->decode_forward(attn_residual, norm_output, context);
    //const std::string decode_norm2 = make_desc("decode_after_norm2");
    //log_tensor_anomaly(norm_output, decode_norm2);

    mlp->decode_forward(norm_output, mlp_output, context);
   // const std::string decode_mlp = make_desc("decode_after_mlp");
    //log_tensor_anomaly(mlp_output, decode_mlp);

    cudaMemcpy(output.data, attn_residual.data, attn_residual.size, cudaMemcpyDeviceToDevice);
    residual_add->decode_forward(mlp_output, output, context);
    //const std::string decode_out = make_desc("decode_after_output_residual");
    //log_tensor_anomaly(output, decode_out);
}