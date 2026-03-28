#include "layer/MLP.h"
#include "cuda_runtime.h"
#include "utils/tensor_debug.h"

void MLP::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    if (linears.size() < 3 || context.batch == nullptr) {
        return;
    }

    const size_t num_tokens = context.batch->num_tokens;
    const size_t intermediate_size = mlp_config.mlp_linears[0].out_features;
    const size_t elem_bytes = Tensor::element_size_bytes(input.dtype);
    if (context.config && context.config->mlp_intermediate_size > 0
        && intermediate_size > context.config->mlp_intermediate_size) {
        LOG_ERROR("MLP intermediate_size exceeds workspace size from model config");
        return;
    }
    const std::string prefix = "layer=" + std::to_string(context.layer_id) + " stage=";

    // Gate view occupies the first half of workspace.
    Tensor gate_output;
    gate_output.data = context.workspace->get_mlp_workspace();
    gate_output.num_elements = num_tokens * intermediate_size;
    gate_output.size = gate_output.num_elements * elem_bytes;
    gate_output.shape = {num_tokens, intermediate_size};
    gate_output.dtype = input.dtype;
    gate_output.device = input.device;

    linears[0]->prefill_forward(input, gate_output, context);
    log_tensor_anomaly(gate_output, prefix + "mlp_gate");

    // Up view occupies the second half of workspace.
    Tensor up_output;
    up_output.data = static_cast<void*>(
        static_cast<char*>(gate_output.data) + gate_output.size
    );
    up_output.num_elements = num_tokens * intermediate_size;
    up_output.size = up_output.num_elements * elem_bytes;
    up_output.shape = {num_tokens, intermediate_size};
    up_output.dtype = input.dtype;
    up_output.device = input.device;

    linears[1]->prefill_forward(input, up_output, context);
    log_tensor_anomaly(up_output, prefix + "mlp_up");

    swiglu->forward(gate_output, up_output, gate_output, context);
    log_tensor_anomaly(gate_output, prefix + "mlp_swiglu");

    //output projection
    linears[2]->prefill_forward(gate_output, output, context);
    log_tensor_anomaly(output, prefix + "mlp_down");


}

void MLP::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
    if (linears.size() < 3 || context.batch == nullptr) {
        return;
    }

    const size_t num_tokens = context.batch->num_tokens;
    const size_t intermediate_size = mlp_config.mlp_linears[0].out_features;
    const size_t elem_bytes = Tensor::element_size_bytes(input.dtype);
    if (context.config && context.config->mlp_intermediate_size > 0
        && intermediate_size > context.config->mlp_intermediate_size) {
        LOG_ERROR("MLP intermediate_size exceeds workspace size from model config");
        return;
    }
   // const std::string prefix = "layer=" + std::to_string(context.layer_id) + " stage=";

    Tensor gate_output;
    gate_output.data = context.workspace->get_mlp_workspace();
    gate_output.num_elements = num_tokens * intermediate_size;
    gate_output.size = gate_output.num_elements * elem_bytes;
    gate_output.shape = {num_tokens, intermediate_size};
    gate_output.dtype = input.dtype;
    gate_output.device = input.device;

    linears[0]->decode_forward(input, gate_output, context);
   // log_tensor_anomaly(gate_output, prefix + "mlp_gate");

    Tensor up_output;
    up_output.data = static_cast<void*>(
        static_cast<char*>(gate_output.data) + gate_output.size
    );
    up_output.num_elements = num_tokens * intermediate_size;
    up_output.size = up_output.num_elements * elem_bytes;
    up_output.shape = {num_tokens, intermediate_size};
    up_output.dtype = input.dtype;
    up_output.device = input.device;

    linears[1]->decode_forward(input, up_output, context);
   // log_tensor_anomaly(up_output, prefix + "mlp_up");

    swiglu->forward(gate_output, up_output, gate_output, context);
   // log_tensor_anomaly(gate_output, prefix + "mlp_swiglu");

    linears[2]->decode_forward(gate_output, output, context);
   // log_tensor_anomaly(output, prefix + "mlp_down");
}

