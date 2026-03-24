#include "LayerNorm.h"
#include "layernorm_kernel.h"

void LayerNorm::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context){

	const size_t hidden_size = config.norm_size;

	if (hidden_size == 0 
        || input.data == nullptr 
        || output.data == nullptr) {
		return;
	}

	const size_t num_tokens = context.batch->num_tokens;

	const float* gamma_ptr = static_cast<const float*>(gamma != nullptr ? gamma : norm_weight.data);

	launch_layernorm_kernel(
		static_cast<const float*>(input.data),
		gamma_ptr,
		static_cast<float*>(output.data),
		num_tokens,
		hidden_size,
		kDefaultEps
	);
}

void LayerNorm::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context){
	const size_t hidden_size = config.norm_size;

	if (hidden_size == 0 
        || input.data == nullptr 
        || output.data == nullptr) {
		return;
	}

	const size_t num_tokens = context.batch->num_tokens;

	const float* gamma_ptr = static_cast<const float*>(gamma != nullptr ? gamma : norm_weight.data);

	launch_layernorm_kernel(
		static_cast<const float*>(input.data),
		gamma_ptr,
		static_cast<float*>(output.data),
		num_tokens,
		hidden_size,
		kDefaultEps
	);
}