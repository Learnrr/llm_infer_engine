#include "RMSNorm.h"
#include "rmsnorm_kernel.h"

void RMSNorm::prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) {

	size_t hidden_size = config.norm_size;
	if (hidden_size == 0 && !input.shape.empty()) {
		hidden_size = input.shape.back();
	}

	if (hidden_size == 0 || input.data == nullptr || output.data == nullptr) {
		return;
	}

	size_t num_tokens = context.batch->num_tokens;

	const float* gamma = static_cast<const float*>(norm_weight.gamma);
	

	launch_rmsnorm_kernel(
		static_cast<const float*>(input.data),
		gamma,
		static_cast<float*>(output.data),
		num_tokens,
		hidden_size,
		kDefaultEps
	);
}

void RMSNorm::decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) {
	size_t hidden_size = config.norm_size;
	if (hidden_size == 0 && !input.shape.empty()) {
		hidden_size = input.shape.back();
	}

	if (hidden_size == 0 || input.data == nullptr || output.data == nullptr) {
		return;
	}

	size_t num_tokens = context.batch->num_tokens;

	float* gamma = static_cast<const float*>(norm_weight.gamma);
	

	launch_rmsnorm_kernel(
		static_cast<const float*>(input.data),
		gamma,
		static_cast<float*>(output.data),
		num_tokens,
		hidden_size,
		kDefaultEps
	);
}
