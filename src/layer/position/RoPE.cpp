#include "layer/position/RoPE.h"
#include "rope_kernel.h"
#include <cuda_runtime.h>
#include "utils/logger.h"
#include "error.h"

void RoPE::apply(
	Tensor& q,
	Tensor& k,
	const ForwardContext& context,
	size_t num_q_heads,
	size_t num_kv_heads,
	size_t head_dim,
	float rope_theta
) const {
	if (context.batch == nullptr) {
		return;
	}

	const size_t num_tokens = q.shape.empty() ? 0 : q.shape[0];
	if (num_tokens == 0 || head_dim == 0 || (head_dim % 2) != 0) {
		return;
	}
	if (context.batch->token_positions.size() < num_tokens) {
		return;
	}

	size_t* d_positions = nullptr;
	cudaError_t cuda_err;
	cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_positions), num_tokens * sizeof(size_t));
	if (cuda_err != cudaSuccess) {
		LOG_ERROR("Failed to allocate device memory for token positions");
		return;
	}
	cuda_err = cudaMemcpy(
		d_positions,
		context.batch->token_positions.data(),
		num_tokens * sizeof(size_t),
		cudaMemcpyHostToDevice
	);
	if (cuda_err != cudaSuccess) {
		LOG_ERROR("Failed to copy token positions to device");
		cudaFree(d_positions);
		return;
	}

	if (q.dtype == DataType::FLOAT32) {
		launch_apply_rope_inplace_float(
			static_cast<float*>(q.data),
			d_positions,
			num_tokens,
			num_q_heads,
			head_dim,
			rope_theta
		);
	} else if (q.dtype == DataType::FLOAT16) {
		launch_apply_rope_inplace_half(
			static_cast<uint16_t*>(q.data),
			d_positions,
			num_tokens,
			num_q_heads,
			head_dim,
			rope_theta
		);
	}

	if (k.dtype == DataType::FLOAT32) {
		launch_apply_rope_inplace_float(
			static_cast<float*>(k.data),
			d_positions,
			num_tokens,
			num_kv_heads,
			head_dim,
			rope_theta
		);
	} else if (k.dtype == DataType::FLOAT16) {
		launch_apply_rope_inplace_half(
			static_cast<uint16_t*>(k.data),
			d_positions,
			num_tokens,
			num_kv_heads,
			head_dim,
			rope_theta
		);
	}

	cudaFree(d_positions);
}

