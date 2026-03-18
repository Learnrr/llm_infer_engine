#pragma once

#include "ForwardContext.h"
#include "Tensor.h"

class RoPE {
public:
    RoPE() = default;
	void apply(
		Tensor& q,
		Tensor& k,
		const ForwardContext& context,
		size_t num_q_heads,
		size_t num_kv_heads,
		size_t head_dim,
		float rope_theta = 10000.0f
	) const;
};

