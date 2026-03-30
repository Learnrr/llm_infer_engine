#include "PostProcessor.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <numeric>
#include "half_float/half.hpp"
#include "utils/logger.h"

using half_float::half;

namespace {

struct bfloat16_host {
    uint16_t bits;

    bfloat16_host() : bits(0) {}
    explicit bfloat16_host(float v) { *this = v; }

    bfloat16_host& operator=(float v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        // round-to-nearest-even before truncating low 16 bits
        const uint32_t lsb = (u >> 16) & 1U;
        u += 0x7FFFU + lsb;
        bits = static_cast<uint16_t>(u >> 16);
        return *this;
    }

    operator float() const {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        float v = 0.0f;
        std::memcpy(&v, &u, sizeof(v));
        return v;
    }
};

static_assert(sizeof(bfloat16_host) == sizeof(uint16_t), "bfloat16_host must be 16-bit");

template <typename T>
void apply_temperature_impl(
    const Tensor& input, 
    Tensor& output, 
    size_t vocab_size, 
    float temperature
) {
    const T* in = static_cast<const T*>(input.data);
    T* out = static_cast<T*>(output.data);
    const float safe_temperature = std::max(temperature, 1e-5f);
    // divide logits by temperature
    for (size_t i = 0; i < vocab_size; ++i) {
        float v = static_cast<float>(in[i]);
        v /= safe_temperature;
        out[i] = static_cast<T>(v);
    }
}

template <typename T>
void apply_repetition_penalty_impl(
    const Tensor& input,
    Tensor& output,
    size_t vocab_size,
    const std::vector<size_t>& recent_token_ids,
    float penalty
) {
    const T* in = static_cast<const T*>(input.data);
    T* out = static_cast<T*>(output.data);

    for (size_t i = 0; i < vocab_size; ++i) {
        float v = static_cast<float>(in[i]);
        if (std::find(recent_token_ids.begin(), recent_token_ids.end(), i) != recent_token_ids.end()) {
            v /= penalty;
        }
        out[i] = static_cast<T>(v);
    }
}

template <typename T>
void top_k_impl(
    const Tensor& input, 
    size_t vocab_size, 
    std::vector<std::pair<size_t, float>>& output, 
    size_t top_k
) {
    const T* in = static_cast<const T*>(input.data);
    std::vector<std::pair<size_t, float>> token_logits;
    token_logits.reserve(vocab_size);

    //token id and its logit value
    for (size_t i = 0; i < vocab_size; ++i) {
        token_logits.emplace_back(i, static_cast<float>(in[i]));
    }
    //sort in descending order on logit values
    std::sort(token_logits.begin(), token_logits.end(),
        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second > b.second;
        });

    output.clear();
    output.reserve(std::min(top_k, token_logits.size()));

    //take the top_k tokens
    for (size_t i = 0; i < top_k && i < token_logits.size(); ++i) {
        output.push_back(token_logits[i]);
    }
}

template <typename T>
size_t argmax_impl(const T* in, size_t vocab_size) {
    if (in == nullptr || vocab_size == 0) {
        return 0;
    }
    size_t best_idx = 0;
    float best_val = static_cast<float>(in[0]);
    for (size_t i = 1; i < vocab_size; ++i) {
        const float v = static_cast<float>(in[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return best_idx;
}

} // namespace

void PostProcessor::process(Tensor& input, ForwardContext& context) {
    if (context.batch == nullptr 
        || context.config == nullptr 
        || input.data == nullptr) {
        return;
    }

    const size_t vocab_size = config.model_config.vocab_size;
    const size_t seq_count = context.batch->batch_size;

    half* logits_f16 = nullptr;
    bfloat16_host* logits_bf16 = nullptr;
    float* logits_f32 = nullptr;
    if (input.dtype == DataType::FLOAT16) {
        logits_f16 = static_cast<half*>(input.data);
    } else if (input.dtype == DataType::BF16) {
        logits_bf16 = static_cast<bfloat16_host*>(input.data);
    } else if (input.dtype == DataType::FLOAT32) {
        logits_f32 = static_cast<float*>(input.data);
    } else {
        return;
    }

    context.batch->sampled_token_ids.clear();
    context.batch->sampled_token_ids.reserve(seq_count);

    for (size_t seq_idx = 0; seq_idx < seq_count; ++seq_idx) {
        void* seq_ptr = nullptr;
        if (input.dtype == DataType::FLOAT16) {
            seq_ptr = static_cast<void*>(logits_f16 + seq_idx * vocab_size);
        } else if (input.dtype == DataType::BF16) {
            seq_ptr = static_cast<void*>(logits_bf16 + seq_idx * vocab_size);
        } else if(input.dtype == DataType::FLOAT32) {
            seq_ptr = static_cast<void*>(logits_f32 + seq_idx * vocab_size);
        } else {
            return;
        }

        // greedy decode: choose argmax directly 
        if(config.greedy_decode) {
            size_t sampled_token = 0;
            if (input.dtype == DataType::FLOAT16) {
                sampled_token = argmax_impl(logits_f16 + seq_idx * vocab_size, vocab_size);
            } else if (input.dtype == DataType::BF16) {
                sampled_token = argmax_impl(logits_bf16 + seq_idx * vocab_size, vocab_size);
            } else if (input.dtype == DataType::FLOAT32) {
                sampled_token = argmax_impl(logits_f32 + seq_idx * vocab_size, vocab_size);
            }
            context.batch->sampled_token_ids.push_back(sampled_token);
            continue;
        }

        Tensor seq_logits(vocab_size, seq_ptr, {vocab_size}, input.dtype, input.device);
        apply_temperature(seq_logits, seq_logits, config.temperature);
        std::vector<std::pair<size_t, float>> top_k_logits;
        top_k(seq_logits, top_k_logits, config.top_k);
        apply_softmax(top_k_logits);

        const float top1_prob = top_k_logits.empty() ? 0.0f : top_k_logits.front().second;
        const size_t top1_token = top_k_logits.empty() ? 0 : top_k_logits.front().first;

        std::vector<std::pair<size_t, float>> top_p_logits;
        top_p(top_k_logits, top_p_logits, config.top_p, config.top_k);

        if (top_p_logits.empty() && !top_k_logits.empty()) {
            top_p_logits.push_back(top_k_logits.front());
            top_p_logits.front().second = 1.0f;
        }

        size_t sampled_token = 0;
        sample(top_p_logits, sampled_token);
        LOG_DEBUG(
            "PostProcessor seq_idx=" + std::to_string(seq_idx) +
            " top1_token=" + std::to_string(top1_token) +
            " top1_prob=" + std::to_string(top1_prob) +
            " top_p_candidates=" + std::to_string(top_p_logits.size()) +
            " sampled_token=" + std::to_string(sampled_token)
        );
        context.batch->sampled_token_ids.push_back(sampled_token);
    }
}

void PostProcessor::apply_temperature(Tensor& input, Tensor& output, float temperature) {
    const size_t vocab_size = config.model_config.vocab_size;
    if (input.dtype == DataType::FLOAT16) {
        apply_temperature_impl<half>(input, output, vocab_size, temperature);
        return;
    }
    if (input.dtype == DataType::BF16) {
        apply_temperature_impl<bfloat16_host>(input, output, vocab_size, temperature);
        return;
    }
    apply_temperature_impl<float>(input, output, vocab_size, temperature);
}

void PostProcessor::apply_repetition_penalty(
    Tensor& input,
    Tensor& output,
    const std::vector<size_t>& recent_token_ids,
    float penalty
) {
    const size_t vocab_size = config.model_config.vocab_size;
    if (input.dtype == DataType::FLOAT16) {
        apply_repetition_penalty_impl<half>(input, output, vocab_size, recent_token_ids, penalty);
        return;
    }
    if (input.dtype == DataType::BF16) {
        apply_repetition_penalty_impl<bfloat16_host>(input, output, vocab_size, recent_token_ids, penalty);
        return;
    }
    apply_repetition_penalty_impl<float>(input, output, vocab_size, recent_token_ids, penalty);
}

void PostProcessor::apply_softmax(std::vector<std::pair<size_t, float>>& input) {
    if (input.empty()) {
        return;
    }

    float max_logit = input.front().second;
    for (const auto& kv : input) {
        max_logit = std::max(max_logit, kv.second);
    }

    float sum_exp = 0.0f;
    for (auto& kv : input) {
        kv.second = std::exp(kv.second - max_logit);
        sum_exp += kv.second;
    }

    if (sum_exp <= 0.0f) {
        return;
    }

    for (auto& kv : input) {
        kv.second /= sum_exp;
    }
}

void PostProcessor::top_k(Tensor& input, std::vector<std::pair<size_t, float>>& output, size_t top_k) {
    const size_t vocab_size = config.model_config.vocab_size;
    if (input.dtype == DataType::FLOAT16) {
        top_k_impl<half>(input, vocab_size, output, top_k);
        return;
    }
    if (input.dtype == DataType::BF16) {
        top_k_impl<bfloat16_host>(input, vocab_size, output, top_k);
        return;
    }
    top_k_impl<float>(input, vocab_size, output, top_k);
}

void PostProcessor::top_p(
    std::vector<std::pair<size_t, float>>& input,
    std::vector<std::pair<size_t, float>>& output,
    float top_p,
    size_t top_k
) {
    if (input.empty()) {
        return;
    }

    const size_t limit = std::min(top_k, input.size());
    float cumulative = 0.0f;

    output.clear();
    output.reserve(limit);

    for (size_t i = 0; i < limit; ++i) {
        cumulative += input[i].second;
        output.push_back(input[i]);
        if (cumulative >= top_p) {
            break;
        }
    }

    float sum = 0.0f;
    for (const auto& kv : output) {
        sum += kv.second;
    }

    if (sum > 0.0f) {
        for (auto& kv : output) {
            kv.second /= sum;
        }
    }
}

void PostProcessor::sample(
    std::vector<std::pair<size_t, 
    float>>& input, 
    size_t& sampled_token
) {
    if (input.empty()) {
        sampled_token = 0;
        return;
    }

    float cdf = 0.0f;
    const float r = static_cast<float>(rand()) / RAND_MAX;
    for (const auto& kv : input) {
        cdf += kv.second;
        if (r < cdf) {
            sampled_token = kv.first;
            return;
        }
    }

    sampled_token = input.back().first;
}
