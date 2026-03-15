#include "PostProcessor.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <numeric>
void PostProcessor::process(Tensor& input, ForwardContext& context) {
    if (context.batch == nullptr) {
        return;
    }

    const size_t vocab_size = config.vocab_size;
    const size_t seq_count = context.batch->batch_size;
    float* logits = static_cast<float*>(input.data);

    context.batch->sampled_token_ids.clear();
    context.batch->sampled_token_ids.reserve(seq_count);

    for (size_t seq_idx = 0; seq_idx < seq_count; ++seq_idx) {
        Tensor seq_logits(
            vocab_size,
            logits + seq_idx * vocab_size,
            {vocab_size},
            input.dtype
        );

        apply_temperature(seq_logits, seq_logits, config.temperature);

        std::vector<pair<size_t, float>> top_k_logits;
        top_k_logits.reserve(config.top_k);
        top_k(seq_logits, top_k_logits, config.top_k);

        std::vector<pair<size_t, float>> top_p_logits;
        top_p_logits.reserve(top_k_logits.size());
        top_p(top_k_logits, top_p_logits, config.top_p, config.top_k);

        apply_softmax(top_p_logits);

        size_t sampled_token = 0;
        sample(top_p_logits, sampled_token);
        context.batch->sampled_token_ids.push_back(sampled_token);
    }
}

void PostProcessor::apply_temperature(Tensor& input, Tensor& output, float temperature){
    const size_t vocab_size = config.vocab_size;
    for(int i=0; i < vocab_size; ++i){
        float logit = static_cast<float*>(input.data)[i];
        logit /= temperature;
        static_cast<float*>(output.data)[i] = logit;
    }
}
void PostProcessor::apply_repetition_penalty(
    Tensor& input, 
    Tensor& output, 
    const std::vector<size_t>& recent_token_ids, 
    float penalty
){
    const size_t vocab_size = config.vocab_size;
    for(int i=0; i < vocab_size; ++i){
        float logit = static_cast<float*>(input.data)[i];
        if(std::find(recent_token_ids.begin(), recent_token_ids.end(), i) != recent_token_ids.end()){
            logit /= penalty; // penalize repeated tokens
        }
        static_cast<float*>(output.data)[i] = logit;
    }
}
void PostProcessor::apply_softmax(std::vector<std::pair<size_t, float>>& input){
    size_t num_tokens = input.size();
    vector<float> logits(num_tokens);
    for(size_t i=0;i<num_tokens; ++i){
        logits[i] = input[i].second;
    }
    float max_logit = *std::max_element(logits.begin(), logits.end());
    for(size_t i=0; i<num_tokens; ++i){
        logits[i] = exp(logits[i] - max_logit);
    }
    float sum_exp = std::accumulate(logits.begin(), logits.end(), 0.0f);
    for(size_t i=0; i<num_tokens; ++i){
        input[i].second = logits[i] / sum_exp;
    }
}
void PostProcessor::top_k(Tensor& input, std::vector<std::pair<size_t, float>>& output, size_t top_k){
    const size_t vocab_size = config.vocab_size;
    std::vector<pair<size_t, float>> token_logits;
    for(int i=0; i < vocab_size; ++i){
        token_logits.emplace_back(i, static_cast<float*>(input.data)[i]);
    }
    std::sort(token_logits.begin(), token_logits.end(), token_compare);
    for(size_t i=0; i<top_k && i < token_logits.size(); ++i){
        output.push_back(token_logits[i]);
    }
}
void PostProcessor::top_p(std::vector<std::pair<size_t, float>>& input, std::vector<std::pair<size_t, float>>& output, float top_p, size_t top_k){
    if (input.empty()) {
        return;
    }

    const size_t limit = std::min(top_k, input.size());
    float cumulative = 0.0f;
    for (size_t i = 0; i < limit; ++i) {
        cumulative += input[i].second;
        output.push_back(input[i]);
        if (cumulative >= top_p) {
            break;
        }
    }
}

void PostProcessor::sample(std::vector<std::pair<size_t, float>>& input, size_t& sampled_token){
    size_t num_tokens = input.size();
    vector<float> probs(num_tokens);
    for(size_t i=0; i<num_tokens; ++i){
        probs[i] = input[i].second;
    }
    float sum_probs = 0.0f;
    float random_value = static_cast<float>(rand()) / RAND_MAX; // random value in [0, 1]
    for(size_t i=0; i<num_tokens; ++i){
        sum_probs += probs[i];
        if(random_value < sum_probs){
            sampled_token = input[i].first;
            break;
        }
    }
}