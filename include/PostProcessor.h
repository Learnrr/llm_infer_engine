
#pragma once
#include "llm_engine_config.h"
#include "Tensor.h"
#include "ForwardContext.h"

class PostProcessor {
    public:
        PostProcessor(const LLMEngineConfig& config): config(config) {}
        void process(Tensor& input, ForwardContext& context);
    private:
        LLMEngineConfig config;
        void apply_temperature(Tensor& input, Tensor& output, float temperature);
        void apply_repetition_penalty(Tensor& input, Tensor& output, const std::vector<size_t>& recent_token_ids, float penalty);
        void apply_softmax(std::vector<std::pair<size_t, float>>& input);
        void top_k(Tensor& input, std::vector<std::pair<size_t, float>>& output, size_t top_k);
        void top_p(std::vector<std::pair<size_t, float>>& input, std::vector<std::pair<size_t, float>>& output, float top_p, size_t top_k);
        void sample(std::vector<std::pair<size_t, float>>& input, size_t& sampled_token);
        //utility functions
        static bool token_compare(const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second > b.second; // Sort in descending order of logit values
        }

};