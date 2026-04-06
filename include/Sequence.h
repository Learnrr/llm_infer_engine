#pragma once
#include<vector>
#include"Cacheblock.h"
#include "define.h"
#include <mutex>
#include <condition_variable>
#include <cstddef>

enum class SequenceState {
    PREPARED,
    WAITING,
    PREFILLING,
    PREFILLED,
    DECODING,
    FINISHED
};

class SequenceConfig {
    public:
        float temperature = 1.0f;
        float top_p = 1.0f;
        int top_k = 50;
        size_t max_tokens = 128;
        float presence_penalty = 0.0f;
        float frequency_penalty = 0.0f;
};

class Sequence {
    public:
        size_t seq_id;
        size_t seq_len;
        std::vector<size_t> token_ids;
        SequenceState state;
        std::vector<std::shared_ptr<CacheBlock>> blocks;
        bool finish_handled = false;

        std::mutex mtx;
        std::condition_variable cv;


        Sequence(size_t seq_id) : seq_id(seq_id), seq_len(0) {}
        Sequence(size_t seq_id, SequenceConfig config) : 
            seq_id(seq_id), seq_len(0), seq_config(config) {}

        void add_token(size_t token_id) {

            token_ids.push_back(token_id);
            if(!blocks.empty()){
                blocks.back()->token_ids.push_back(token_id);
            }
            seq_len++;

        }
        //sequence level generation parameters
        SequenceConfig seq_config;

        // sequence level metrics
        size_t submitted_time = 0;
        size_t first_token_time = 0;
        size_t last_token_time = 0;
        size_t generated_token_count = 0;
        size_t itl_sum = 0;
        size_t itl_count = 0;

};