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

        void add_token(size_t token_id) {

            token_ids.push_back(token_id);
            if(!blocks.empty()){
                blocks.back()->token_ids.push_back(token_id);
            }
            seq_len++;

        }

        // sequence level metrics
        size_t submitted_time = 0;
        size_t first_token_time = 0;
        size_t last_token_time = 0;
        size_t generated_token_count = 0;
        size_t itl_sum = 0;
        size_t itl_count = 0;

};