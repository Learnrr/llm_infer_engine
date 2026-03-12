#pragma once
#include<vector>
#include"CacheBlock.h"
#include "define.h"
#include "Engine.h"
#include <mutex>
#include <condition_variable>

enum class SequenceState {
    PREPARE,
    READY,
    PREFILLING,
    PREFILLED,
    DECODING,
    FINISHED
};

class Sequence {
    public:
        size_t sequence_id;
        size_t seq_len;
        vector<size_t> token_ids;
        SequenceState state;
        vector<CacheBlock*> blocks;

        std::mutex mtx;
        std::condition_variable cv;

        Sequence(size_t seq_len) : seq_len(seq_len) {}

        void add_token(size_t token_id) {

            token_ids.push_back(token_id);
            blocks.back()->token_ids.push_back(token_id);
            seq_len++;

        }

}