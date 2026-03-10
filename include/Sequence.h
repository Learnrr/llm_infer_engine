#pragma once
#include<vector>
#include"CacheBlock.h"
#include "define.h"
#include "Engine.h"

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
        Sequence(size_t seq_len) : seq_len(seq_len) {}

        void add_token(size_t token_id, size_t position) {

            if(position % BLOCK_SIZE == 0) {
                size_t block_id = position / BLOCK_SIZE;
                CacheBlock* block = Engine::get_instance()->cache_manager->allocate_cache_block();
                if (block) {
                    blocks.push_back(block);
                } else {
                    //failure handling for cache block allocation
                }
            }

            token_ids.push_back(token_id);
            blocks.back()->token_ids.push_back(token_id);
            seq_len++;

        }

}