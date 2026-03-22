#pragma once
#include"Tensor.h"

class KVCache {
    public:
        Tensor key;
        Tensor value;

        size_t seq_len;
        size_t head_num;
        size_t head_size;
        size_t max_seq_len;

        KVCache(size_t seq_len, size_t head_num, size_t head_size, size_t max_seq_len) : 
            seq_len(seq_len), 
            head_num(head_num), 
            head_size(head_size), 
            max_seq_len(max_seq_len),
            key(max_seq_len * head_num * head_size, nullptr, {max_seq_len, head_num, head_size}, DataType::FLOAT16),
            value(max_seq_len * head_num * head_size, nullptr, {max_seq_len, head_num, head_size}, DataType::FLOAT16) {}

        
};