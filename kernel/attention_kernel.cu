/*
fused attention cuda kernel, including read kv from blocked cache through config, compute qk,
softmax, and compute pv in one kernel, and write output to attn_output.
*/
#include "Tensor.h"

void __global__ attention_qk_softmax_Pv_kernel(
    const void* q, 
    size_t* block_ids, 
    size_t* block_offsets, 
    void* attn_output
){
    // Implement the fused attention kernel logic here

}