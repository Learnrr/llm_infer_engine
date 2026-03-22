/*
cd tests
nvcc -std=c++17 -O2 -DBLOCK_SIZE=16 -I../include -I../include/layer -I \
    ../include/model -I../include/kernel test_attention.cpp  \
    ../src/layer/Attention.cpp ../src/layer/position/RoPE.cpp  \
    ../src/Workspace.cpp ../kernel/projection.cu ../kernel/attention_kernel.cu  \
    ../kernel/output_projection_kernel.cu ../kernel/write_kvcache_kernel.cu  \
    ../kernel/rope_kernel.cu -o ../build/tests/test_attention.exe
./../build/tests/test_attention.exe
*/

#include "layer/Attention.h"
#include "Workspace.h"
#include "llm_engine_config.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) && (count > 0);
}

void CheckCuda(cudaError_t err) {
    assert(err == cudaSuccess);
}

bool AlmostEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

LLMEngineConfig BuildTinyEngineConfig() {
    LLMEngineConfig cfg(
        1,      // max_batch_size
        16,     // max_sequence_length
        1 << 20, // total_cache_size
        16      // block_size
    );

    cfg.model_config.max_seq_len = 16;
    cfg.model_config.hidden_size = 4;
    cfg.model_config.num_hidden_layers = 1;
    cfg.model_config.vocab_size = 32;
    cfg.model_config.num_heads = 1;
    cfg.model_config.num_kv_heads = 1;
    cfg.model_config.head_dim = 4;
    cfg.model_config.data_type = DataType::FLOAT32;
    cfg.model_config.mlp_intermediate_size = 8;
    return cfg;
}

void TestAttentionPrefillForwardWritesCacheAndOutput() {
    LLMEngineConfig engine_cfg = BuildTinyEngineConfig();

    Workspace workspace;
    ErrorCode ws_err = workspace.init(engine_cfg);
    assert(ws_err == ErrorCode::SUCCESS);

    // Input [1, 0, 0, 0] so projection output equals weight row0.
    const std::vector<float> h_input = {1.0f, 0.0f, 0.0f, 0.0f};
    float* d_input = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    // qkv_proj shape = [in_features=4, out_features=12].
    // row0 -> q=[1,1,1,1], k=[1,1,1,1], v=[2,3,4,5], other rows -> zeros.
    // With one token, attention output should be exactly v=[2,3,4,5].
    const std::vector<float> h_qkv_weight = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float* d_qkv_weight = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_qkv_weight), h_qkv_weight.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(
        d_qkv_weight,
        h_qkv_weight.data(),
        h_qkv_weight.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Identity output projection for [4]-dim hidden state.
    const std::vector<float> h_o_weight = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float* d_o_weight = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_o_weight), h_o_weight.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_o_weight, h_o_weight.data(), h_o_weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate one cache block in device memory: [block_size, layers, kv_heads, head_dim].
    const size_t block_size = 16;
    const size_t cache_elems = block_size
        * engine_cfg.model_config.num_hidden_layers
        * engine_cfg.model_config.num_kv_heads
        * engine_cfg.model_config.head_dim;

    float* d_kcache = nullptr;
    float* d_vcache = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_kcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_vcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_kcache, 0, cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_vcache, 0, cache_elems * sizeof(float)));

    auto seq = std::make_shared<Sequence>(0);
    seq->seq_len = 1;
    seq->blocks.push_back(std::make_shared<CacheBlock>(0, d_kcache, d_vcache));

    Batch batch;
    batch.sequences.push_back(seq);
    batch.num_tokens = 1;
    batch.batch_size = 1;
    batch.token_positions = {0};
    batch.token_ids = {42};

    ForwardContext context;
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &engine_cfg.model_config;

    AttentionLayerConfig attn_cfg;
    attn_cfg.num_attention_heads = 1;
    attn_cfg.num_kv_heads = 1;
    attn_cfg.head_dim = 4;

    AttentionLayerWeightLayout attn_layout;
    attn_layout.qkv_proj_weight.data = d_qkv_weight;
    attn_layout.qkv_proj_weight.num_elements = h_qkv_weight.size();
    attn_layout.qkv_proj_weight.size = h_qkv_weight.size() * sizeof(float);
    attn_layout.qkv_proj_weight.shape = {4, 12};
    attn_layout.qkv_proj_weight.dtype = DataType::FLOAT32;
    attn_layout.qkv_proj_weight.device = "gpu";

    attn_layout.o_proj_weight.data = d_o_weight;
    attn_layout.o_proj_weight.num_elements = h_o_weight.size();
    attn_layout.o_proj_weight.size = h_o_weight.size() * sizeof(float);
    attn_layout.o_proj_weight.shape = {4, 4};
    attn_layout.o_proj_weight.dtype = DataType::FLOAT32;
    attn_layout.o_proj_weight.device = "gpu";

    Attention attention(attn_cfg, attn_layout);

    Tensor input;
    input.data = d_input;
    input.num_elements = h_input.size();
    input.size = h_input.size() * sizeof(float);
    input.shape = {1, 4};
    input.dtype = DataType::FLOAT32;
    input.device = "gpu";

    Tensor output;
    output.data = workspace.get_attn_output_workspace();
    output.size = 4 * sizeof(float);
    output.shape = {1, 4};
    output.dtype = DataType::FLOAT32;
    output.device = "gpu";
    output.num_elements = 4;
    CheckCuda(cudaMemset(output.data, 0, output.size));

    attention.prefill_forward(input, output, context);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_out(4, 0.0f);
    CheckCuda(cudaMemcpy(h_out.data(), output.data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    assert(AlmostEqual(h_out[0], 2.0f));
    assert(AlmostEqual(h_out[1], 3.0f));
    assert(AlmostEqual(h_out[2], 4.0f));
    assert(AlmostEqual(h_out[3], 5.0f));

    std::vector<float> h_k_written(4, 0.0f);
    std::vector<float> h_v_written(4, 0.0f);
    CheckCuda(cudaMemcpy(h_k_written.data(), d_kcache, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(h_v_written.data(), d_vcache, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    assert(AlmostEqual(h_k_written[0], 1.0f));
    assert(AlmostEqual(h_k_written[1], 1.0f));
    assert(AlmostEqual(h_k_written[2], 1.0f));
    assert(AlmostEqual(h_k_written[3], 1.0f));
    assert(AlmostEqual(h_v_written[0], 2.0f));
    assert(AlmostEqual(h_v_written[1], 3.0f));
    assert(AlmostEqual(h_v_written[2], 4.0f));
    assert(AlmostEqual(h_v_written[3], 5.0f));

    cudaFree(d_input);
    cudaFree(d_qkv_weight);
    cudaFree(d_o_weight);
    cudaFree(d_kcache);
    cudaFree(d_vcache);
}

void TestAttentionDecodeForwardWritesCacheAndOutput() {
    LLMEngineConfig engine_cfg = BuildTinyEngineConfig();

    Workspace workspace;
    ErrorCode ws_err = workspace.init(engine_cfg);
    assert(ws_err == ErrorCode::SUCCESS);

    const std::vector<float> h_input = {1.0f, 0.0f, 0.0f, 0.0f};
    float* d_input = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    const std::vector<float> h_qkv_weight = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float* d_qkv_weight = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_qkv_weight), h_qkv_weight.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(
        d_qkv_weight,
        h_qkv_weight.data(),
        h_qkv_weight.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const std::vector<float> h_o_weight = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float* d_o_weight = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_o_weight), h_o_weight.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_o_weight, h_o_weight.data(), h_o_weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    const size_t block_size = 16;
    const size_t cache_elems = block_size
        * engine_cfg.model_config.num_hidden_layers
        * engine_cfg.model_config.num_kv_heads
        * engine_cfg.model_config.head_dim;

    float* d_kcache = nullptr;
    float* d_vcache = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_kcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_vcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_kcache, 0, cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_vcache, 0, cache_elems * sizeof(float)));

    auto seq = std::make_shared<Sequence>(1);
    seq->seq_len = 1;
    seq->blocks.push_back(std::make_shared<CacheBlock>(1, d_kcache, d_vcache));

    Batch batch;
    batch.sequences.push_back(seq);
    batch.num_tokens = 1;
    batch.batch_size = 1;
    batch.token_positions = {0};
    batch.token_ids = {99};

    ForwardContext context;
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &engine_cfg.model_config;

    AttentionLayerConfig attn_cfg;
    attn_cfg.num_attention_heads = 1;
    attn_cfg.num_kv_heads = 1;
    attn_cfg.head_dim = 4;

    AttentionLayerWeightLayout attn_layout;
    attn_layout.qkv_proj_weight.data = d_qkv_weight;
    attn_layout.qkv_proj_weight.num_elements = h_qkv_weight.size();
    attn_layout.qkv_proj_weight.size = h_qkv_weight.size() * sizeof(float);
    attn_layout.qkv_proj_weight.shape = {4, 12};
    attn_layout.qkv_proj_weight.dtype = DataType::FLOAT32;
    attn_layout.qkv_proj_weight.device = "gpu";

    attn_layout.o_proj_weight.data = d_o_weight;
    attn_layout.o_proj_weight.num_elements = h_o_weight.size();
    attn_layout.o_proj_weight.size = h_o_weight.size() * sizeof(float);
    attn_layout.o_proj_weight.shape = {4, 4};
    attn_layout.o_proj_weight.dtype = DataType::FLOAT32;
    attn_layout.o_proj_weight.device = "gpu";

    Attention attention(attn_cfg, attn_layout);

    Tensor input;
    input.data = d_input;
    input.num_elements = h_input.size();
    input.size = h_input.size() * sizeof(float);
    input.shape = {1, 4};
    input.dtype = DataType::FLOAT32;
    input.device = "gpu";

    Tensor output;
    output.data = workspace.get_attn_output_workspace();
    output.size = 4 * sizeof(float);
    output.shape = {1, 4};
    output.dtype = DataType::FLOAT32;
    output.device = "gpu";
    output.num_elements = 4;
    CheckCuda(cudaMemset(output.data, 0, output.size));

    attention.decode_forward(input, output, context);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_out(4, 0.0f);
    CheckCuda(cudaMemcpy(h_out.data(), output.data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    assert(AlmostEqual(h_out[0], 2.0f));
    assert(AlmostEqual(h_out[1], 3.0f));
    assert(AlmostEqual(h_out[2], 4.0f));
    assert(AlmostEqual(h_out[3], 5.0f));

    std::vector<float> h_v_written(4, 0.0f);
    CheckCuda(cudaMemcpy(h_v_written.data(), d_vcache, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    assert(AlmostEqual(h_v_written[0], 2.0f));
    assert(AlmostEqual(h_v_written[1], 3.0f));
    assert(AlmostEqual(h_v_written[2], 4.0f));
    assert(AlmostEqual(h_v_written[3], 5.0f));

    cudaFree(d_input);
    cudaFree(d_qkv_weight);
    cudaFree(d_o_weight);
    cudaFree(d_kcache);
    cudaFree(d_vcache);
}

void TestAttentionMultiTokenPrefillAndDecode() {
    LLMEngineConfig engine_cfg = BuildTinyEngineConfig();

    Workspace workspace;
    ErrorCode ws_err = workspace.init(engine_cfg);
    assert(ws_err == ErrorCode::SUCCESS);

    // 两个token，输入分别为[1,0,0,0]和[0,1,0,0]
    const std::vector<float> h_input = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };
    float* d_input = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    // qkv_proj: row0 -> q=[1,1,1,1], k=[1,1,1,1], v=[2,3,4,5]
    //           row1 -> q=[2,2,2,2], k=[2,2,2,2], v=[3,4,5,6]
    // 其余为0
    const std::vector<float> h_qkv_weight = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float* d_qkv_weight = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_qkv_weight), h_qkv_weight.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(
        d_qkv_weight,
        h_qkv_weight.data(),
        h_qkv_weight.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const std::vector<float> h_o_weight = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float* d_o_weight = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_o_weight), h_o_weight.size() * sizeof(float)));
    CheckCuda(cudaMemcpy(d_o_weight, h_o_weight.data(), h_o_weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    const size_t block_size = 16;
    const size_t cache_elems = block_size
        * engine_cfg.model_config.num_hidden_layers
        * engine_cfg.model_config.num_kv_heads
        * engine_cfg.model_config.head_dim;

    float* d_kcache = nullptr;
    float* d_vcache = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_kcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_vcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_kcache, 0, cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_vcache, 0, cache_elems * sizeof(float)));

    auto seq = std::make_shared<Sequence>(0);
    seq->seq_len = 2;
    seq->blocks.push_back(std::make_shared<CacheBlock>(0, d_kcache, d_vcache));

    Batch batch;
    batch.sequences.push_back(seq);
    batch.sequences.push_back(seq);
    batch.num_tokens = 2;
    batch.batch_size = 1;
    // Use distinct cache positions for two tokens to avoid KV overwrite.
    batch.token_positions = {0, 1};
    batch.token_ids = {42, 43};

    ForwardContext context;
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &engine_cfg.model_config;

    AttentionLayerConfig attn_cfg;
    attn_cfg.num_attention_heads = 1;
    attn_cfg.num_kv_heads = 1;
    attn_cfg.head_dim = 4;

    AttentionLayerWeightLayout attn_layout;
    attn_layout.qkv_proj_weight.data = d_qkv_weight;
    attn_layout.qkv_proj_weight.num_elements = h_qkv_weight.size();
    attn_layout.qkv_proj_weight.size = h_qkv_weight.size() * sizeof(float);
    attn_layout.qkv_proj_weight.shape = {4, 12};
    attn_layout.qkv_proj_weight.dtype = DataType::FLOAT32;
    attn_layout.qkv_proj_weight.device = "gpu";

    attn_layout.o_proj_weight.data = d_o_weight;
    attn_layout.o_proj_weight.num_elements = h_o_weight.size();
    attn_layout.o_proj_weight.size = h_o_weight.size() * sizeof(float);
    attn_layout.o_proj_weight.shape = {4, 4};
    attn_layout.o_proj_weight.dtype = DataType::FLOAT32;
    attn_layout.o_proj_weight.device = "gpu";

    Attention attention(attn_cfg, attn_layout);

    // --- prefill ---
    Tensor input;
    input.data = d_input;
    input.num_elements = h_input.size();
    input.size = h_input.size() * sizeof(float);
    input.shape = {2, 4};
    input.dtype = DataType::FLOAT32;
    input.device = "gpu";

    Tensor output;
    output.data = workspace.get_attn_output_workspace();
    output.size = 2 * 4 * sizeof(float);
    output.shape = {2, 4};
    output.dtype = DataType::FLOAT32;
    output.device = "gpu";
    output.num_elements = 8;
    CheckCuda(cudaMemset(output.data, 0, output.size));

    attention.prefill_forward(input, output, context);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_out(8, 0.0f);
    CheckCuda(cudaMemcpy(h_out.data(), output.data, 8 * sizeof(float), cudaMemcpyDeviceToHost));
    // token0: 只看到自己，应该等于v0=[2,3,4,5]
    assert(AlmostEqual(h_out[0], 2.0f));
    assert(AlmostEqual(h_out[1], 3.0f));
    assert(AlmostEqual(h_out[2], 4.0f));
    assert(AlmostEqual(h_out[3], 5.0f));
    // token1: 看到v0和v1，注意力分数由q1*k0和q1*k1决定
    // q1=[2,2,2,2], k0=[1,1,1,1], k1=[2,2,2,2]
    // q1*k0=8, q1*k1=16, softmax(8,16)≈[0.000335,0.999665]
    // v0=[2,3,4,5], v1=[3,4,5,6]
    // output1 ≈ 0.000335*[2,3,4,5] + 0.999665*[3,4,5,6] ≈ [2.999665,3.999665,4.999665,5.999665]
    assert(AlmostEqual(h_out[4], 2.999665f, 1e-3f));
    assert(AlmostEqual(h_out[5], 3.999665f, 1e-3f));
    assert(AlmostEqual(h_out[6], 4.999665f, 1e-3f));
    assert(AlmostEqual(h_out[7], 5.999665f, 1e-3f));

    // --- decode ---
    // 追加一个新token，输入[0,0,1,0]，q=[0,0,0,0], k=[0,0,0,0], v=[7,8,9,10]
    const std::vector<float> h_input2 = {0.0f, 0.0f, 1.0f, 0.0f};
    float* d_input2 = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input2), 4 * sizeof(float)));
    CheckCuda(cudaMemcpy(d_input2, h_input2.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));

    // decode_forward will write token2 cache from qkv projection (row2 gives v=[7,8,9,10]).

    auto seq2 = std::make_shared<Sequence>(0);
    seq2->seq_len = 3;
    seq2->blocks.push_back(std::make_shared<CacheBlock>(0, d_kcache, d_vcache));

    Batch batch2;
    batch2.sequences.push_back(seq2);
    batch2.num_tokens = 1;
    batch2.batch_size = 1;
    batch2.token_positions = {2};
    batch2.token_ids = {44};

    ForwardContext context2;
    context2.layer_id = 0;
    context2.batch = &batch2;
    context2.workspace = &workspace;
    context2.config = &engine_cfg.model_config;

    Tensor input2;
    input2.data = d_input2;
    input2.num_elements = 4;
    input2.size = 4 * sizeof(float);
    input2.shape = {1, 4};
    input2.dtype = DataType::FLOAT32;
    input2.device = "gpu";

    Tensor output2;
    output2.data = workspace.get_attn_output_workspace();
    output2.size = 4 * sizeof(float);
    output2.shape = {1, 4};
    output2.dtype = DataType::FLOAT32;
    output2.device = "gpu";
    output2.num_elements = 4;
    CheckCuda(cudaMemset(output2.data, 0, output2.size));

    attention.decode_forward(input2, output2, context2);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_out2(4, 0.0f);
    CheckCuda(cudaMemcpy(h_out2.data(), output2.data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    // q=[0,0,0,0], k0=[1,1,1,1], k1=[2,2,2,2], k2=[0,0,0,0]
    // q*k0=0, q*k1=0, q*k2=0, softmax全1/3
    // v0=[2,3,4,5], v1=[3,4,5,6], v2=[7,8,9,10]
    // output = (1/3)*[2,3,4,5] + (1/3)*[3,4,5,6] + (1/3)*[7,8,9,10]
    // = [4,5,6,7]
    assert(AlmostEqual(h_out2[0], 4.0f, 1e-3f));
    assert(AlmostEqual(h_out2[1], 5.0f, 1e-3f));
    assert(AlmostEqual(h_out2[2], 6.0f, 1e-3f));
    assert(AlmostEqual(h_out2[3], 7.0f, 1e-3f));

    cudaFree(d_input);
    cudaFree(d_input2);
    cudaFree(d_qkv_weight);
    cudaFree(d_o_weight);
    cudaFree(d_kcache);
    cudaFree(d_vcache);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "test_attention skipped: no CUDA device available\n";
        return 0;
    }

    TestAttentionPrefillForwardWritesCacheAndOutput();
    TestAttentionDecodeForwardWritesCacheAndOutput();
    //TestAttentionMultiTokenPrefillAndDecode();
    std::cout << "test_attention passed\n";
    return 0;
}
