/*
cd tests
nvcc -std=c++17 -O2 -DBLOCK_SIZE=16 -I../include -I../include/layer -I../include/layer/activation -I../include/model -I../include/kernel \
    test_transformer_layer.cpp ../src/layer/TransformerLayer.cpp ../src/layer/Attention.cpp ../src/layer/MLP.cpp ../src/layer/Linear.cpp \
    ../src/layer/ResidualAdd.cpp ../src/layer/RMSNorm.cpp ../src/layer/activation/SwiGLU.cpp ../src/layer/position/RoPE.cpp ../src/Workspace.cpp \
    ../kernel/projection.cu ../kernel/attention_kernel.cu ../kernel/output_projection_kernel.cu ../kernel/write_kvcache_kernel.cu \
    ../kernel/rope_kernel.cu ../kernel/linear_kernel.cu ../kernel/swiglu_kernel.cu ../kernel/residual_add_kernel.cu ../kernel/rmsnorm_kernel.cu \
    -o ../build/tests/test_transformer_layer.exe
./../build/tests/test_transformer_layer.exe
*/

#include "layer/TransformerLayer.h"
#include "Workspace.h"
#include "llm_engine_config.h"

#include <cassert>
#include <cmath>
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

bool AlmostEqual(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

LLMEngineConfig BuildTinyEngineConfig() {
    LLMEngineConfig cfg(
        1,       // max_batch_size
        16,      // max_sequence_length
        1 << 20, // total_cache_size
        16       // block_size
    );

    cfg.model_config.max_seq_len = 16;
    cfg.model_config.hidden_size = 4;
    cfg.model_config.num_hidden_layers = 1;
    cfg.model_config.vocab_size = 32;
    cfg.model_config.num_heads = 1;
    cfg.model_config.num_kv_heads = 1;
    cfg.model_config.head_dim = 4;
    cfg.model_config.data_type = DataType::FLOAT32;
    cfg.model_config.mlp_intermediate_size = 4;
    return cfg;
}

void FillWeightTensor(Tensor& t, float* ptr, const std::vector<size_t>& shape) {
    t.data = ptr;
    t.shape = shape;
    t.dtype = DataType::FLOAT32;
    t.device = "gpu";

    size_t elems = 1;
    for (size_t d : shape) {
        elems *= d;
    }
    t.num_elements = elems;
    t.size = elems * sizeof(float);
}

std::shared_ptr<TransformerLayerConfig> BuildTransformerConfig() {
    auto cfg = std::make_shared<TransformerLayerConfig>();

    cfg->attention_config.num_attention_heads = 1;
    cfg->attention_config.num_kv_heads = 1;
    cfg->attention_config.head_dim = 4;

    cfg->norm_configs.resize(2);
    cfg->norm_configs[0].norm_size = 4;
    cfg->norm_configs[1].norm_size = 4;

    cfg->mlp_config.mlp_type = MLPLayerConfig::MLPType::SwiGLU;
    cfg->mlp_config.has_bias = false;
    cfg->mlp_config.intermediate_size = 4;
    cfg->mlp_config.activation_after_linear_idx = 0;
    cfg->mlp_config.mlp_linears = {
        {4, 4},
        {4, 4},
        {4, 4}
    };

    return cfg;
}

std::shared_ptr<TransformerLayerWeightLayout> BuildTransformerWeights(
    float* d_qkv_weight,
    float* d_o_weight,
    float* d_norm1_gamma,
    float* d_norm2_gamma,
    float* d_mlp_gate_weight,
    float* d_mlp_up_weight,
    float* d_mlp_down_weight
) {
    auto layout = std::make_shared<TransformerLayerWeightLayout>();

    FillWeightTensor(layout->attention_weights.qkv_proj_weight, d_qkv_weight, {4, 12});
    FillWeightTensor(layout->attention_weights.o_proj_weight, d_o_weight, {4, 4});

    layout->norm_weights.resize(2);
    FillWeightTensor(layout->norm_weights[0].norm_weight, d_norm1_gamma, {4});
    layout->norm_weights[0].gamma = d_norm1_gamma;
    FillWeightTensor(layout->norm_weights[1].norm_weight, d_norm2_gamma, {4});
    layout->norm_weights[1].gamma = d_norm2_gamma;

    layout->mlp_weights.mlp_linears_weight.resize(3);
    FillWeightTensor(layout->mlp_weights.mlp_linears_weight[0].linear_weight, d_mlp_gate_weight, {4, 4});
    FillWeightTensor(layout->mlp_weights.mlp_linears_weight[1].linear_weight, d_mlp_up_weight, {4, 4});
    FillWeightTensor(layout->mlp_weights.mlp_linears_weight[2].linear_weight, d_mlp_down_weight, {4, 4});

    return layout;
}

void RunAndCheckTransformer(bool use_prefill) {
    LLMEngineConfig engine_cfg = BuildTinyEngineConfig();

    Workspace workspace;
    ErrorCode ws_err = workspace.init(engine_cfg);
    assert(ws_err == ErrorCode::SUCCESS);

    const std::vector<float> h_input = {1.0f, 0.0f, 0.0f, 0.0f};

    const std::vector<float> h_qkv_weight = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    const std::vector<float> h_o_weight = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    const std::vector<float> h_norm_gamma = {1.0f, 1.0f, 1.0f, 1.0f};

    const std::vector<float> h_mlp_gate_weight = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    const std::vector<float> h_mlp_up_weight = h_mlp_gate_weight;

    // Zero down-projection so MLP contribution is predictable zero.
    const std::vector<float> h_mlp_down_weight(16, 0.0f);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_qkv_weight = nullptr;
    float* d_o_weight = nullptr;
    float* d_norm1_gamma = nullptr;
    float* d_norm2_gamma = nullptr;
    float* d_mlp_gate_weight = nullptr;
    float* d_mlp_up_weight = nullptr;
    float* d_mlp_down_weight = nullptr;
    float* d_kcache = nullptr;
    float* d_vcache = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_qkv_weight), h_qkv_weight.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_o_weight), h_o_weight.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_norm1_gamma), h_norm_gamma.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_norm2_gamma), h_norm_gamma.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_mlp_gate_weight), h_mlp_gate_weight.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_mlp_up_weight), h_mlp_up_weight.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_mlp_down_weight), h_mlp_down_weight.size() * sizeof(float)));

    const size_t cache_elems = static_cast<size_t>(16)
        * engine_cfg.model_config.num_hidden_layers
        * engine_cfg.model_config.num_kv_heads
        * engine_cfg.model_config.head_dim;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_kcache), cache_elems * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_vcache), cache_elems * sizeof(float)));

    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_qkv_weight, h_qkv_weight.data(), h_qkv_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_o_weight, h_o_weight.data(), h_o_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_norm1_gamma, h_norm_gamma.data(), h_norm_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_norm2_gamma, h_norm_gamma.data(), h_norm_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_mlp_gate_weight, h_mlp_gate_weight.data(), h_mlp_gate_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_mlp_up_weight, h_mlp_up_weight.data(), h_mlp_up_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_mlp_down_weight, h_mlp_down_weight.data(), h_mlp_down_weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    CheckCuda(cudaMemset(d_output, 0, h_input.size() * sizeof(float)));
    CheckCuda(cudaMemset(d_kcache, 0, cache_elems * sizeof(float)));
    CheckCuda(cudaMemset(d_vcache, 0, cache_elems * sizeof(float)));

    auto layer_cfg = BuildTransformerConfig();
    auto layer_layout = BuildTransformerWeights(
        d_qkv_weight,
        d_o_weight,
        d_norm1_gamma,
        d_norm2_gamma,
        d_mlp_gate_weight,
        d_mlp_up_weight,
        d_mlp_down_weight
    );

    TransformerLayer layer(4, 1, layer_layout, layer_cfg);

    auto seq = std::make_shared<Sequence>(0);
    seq->seq_len = 1;
    seq->blocks.push_back(std::make_shared<CacheBlock>(0, d_kcache, d_vcache));

    Batch batch;
    batch.sequences.push_back(seq);
    batch.num_tokens = 1;
    batch.batch_size = 1;
    batch.token_positions = {0};
    batch.token_ids = {123};

    ForwardContext context{};
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &engine_cfg.model_config;

    Tensor input;
    input.data = d_input;
    input.num_elements = 4;
    input.size = 4 * sizeof(float);
    input.shape = {1, 4};
    input.dtype = DataType::FLOAT32;
    input.device = "gpu";

    Tensor output;
    output.data = d_output;
    output.num_elements = 4;
    output.size = 4 * sizeof(float);
    output.shape = {1, 4};
    output.dtype = DataType::FLOAT32;
    output.device = "gpu";

    if (use_prefill) {
        layer.prefill_forward(input, output, context);
    } else {
        layer.decode_forward(input, output, context);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(4, 0.0f);
    CheckCuda(cudaMemcpy(h_output.data(), d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    const float inv_rms = 1.0f / std::sqrt((1.0f / 4.0f) + 1e-5f);
    const std::vector<float> expected = {
        1.0f + 2.0f * inv_rms,
        3.0f * inv_rms,
        4.0f * inv_rms,
        5.0f * inv_rms
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::isfinite(h_output[i]));
        if (!AlmostEqual(h_output[i], expected[i])) {
            std::cerr << "Transformer output mismatch at index " << i
                      << ", got=" << h_output[i]
                      << ", expected=" << expected[i] << "\n";
            assert(false);
        }
    }

    cudaFree(d_vcache);
    cudaFree(d_kcache);
    cudaFree(d_mlp_down_weight);
    cudaFree(d_mlp_up_weight);
    cudaFree(d_mlp_gate_weight);
    cudaFree(d_norm2_gamma);
    cudaFree(d_norm1_gamma);
    cudaFree(d_o_weight);
    cudaFree(d_qkv_weight);
    cudaFree(d_output);
    cudaFree(d_input);
}

void TestTransformerPrefillForward() {
    RunAndCheckTransformer(true);
}

void TestTransformerDecodeForward() {
    RunAndCheckTransformer(false);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "test_transformer_layer skipped: no CUDA device available\n";
        return 0;
    }

    TestTransformerPrefillForward();
    TestTransformerDecodeForward();

    std::cout << "test_transformer_layer passed\n";
    return 0;
}
