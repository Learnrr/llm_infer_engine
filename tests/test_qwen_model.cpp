/*
cd tests
nvcc -std=c++17 -O2 -DBLOCK_SIZE=16 -I../include -I../include/model -I../include/layer -I../include/layer/activation -I../include/kernel \
    test_qwen_model.cpp ../src/model/IModel.cpp ../src/model/QWEN_Model.cpp ../src/model/ModelWeights.cpp ../src/Workspace.cpp ../src/PostProcessor.cpp \
    ../src/layer/Embedding.cpp ../src/layer/TransformerLayer.cpp ../src/layer/Attention.cpp ../src/layer/MLP.cpp ../src/layer/Linear.cpp \
    ../src/layer/ResidualAdd.cpp ../src/layer/RMSNorm.cpp ../src/layer/activation/SwiGLU.cpp ../src/layer/position/RoPE.cpp \
    ../kernel/embedding_kernel.cu ../kernel/projection.cu ../kernel/attention_kernel.cu ../kernel/output_projection_kernel.cu \
    ../kernel/write_kvcache_kernel.cu ../kernel/rope_kernel.cu ../kernel/linear_kernel.cu ../kernel/swiglu_kernel.cu \
    ../kernel/residual_add_kernel.cu ../kernel/rmsnorm_kernel.cu ../kernel/transpose_kernel.cu\
    -o ../build/tests/test_qwen_model.exe
./../build/tests/test_qwen_model.exe
*/

#include "QWEN_Model.h"

#include "llm_engine_config.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <variant>

#include <cuda_runtime.h>

namespace {

namespace fs = std::filesystem;

fs::path ResolveRepoRoot() {
    std::vector<fs::path> candidates;

    // Most common when running from repo/tests.
    candidates.push_back(fs::current_path().parent_path());
    candidates.push_back(fs::current_path());

    // Try paths derived from __FILE__ (absolute or relative depending on compiler flags).
    const fs::path this_file = fs::path(__FILE__);
    if (!this_file.empty()) {
        candidates.push_back(this_file.parent_path().parent_path());
        candidates.push_back(fs::absolute(this_file).parent_path().parent_path());
    }

    for (const auto& cand : candidates) {
        if (cand.empty()) {
            continue;
        }
        if (fs::exists(cand / "qwen7b_model_config.json") &&
            fs::exists(cand / "weights" / "Qwen2.5-7B-Instruct")) {
            return cand;
        }
    }

    std::cerr << "Failed to resolve repo root. cwd=" << fs::current_path() << " __FILE__=" << __FILE__ << "\n";
    assert(false && "Could not resolve repository root for test assets");
    return {};
}

ModelConfig BuildRealConfig() {
    const fs::path repo_root = ResolveRepoRoot();
    const fs::path config_path = repo_root / "qwen7b_model_config.json";
    assert(fs::exists(config_path));

    const fs::path model_dir = repo_root / "weights" / "Qwen2.5-7B-Instruct";
    const fs::path names_path = repo_root / "weights" / "qwen2.5_7b_instruct_weight_names.txt";
    const fs::path index_path = model_dir / "model.safetensors.index.json";
    assert(fs::exists(model_dir));
    assert(fs::exists(names_path));
    assert(fs::exists(index_path));

    ModelConfig cfg;
    ErrorCode cfg_err = cfg.build_from_file(config_path.string().c_str());
    assert(cfg_err == ErrorCode::SUCCESS);

    cfg.model_path = model_dir.string();
    cfg.weight_names_path = names_path.string();
    cfg.model_safetensors_index_json = index_path.string();

    // Keep workspace and test runtime manageable while preserving real model topology.
    cfg.max_seq_len = 16;
    // Kernels in this project currently run as float paths.
    cfg.data_type = DataType::FLOAT32;

    return cfg;
}

bool HasEnoughGpuMemoryForModel(const ModelConfig& cfg) {
    ModelWeights probe;
    auto total_size_or_error = probe.read_total_size(cfg.model_safetensors_index_json.c_str());
    if (std::holds_alternative<ErrorCode>(total_size_or_error)) {
        return false;
    }

    const size_t needed = std::get<size_t>(total_size_or_error);
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_err != cudaSuccess) {
        return false;
    }
    return free_mem >= needed;
}

size_t CacheBlockNumElements(const ModelConfig& cfg, size_t block_size) {
    return block_size * cfg.num_hidden_layers * cfg.num_kv_heads * cfg.head_dim;
}

void TestQwenPrefillAndDecodeWithRealWeights() {
    ModelConfig cfg = BuildRealConfig();
    assert(HasEnoughGpuMemoryForModel(cfg));

    QWEN_Model model;
    model.init(cfg);

    model.load_weights(cfg.model_path.c_str());

    LLMEngineConfig engine_cfg(
        1,
        cfg.max_seq_len,
        1 << 20,
        16
    );
    engine_cfg.model_config = cfg;

    Workspace workspace;
    ErrorCode ws_err = workspace.init(engine_cfg);
    assert(ws_err == ErrorCode::SUCCESS);

    const size_t block_size = 16;
    const size_t cache_elems = CacheBlockNumElements(cfg, block_size);
    float* d_kcache = nullptr;
    float* d_vcache = nullptr;
    cudaError_t kcache_err = cudaMalloc(reinterpret_cast<void**>(&d_kcache), cache_elems * sizeof(float));
    cudaError_t vcache_err = cudaMalloc(reinterpret_cast<void**>(&d_vcache), cache_elems * sizeof(float));
    assert(kcache_err == cudaSuccess);
    assert(vcache_err == cudaSuccess);
    assert(cudaMemset(d_kcache, 0, cache_elems * sizeof(float)) == cudaSuccess);
    assert(cudaMemset(d_vcache, 0, cache_elems * sizeof(float)) == cudaSuccess);

    auto seq = std::make_shared<Sequence>(0);
    seq->seq_len = 1;
    seq->blocks.push_back(std::make_shared<CacheBlock>(0, d_kcache, d_vcache));

    Batch prefill_batch;
    prefill_batch.sequences.push_back(seq);
    prefill_batch.token_ids = {1};
    prefill_batch.num_tokens = 1;
    prefill_batch.token_positions = {0};
    prefill_batch.batch_size = 1;

    model.prefill_forward(prefill_batch, workspace);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    Batch decode_batch;
    decode_batch.sequences.push_back(seq);
    decode_batch.token_ids = {2};
    decode_batch.num_tokens = 1;
    decode_batch.token_positions = {1};
    decode_batch.batch_size = 1;

    seq->seq_len = 2;
    model.decode_forward(decode_batch, workspace);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    assert(decode_batch.sampled_token_ids.size() == 1);
    assert(decode_batch.sampled_token_ids[0] < cfg.vocab_size);
    assert(decode_batch.token_ids.size() == 2);

    assert(cudaFree(d_kcache) == cudaSuccess);
    assert(cudaFree(d_vcache) == cudaSuccess);
}

} // namespace

int main() {
    TestQwenPrefillAndDecodeWithRealWeights();
    std::cout << "test_qwen_model passed\n";
    return 0;
}
