/*
cd tests
nvcc -std=c++17 -Iinclude -Iinclude/layer -Iinclude/model \
-Iinclude/kernel test_workspace.cpp Workspace.cpp \
-o ..\build\tests\test_workspace.exe \
./../build/tests/test_workspace.exe [llm_engine_config.json]
*/

#include "Workspace.h"
#include "llm_engine_config.h"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include "define.h"
namespace {

bool IsAligned(size_t value, size_t alignment) {
    return alignment != 0 && (value % alignment) == 0;
}

void ValidateLayout(const WorkspaceLayout& layout) {
    constexpr size_t kAlign = 256;

    assert(layout.total_size > 0 && "workspace total size should be > 0");

    assert(layout.hidden_offset <= layout.hidden2_offset);
    assert(layout.hidden2_offset <= layout.layer_workspace.attn_norm_offset);
    assert(layout.layer_workspace.attn_norm_offset <= layout.layer_workspace.attention_workspace.qkv_offset);
    assert(layout.layer_workspace.attention_workspace.qkv_offset <= layout.layer_workspace.attention_workspace.attn_out_offset);
    assert(layout.layer_workspace.attention_workspace.attn_out_offset <= layout.layer_workspace.attention_workspace.context_offset);
    assert(layout.layer_workspace.attention_workspace.context_offset <= layout.layer_workspace.mlp_workspace.mlp_offset);
    assert(layout.layer_workspace.mlp_workspace.mlp_offset <= layout.layer_workspace.mlp_norm_offset);
    assert(layout.layer_workspace.mlp_norm_offset <= layout.layer_workspace.mlp_workspace.mlp_out_offset);
    assert(layout.layer_workspace.mlp_workspace.mlp_out_offset <= layout.temp_offset);
    assert(layout.temp_offset <= layout.logits_offset);
    assert(layout.logits_offset <= layout.total_size);

    assert(IsAligned(layout.hidden_offset, kAlign));
    assert(IsAligned(layout.hidden2_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.attn_norm_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.attention_workspace.qkv_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.attention_workspace.attn_out_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.attention_workspace.context_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.mlp_workspace.mlp_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.mlp_norm_offset, kAlign));
    assert(IsAligned(layout.layer_workspace.mlp_workspace.mlp_out_offset, kAlign));
    assert(IsAligned(layout.temp_offset, kAlign));
    assert(IsAligned(layout.logits_offset, kAlign));
    assert(IsAligned(layout.total_size, kAlign));
}

LLMEngineConfig BuildDefaultEngineConfig() {
    LLMEngineConfig cfg(
        1,                              // max_batch_size
        32768,                          // max_sequence_length
        2ULL * 1024ULL * 1024ULL * 1024ULL, // total_cache_size
        16                              // block_size
    );

    // Minimal model config for workspace sizing.
    cfg.model_config.max_seq_len = 32768;
    cfg.model_config.hidden_size = 3584;
    cfg.model_config.vocab_size = 152064;
    cfg.model_config.num_heads = 28;
    cfg.model_config.num_kv_heads = 4;
    cfg.model_config.head_dim = 128;
    cfg.model_config.mlp_intermediate_size = 18944;
    cfg.model_config.data_type = DataType::FLOAT16; // float16 bytes

    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    LLMEngineConfig engine_config = BuildDefaultEngineConfig();

    if (argc >= 2) {
        //const std::string config_path = argv[1];
        const std::string config_path = "/llm_infer_engine/llm_engine_config.json";
        engine_config.build_from_file(config_path.c_str());
        std::cout << "Loaded engine config from: " << config_path << "\n";
    } else {
        std::cout << "Using built-in default engine config for workspace test\n";
    }

    Workspace workspace;
    ErrorCode init_error = workspace.init(engine_config);

    if (init_error != ErrorCode::SUCCESS) {
        std::cerr << "Workspace::init failed with error code: " << static_cast<int>(init_error) << "\n";
        return 1;
    }

    assert(workspace.get_workspace() != nullptr && "workspace device pointer should not be null after init");

    WorkspaceLayout layout = workspace.get_layout();
    ValidateLayout(layout);

    std::cout << "Workspace layout validated successfully\n";
    std::cout << "Total workspace bytes: " << layout.total_size << "\n";
    std::cout << "test_workspace passed\n";

    return 0;
}
