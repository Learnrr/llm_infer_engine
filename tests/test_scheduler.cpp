/*
cd tests

nvcc -std=c++17 -O2 -I../include -I../include/model \
-I../include/layer -I../include/layer/activation \
-I../include/kernel -I../include/utils \
-I/usr/local/cuda/include test_scheduler.cpp \
../src/Scheduler.cpp ../src/KVCacheManager.cpp \
../src/Workspace.cpp ../src/PostProcessor.cpp \
../src/model/IModel.cpp ../src/model/QWEN_Model.cpp \
../src/model/ModelWeights.cpp ../src/layer/Embedding.cpp \
../src/layer/TransformerLayer.cpp ../src/layer/Attention.cpp \
../src/layer/MLP.cpp ../src/layer/Linear.cpp ../src/layer/ResidualAdd.cpp \
../src/layer/RMSNorm.cpp ../src/layer/activation/SwiGLU.cpp \
../src/layer/position/RoPE.cpp ../kernel/embedding_kernel.cu \
../kernel/projection.cu ../kernel/attention_kernel.cu \
../kernel/output_projection_kernel.cu ../kernel/write_kvcache_kernel.cu \
../kernel/rope_kernel.cu ../kernel/linear_kernel.cu ../kernel/swiglu_kernel.cu \
../kernel/residual_add_kernel.cu ../kernel/rmsnorm_kernel.cu \
../kernel/transpose_kernel.cu -L/usr/local/cuda/lib64 -lcudart \
-o ../build/tests/test_scheduler

./../build/tests/test_scheduler.exe
*/

#include "Scheduler.h"
#include "model/QWEN_Model.h"
#include "KVCacheManager.h"
#include "Workspace.h"
#include "llm_engine_config.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
void TestSchedulerEndToEnd() {
    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count <= 0) {
        std::cout << "test_scheduler skipped (no CUDA device)\n";
        return;
    }


    LLMEngineConfig engine_cfg;
    ErrorCode engine_ok = engine_cfg.build_from_file("../llm_engine_config.json");
    assert(engine_ok == ErrorCode::SUCCESS);

    KVCacheManager cache_manager;
    ErrorCode cache_init_err = cache_manager.init(engine_cfg);
    assert(cache_init_err == ErrorCode::SUCCESS);

    QWEN_Model model;
    model.init(engine_cfg);
    model.load_weights(engine_cfg.model_config.model_path.c_str());

    Workspace workspace;
    ErrorCode ws_ok = workspace.init(engine_cfg);
    assert(ws_ok == ErrorCode::SUCCESS);

    Scheduler scheduler(&cache_manager, &model, &workspace, engine_cfg);

    std::vector<size_t> input_tokens = {10, 11};
    ErrorCode add_err = scheduler.addSequence(1, input_tokens);
    assert(add_err == ErrorCode::SUCCESS);

    std::thread scheduler_thread([&scheduler]() {
        scheduler.schedule();
    });

    std::shared_ptr<Sequence> finished_seq;
    bool finished_observed = false;
    bool decode_progress_observed = false;
    const size_t initial_token_count = input_tokens.size();
    for (int i = 0; i < 300; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::shared_ptr<Sequence> live_seq;
        ErrorCode live_err = scheduler.getSequenceById(1, live_seq);
        if (live_err == ErrorCode::SUCCESS && live_seq != nullptr &&
            live_seq->token_ids.size() > initial_token_count) {
            decode_progress_observed = true;
        }

        ErrorCode get_err = scheduler.getFinishedSequenceById(1, finished_seq);
        if (get_err == ErrorCode::SUCCESS && finished_seq != nullptr) {
            finished_observed = true;
            break;
        }
    }

    scheduler.request_stop();
    scheduler_thread.join();

    assert(decode_progress_observed);

    if (finished_observed) {
        assert(finished_seq != nullptr);
        assert(finished_seq->state == SequenceState::FINISHED);
        assert(!finished_seq->token_ids.empty());
        // token_ids should end with eos when scheduler marks FINISHED by eos.
        assert(finished_seq->token_ids.back() == engine_cfg.model_config.eos_token_id);
        // token_ids length should not exceed configured max length.
        assert(finished_seq->token_ids.size() <= engine_cfg.model_config.max_seq_len);

        // eos should only appear once and at the end.
        size_t eos_count = 0;
        for (size_t t : finished_seq->token_ids) {
            if (t == engine_cfg.model_config.eos_token_id) ++eos_count;
        }
        assert(eos_count == 1);

        ErrorCode remove_err = scheduler.removeFinishedSequenceById(1);
        assert(remove_err == ErrorCode::SUCCESS);

        std::shared_ptr<Sequence> removed_seq;
        ErrorCode get_removed_err = scheduler.getFinishedSequenceById(1, removed_seq);
        assert(get_removed_err == ErrorCode::SEQUENCE_NOT_FOUND);
    }
}

int main() {
    TestSchedulerEndToEnd();
    std::cout << "test_scheduler passed\n";
    return 0;
}
