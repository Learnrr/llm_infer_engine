/*
cd tests
nvcc -std=c++17 -O2 -I../ -I../include -I../include/model -I../include/utils \
    test_postprocessor.cpp ../src/PostProcessor.cpp \
    -o ../build/tests/test_postprocessor.exe
./../build/tests/test_postprocessor.exe
*/

#include "PostProcessor.h"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

ModelConfig BuildConfig(size_t vocab_size, float temperature, size_t top_k, float top_p) {
    ModelConfig cfg;
    cfg.vocab_size = vocab_size;
    cfg.temperature = temperature;
    cfg.top_k = top_k;
    cfg.top_p = top_p;
    return cfg;
}

ForwardContext BuildContext(Batch* batch) {
    ForwardContext ctx{};
    ctx.batch = batch;
    ctx.workspace = nullptr;
    ctx.config = nullptr;
    ctx.layer_id = 0;
    return ctx;
}

void TestProcessSkipsNullBatch() {
    ModelConfig cfg = BuildConfig(5, 1.0f, 1, 0.9f);
    PostProcessor post_processor(cfg);

    std::vector<float> logits = {0.1f, 0.2f, 0.9f, 0.3f, 0.0f};
    Tensor input(logits.size(), logits.data(), {1, logits.size()}, DataType::FLOAT32, "cpu");

    ForwardContext ctx = BuildContext(nullptr);
    post_processor.process(input, ctx);
}

void TestProcessTopK1PicksArgmaxPerSequence() {
    ModelConfig cfg = BuildConfig(5, 1.0f, 1, 0.9f);
    PostProcessor post_processor(cfg);

    std::vector<float> logits = {
        0.1f, 0.2f, 0.9f, 0.3f, 0.0f,
        -1.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };
    Tensor input(logits.size(), logits.data(), {2, cfg.vocab_size}, DataType::FLOAT32, "cpu");

    Batch batch{};
    batch.batch_size = 2;

    ForwardContext ctx = BuildContext(&batch);
    post_processor.process(input, ctx);

    assert(batch.sampled_token_ids.size() == 2);
    assert(batch.sampled_token_ids[0] == 2);
    assert(batch.sampled_token_ids[1] == 1);
}

void TestTemperatureZeroStillProducesValidSampling() {
    ModelConfig cfg = BuildConfig(4, 0.0f, 1, 1.0f);
    PostProcessor post_processor(cfg);

    std::vector<float> logits = {1.0f, 5.0f, 4.0f, -2.0f};
    Tensor input(logits.size(), logits.data(), {1, cfg.vocab_size}, DataType::FLOAT32, "cpu");

    Batch batch{};
    batch.batch_size = 1;

    ForwardContext ctx = BuildContext(&batch);
    post_processor.process(input, ctx);

    assert(batch.sampled_token_ids.size() == 1);
    assert(batch.sampled_token_ids[0] == 1);
}

} // namespace

int main() {
    TestProcessSkipsNullBatch();
    TestProcessTopK1PicksArgmaxPerSequence();
    TestTemperatureZeroStillProducesValidSampling();

    std::cout << "test_postprocessor passed\n";
    return 0;
}
