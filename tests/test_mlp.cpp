/*
cd tests
nvcc -std=c++17 -O2 -I../include -I../include/layer -I../include/layer/activation -I../include/model -I../include/kernel \
    test_mlp.cpp ../src/layer/MLP.cpp ../src/layer/Linear.cpp ../src/layer/activation/SwiGLU.cpp ../src/Workspace.cpp \
    ../kernel/linear_kernel.cu ../kernel/swiglu_kernel.cu \
    -o ../build/tests/test_mlp.exe
./../build/tests/test_mlp.exe
*/

#include "layer/MLP.h"
#include "Workspace.h"
#include "llm_engine_config.h"

#include <cassert>
#include <cmath>
#include <iostream>
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

bool AlmostEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

LLMEngineConfig BuildTinyEngineConfig() {
    LLMEngineConfig cfg(
        1,      // max_batch_size
        8,      // max_sequence_length
        1 << 20, // total_cache_size
        16      // block_size
    );

    cfg.model_config.max_seq_len = 8;
    cfg.model_config.hidden_size = 2;
    cfg.model_config.num_hidden_layers = 1;
    cfg.model_config.vocab_size = 32;
    cfg.model_config.num_heads = 1;
    cfg.model_config.num_kv_heads = 1;
    cfg.model_config.head_dim = 2;
    cfg.model_config.data_type = DataType::FLOAT32;
    cfg.model_config.mlp_intermediate_size = 2;
    return cfg;
}

std::vector<float> MatmulCpu(
    const std::vector<float>& input,
    size_t rows,
    size_t in_features,
    const std::vector<float>& weight,
    size_t out_features
) {
    std::vector<float> out(rows * out_features, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < out_features; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                sum += input[r * in_features + k] * weight[k * out_features + c];
            }
            out[r * out_features + c] = sum;
        }
    }
    return out;
}

std::vector<float> ComputeQwenMlpReference(
    const std::vector<float>& input,
    size_t num_tokens,
    const std::vector<float>& w_gate,
    const std::vector<float>& w_up,
    const std::vector<float>& w_down
) {
    const size_t hidden = 2;
    const size_t inter = 2;

    const std::vector<float> gate = MatmulCpu(input, num_tokens, hidden, w_gate, inter);
    const std::vector<float> up = MatmulCpu(input, num_tokens, hidden, w_up, inter);

    std::vector<float> swiglu(num_tokens * inter, 0.0f);
    for (size_t i = 0; i < swiglu.size(); ++i) {
        swiglu[i] = gate[i] * Sigmoid(gate[i]) * up[i];
    }

    return MatmulCpu(swiglu, num_tokens, inter, w_down, hidden);
}

MLP BuildMlp(float* d_w_gate, float* d_w_up, float* d_w_down) {
    MLPLayerConfig cfg;
    cfg.intermediate_size = 2;
    cfg.activation_after_linear_idx = 0;

    LinearConfig gate_cfg;
    gate_cfg.in_features = 2;
    gate_cfg.out_features = 2;
    LinearConfig up_cfg = gate_cfg;
    LinearConfig down_cfg;
    down_cfg.in_features = 2;
    down_cfg.out_features = 2;

    cfg.mlp_linears = {gate_cfg, up_cfg, down_cfg};

    static MLPLayerWeightLayout layout;
    layout.mlp_linears_weight.clear();
    layout.mlp_linears_weight.resize(3);
    layout.mlp_linears_weight[0].linear_weight.data = d_w_gate;
    layout.mlp_linears_weight[0].linear_weight.num_elements = 4;
    layout.mlp_linears_weight[0].linear_weight.size = 4 * sizeof(float);
    layout.mlp_linears_weight[0].linear_weight.shape = {2, 2};
    layout.mlp_linears_weight[0].linear_weight.dtype = DataType::FLOAT32;
    layout.mlp_linears_weight[0].linear_weight.device = "gpu";

    layout.mlp_linears_weight[1].linear_weight.data = d_w_up;
    layout.mlp_linears_weight[1].linear_weight.num_elements = 4;
    layout.mlp_linears_weight[1].linear_weight.size = 4 * sizeof(float);
    layout.mlp_linears_weight[1].linear_weight.shape = {2, 2};
    layout.mlp_linears_weight[1].linear_weight.dtype = DataType::FLOAT32;
    layout.mlp_linears_weight[1].linear_weight.device = "gpu";

    layout.mlp_linears_weight[2].linear_weight.data = d_w_down;
    layout.mlp_linears_weight[2].linear_weight.num_elements = 4;
    layout.mlp_linears_weight[2].linear_weight.size = 4 * sizeof(float);
    layout.mlp_linears_weight[2].linear_weight.shape = {2, 2};
    layout.mlp_linears_weight[2].linear_weight.dtype = DataType::FLOAT32;
    layout.mlp_linears_weight[2].linear_weight.device = "gpu";

    return MLP(cfg, layout);
}

void RunAndCheckMlp(bool use_prefill) {
    const size_t num_tokens = 2;

    // Input [num_tokens, hidden=2].
    const std::vector<float> h_input = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    // Weights [in=2, out=2].
    const std::vector<float> h_w_gate = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const std::vector<float> h_w_up = {
        2.0f, 0.0f,
        0.0f, 3.0f
    };
    const std::vector<float> h_w_down = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_w_gate = nullptr;
    float* d_w_up = nullptr;
    float* d_w_down = nullptr;

    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_input), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), h_input.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_w_gate), h_w_gate.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_w_up), h_w_up.size() * sizeof(float)));
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_w_down), h_w_down.size() * sizeof(float)));

    CheckCuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_w_gate, h_w_gate.data(), h_w_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_w_up, h_w_up.data(), h_w_up.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_w_down, h_w_down.data(), h_w_down.size() * sizeof(float), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemset(d_output, 0, h_input.size() * sizeof(float)));

    Tensor input(
        h_input.size(),
        d_input,
        {num_tokens, 2},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor output(
        h_input.size(),
        d_output,
        {num_tokens, 2},
        DataType::FLOAT32,
        "gpu"
    );

    LLMEngineConfig engine_cfg = BuildTinyEngineConfig();
    Workspace workspace;
    ErrorCode ws_err = workspace.init(engine_cfg);
    assert(ws_err == ErrorCode::SUCCESS);

    Batch batch;
    batch.num_tokens = num_tokens;

    ForwardContext context{};
    context.layer_id = 0;
    context.batch = &batch;
    context.workspace = &workspace;
    context.config = &engine_cfg.model_config;

    MLP mlp = BuildMlp(d_w_gate, d_w_up, d_w_down);

    if (use_prefill) {
        mlp.prefill_forward(input, output, context);
    } else {
        mlp.decode_forward(input, output, context);
    }

    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());

    std::vector<float> h_output(h_input.size(), 0.0f);
    CheckCuda(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<float> expected = ComputeQwenMlpReference(
        h_input,
        num_tokens,
        h_w_gate,
        h_w_up,
        h_w_down
    );

    for (size_t i = 0; i < expected.size(); ++i) {
        if (!AlmostEqual(h_output[i], expected[i])) {
            std::cerr << "MLP mismatch at index " << i
                      << ", got=" << h_output[i]
                      << ", expected=" << expected[i] << "\n";
            assert(false);
        }
    }

    cudaFree(d_w_down);
    cudaFree(d_w_up);
    cudaFree(d_w_gate);
    cudaFree(d_output);
    cudaFree(d_input);
}

void TestMlpPrefillForward() {
    RunAndCheckMlp(true);
}

void TestMlpDecodeForward() {
    RunAndCheckMlp(false);
}

} // namespace

int main() {
    if (!HasCudaDevice()) {
        std::cout << "No CUDA device found, skipping test_mlp\n";
        return 0;
    }

    TestMlpPrefillForward();
    TestMlpDecodeForward();

    std::cout << "test_mlp passed\n";
    return 0;
}
