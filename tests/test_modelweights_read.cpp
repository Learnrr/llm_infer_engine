/*
cd tests
nvcc -std=c++17 -O2 -I../ -I../include -I../include/model -I \
    ../include/utils -I../include/layer test_modelweights_read.cpp \
    ../src/model/ModelWeights.cpp -o ../build/tests/test_modelweights_read.exe
./../build/tests/test_modelweights_read.exe [model_path] [weight_names_path] [config_path]
*/


#include "ModelWeights.h"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"

namespace {

std::string ResolveWeightEntryPath(const std::string& input_path) {
    namespace fs = std::filesystem;

    fs::path p(input_path);
    if (fs::is_directory(p)) {
        for (const auto& entry : fs::directory_iterator(p)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string fname = entry.path().filename().string();
            if (std::regex_match(fname, std::regex(R"(.*-00001-of-\d{5}\.safetensors$)"))) {
                return entry.path().string();
            }
        }

        fs::path single = p / "qwen-7b.safetensors";
        if (fs::exists(single)) {
            return single.string();
        }
        return input_path;
    }

    if (fs::exists(p)) {
        const std::string fname = p.filename().string();
        std::smatch m;
        const std::regex shard_pat(R"((.*)-(\d{5})-of-(\d{5})\.safetensors$)");
        if (std::regex_match(fname, m, shard_pat)) {
            fs::path entry = p.parent_path() /
                (m[1].str() + "-00001-of-" + m[3].str() + ".safetensors");
            if (fs::exists(entry)) {
                return entry.string();
            }
        }
        return input_path;
    }

    return input_path;
}

std::string ResolveIndexPath(const std::string& model_path) {
    namespace fs = std::filesystem;
    fs::path p(model_path);
    fs::path base_dir = fs::is_directory(p) ? p : p.parent_path();
    return (base_dir / "model.safetensors.index.json").string();
}

std::shared_ptr<LayerNormLayerWeightLayout> FindLastLayerNorm(const WeightLayout& layout) {
    std::shared_ptr<LayerNormLayerWeightLayout> last_norm;
    for (size_t i = 0; i < layout.layer_weights.size(); ++i) {
        auto maybe_norm = layout.get_layer_layout<LayerNormLayerWeightLayout>(i);
        if (maybe_norm) {
            last_norm = maybe_norm;
        }
    }
    return last_norm;
}

void AssertDeviceTensorPrefixEqualsHost(const Tensor& device_tensor, const Tensor& host_tensor) {
    assert(device_tensor.data != nullptr && "device tensor data should not be null");
    assert(host_tensor.data != nullptr && "host tensor data should not be null");
    assert(device_tensor.size > 0 && host_tensor.size > 0 && "tensor byte size should be > 0");

    const size_t bytes_to_compare = std::min<size_t>(256, std::min(device_tensor.size, host_tensor.size));
    std::vector<char> from_device(bytes_to_compare);

    cudaError_t copy_err = cudaMemcpy(
        from_device.data(),
        device_tensor.data,
        bytes_to_compare,
        cudaMemcpyDeviceToHost
    );
    assert(copy_err == cudaSuccess && "cudaMemcpy device->host failed in test validation");

    const int cmp = std::memcmp(from_device.data(), host_tensor.data, bytes_to_compare);
    assert(cmp == 0 && "device tensor prefix bytes mismatch vs source tensor");
}

std::string BytesToGiBString(size_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << gib << " GiB";
    return oss.str();
}

bool TestModelWeightsEndToEnd(
    const std::string& model_path,
    const std::string& names_path,
    const std::string& config_path
) {
    assert(std::filesystem::exists(model_path) && "Model path not found");
    assert(std::filesystem::exists(names_path) && "Weight names file not found");
    assert(std::filesystem::exists(config_path) && "Config file not found");

    const std::string index_path = ResolveIndexPath(model_path);
    assert(std::filesystem::exists(index_path) && "model.safetensors.index.json not found");

    ModelConfig config;
    config.build_from_file(config_path.c_str());
    config.model_path = model_path;
    config.weight_names_path = names_path;
    config.model_safetensors_index_json = index_path;

    ModelWeights model_weights;

    auto total_size_or_error = model_weights.read_total_size(index_path.c_str());
    if (std::holds_alternative<size_t>(total_size_or_error)) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            const size_t needed = std::get<size_t>(total_size_or_error);
            std::cout << "GPU free memory: " << BytesToGiBString(free_mem)
                      << ", required by model: " << BytesToGiBString(needed) << "\n";
            if (needed > free_mem) {
                std::cout << "SKIP: insufficient GPU memory for full model load test\n";
                return false;
            }
        }
    }

    const ErrorCode init_err = model_weights.init(config);
    if (init_err == ErrorCode::CUDA_FAILURE) {
        std::cout << "SKIP: ModelWeights::init failed due to CUDA memory allocation\n";
        return false;
    }
    assert(init_err == ErrorCode::SUCCESS && "ModelWeights::init should succeed");

    const ErrorCode load_err = model_weights.load_weights(config.model_path.c_str());
    assert(load_err == ErrorCode::SUCCESS && "ModelWeights::load_weights should succeed");

    assert(model_weights.weights != nullptr && "GPU weights pointer should not be null after init");
    assert(model_weights.layout.embedding_weights.data != nullptr && "embedding layout pointer should not be null");
    assert(!model_weights.headers.empty() && "headers should not be empty after init");
    assert(!model_weights.weight_names.empty() && "weight_names should not be empty after init");
    assert(model_weights.headers.count("model.embed_tokens.weight") == 1);
    assert(model_weights.headers.count("model.norm.weight") == 1);

    const WeightHeader& embed_header = model_weights.headers.at("model.embed_tokens.weight");
    std::ifstream embed_stream(embed_header.shard_file, std::ios::binary);
    assert(embed_stream.is_open() && "Failed to open shard for embed tensor");
    Tensor embed_host = model_weights.load_layer(embed_stream, "model.embed_tokens.weight");
    assert(embed_host.data != nullptr && "embed host tensor should not be null");
    AssertDeviceTensorPrefixEqualsHost(model_weights.layout.embedding_weights, embed_host);

    const WeightHeader& norm_header = model_weights.headers.at("model.norm.weight");
    std::ifstream norm_stream(norm_header.shard_file, std::ios::binary);
    assert(norm_stream.is_open() && "Failed to open shard for model.norm tensor");
    Tensor norm_host = model_weights.load_layer(norm_stream, "model.norm.weight");
    assert(norm_host.data != nullptr && "model.norm host tensor should not be null");

    auto final_norm_layout = FindLastLayerNorm(model_weights.layout);
    assert(final_norm_layout && "final LayerNorm layout not found");
    AssertDeviceTensorPrefixEqualsHost(final_norm_layout->norm_weight, norm_host);

    std::free(embed_host.data);
    std::free(norm_host.data);
    cudaError_t free_err = cudaFree(model_weights.weights);
    assert(free_err == cudaSuccess && "cudaFree(model_weights.weights) failed");
    model_weights.weights = nullptr;

    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = "/Qwen2.5-7B-Instruct";
    std::string names_path = "/llm_infer_engine/weights/qwen-7b_weight_names.txt";
    std::string config_path = "/llm_infer_engine/qwen7b_model_config.json";

    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        names_path = argv[2];
    }
    if (argc >= 4) {
        config_path = argv[3];
    }

    model_path = ResolveWeightEntryPath(model_path);

    std::cout << "Using model path: " << model_path << "\n";
    std::cout << "Using names file: " << names_path << "\n";
    std::cout << "Using config file: " << config_path << "\n";

    const bool ran_full_test = TestModelWeightsEndToEnd(model_path, names_path, config_path);
    if (ran_full_test) {
        std::cout << "test_modelweights_read passed\n";
    } else {
        std::cout << "test_modelweights_read skipped\n";
    }
    return 0;
}
