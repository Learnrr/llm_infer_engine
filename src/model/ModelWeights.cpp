#include "ModelWeights.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <utility>
#include <unordered_set>
#include <unordered_map>

std::vector<std::string> ResolveSafetensorShardsFromIndex(
    const std::filesystem::path& index_path) {

    std::vector<std::string> shards;
    if (!std::filesystem::exists(index_path)) {
        return shards;
    }

    std::ifstream infile(index_path);
    if (!infile.is_open()) {
        return shards;
    }

    json index_json;
    try {
        infile >> index_json;
    } catch (...) {
        return {};
    }

    if (!index_json.contains("weight_map") || !index_json["weight_map"].is_object()) {
        return {};
    }

    std::unordered_set<std::string> seen;
    for (const auto& item : index_json["weight_map"].items()) {
        const std::string rel_name = item.value().get<std::string>();
        const std::filesystem::path shard_path = index_path.parent_path() / rel_name;
        const std::string shard = shard_path.string();
        if (seen.insert(shard).second) {
            if (!std::filesystem::exists(shard_path)) {
                return {};
            }
            shards.push_back(shard);
        }
    }

    return shards;
}

std::vector<std::string> ResolveSafetensorShards(const std::string& model_path) {
    std::filesystem::path p(model_path);
    if (!std::filesystem::exists(p)) {
        return {};
    }

    const std::filesystem::path base_dir = std::filesystem::is_directory(p) ? p : p.parent_path();
    const std::filesystem::path index_path = base_dir / "model.safetensors.index.json";
    return ResolveSafetensorShardsFromIndex(index_path);
}


ErrorCode WeightLayout::build_weight_layout(const ModelConfig& config) {
    size_t offset = 0;
    const DataType layout_dtype = DataType::FLOAT16;
    if (weights == nullptr) {
        LOG_ERROR("build_weight_layout called with null weights pointer");
        return ErrorCode::INVALID_INPUT;
    }
    {
        std::ostringstream oss;
        oss << "build_weight_layout begin, layer_config_count=" << config.layer_configs.size()
            << ", weights_ptr=" << weights;
        LOG_DEBUG(oss.str());
    }
    layer_weights.clear();
    layer_weights.reserve(config.layer_configs.size());
    //embedding weights
    embedding_weights = Tensor(
        config.vocab_size * config.hidden_size,
        static_cast<void*>(static_cast<char*>(weights) + offset),
        {config.vocab_size, config.hidden_size},
        layout_dtype
    );
    offset += embedding_weights.size;
    {
        std::ostringstream oss;
        oss << "layout embedding done, bytes=" << embedding_weights.size << ", offset=" << offset;
        LOG_DEBUG(oss.str());
    }
    //transformer layers
    size_t cfg_idx = 0;
    for (const auto& layer_cfg_base : config.layer_configs) {
        if (!layer_cfg_base) {
            std::ostringstream oss;
            oss << "null layer config at index=" << cfg_idx;
            LOG_ERROR(oss.str());
            return ErrorCode::INVALID_INPUT;
        }
        if (auto transformer_cfg = std::dynamic_pointer_cast<TransformerLayerConfig>(layer_cfg_base)) {
            {
                std::ostringstream oss;
                oss << "layout cfg_idx=" << cfg_idx
                    << " type=Transformer"
                    << " q_heads=" << transformer_cfg->attention_config.num_attention_heads
                    << " kv_heads=" << transformer_cfg->attention_config.num_kv_heads
                    << " head_dim=" << transformer_cfg->attention_config.head_dim
                    << " mlp_linears=" << transformer_cfg->mlp_config.mlp_linears.size();
                LOG_DEBUG(oss.str());
            }
            auto transformer_layout = std::make_shared<TransformerLayerWeightLayout>();

            //attention qkv weights
            size_t q_hidden;
            q_hidden = transformer_cfg->attention_config.num_attention_heads 
                    * transformer_cfg->attention_config.head_dim;
            size_t kv_hidden;
            kv_hidden = transformer_cfg->attention_config.num_kv_heads 
                    * transformer_cfg->attention_config.head_dim;

            transformer_layout->attention_weights.qkv_proj_weight = Tensor(
                config.hidden_size * (q_hidden + 2 * kv_hidden),
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size, q_hidden + 2 * kv_hidden},
                layout_dtype
            );
            offset += transformer_layout->attention_weights.qkv_proj_weight.size;
            {
                std::ostringstream oss;
                oss << "  qkv_proj bytes=" << transformer_layout->attention_weights.qkv_proj_weight.size
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }
            //attention output projection weights
            transformer_layout->attention_weights.o_proj_weight = Tensor(
                q_hidden * config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {q_hidden, config.hidden_size},
                layout_dtype
            );
            offset += transformer_layout->attention_weights.o_proj_weight.size;
            {
                std::ostringstream oss;
                oss << "  o_proj bytes=" << transformer_layout->attention_weights.o_proj_weight.size
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }
            //layer norm weights
            transformer_layout->norm_weights.resize(2);
            transformer_layout->norm_weights[0] = LayerNormLayerWeightLayout();
            transformer_layout->norm_weights[1] = LayerNormLayerWeightLayout();
            transformer_layout->norm_weights[0].norm_weight = Tensor(
                config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size},
                layout_dtype
            );
            transformer_layout->norm_weights[0].gamma = transformer_layout->norm_weights[0].norm_weight.data;
            offset += transformer_layout->norm_weights[0].norm_weight.size;
            transformer_layout->norm_weights[1].norm_weight = Tensor(
                config.hidden_size,
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {config.hidden_size},
                layout_dtype
            );
            transformer_layout->norm_weights[1].gamma = transformer_layout->norm_weights[1].norm_weight.data;
            offset += transformer_layout->norm_weights[1].norm_weight.size;
            {
                std::ostringstream oss;
                oss << "  norms bytes="
                    << (transformer_layout->norm_weights[0].norm_weight.size + transformer_layout->norm_weights[1].norm_weight.size)
                    << ", offset=" << offset;
                LOG_DEBUG(oss.str());
            }


            //mlp weights
            size_t intermediate_size = transformer_cfg->mlp_config.intermediate_size;
            if (intermediate_size == 0 && !transformer_cfg->mlp_config.mlp_linears.empty()) {
                intermediate_size = transformer_cfg->mlp_config.mlp_linears[0].out_features;
            }
            transformer_layout->mlp_weights.mlp_linears_weight.clear();
            transformer_layout->mlp_weights.mlp_linears_weight.reserve(
                transformer_cfg->mlp_config.mlp_linears.size()
            );
            for(const auto& linear_cfg : transformer_cfg->mlp_config.mlp_linears){
                transformer_layout->mlp_weights.mlp_linears_weight.emplace_back();
                auto& linear_layout = transformer_layout->mlp_weights.mlp_linears_weight.back();
                linear_layout.linear_weight = Tensor(
                    linear_cfg.in_features * linear_cfg.out_features,
                    static_cast<void*>(static_cast<char*>(weights) + offset),
                    {linear_cfg.in_features, linear_cfg.out_features},
                    layout_dtype
                );
                offset += linear_layout.linear_weight.size;
                {
                    std::ostringstream oss;
                    oss << "  mlp linear weight bytes=" << linear_layout.linear_weight.size
                        << ", in=" << linear_cfg.in_features << ", out=" << linear_cfg.out_features
                        << ", offset=" << offset;
                    LOG_DEBUG(oss.str());
                }
                if (transformer_cfg->mlp_config.has_bias) {
                    linear_layout.linear_bias = Tensor(
                        linear_cfg.out_features,
                        static_cast<void*>(static_cast<char*>(weights) + offset),
                        {linear_cfg.out_features},
                        layout_dtype
                    );
                    offset += linear_layout.linear_bias.size;
                    {
                        std::ostringstream oss;
                        oss << "  mlp linear bias bytes=" << linear_layout.linear_bias.size
                            << ", offset=" << offset;
                        LOG_DEBUG(oss.str());
                    }
                }
            }

            layer_weights.push_back(transformer_layout);
            ++cfg_idx;
            continue;
        }

        if (auto linear_cfg = std::dynamic_pointer_cast<LinearLayerConfig>(layer_cfg_base)) {
            {
                std::ostringstream oss;
                oss << "layout cfg_idx=" << cfg_idx << " type=Linear"
                    << " in=" << linear_cfg->linear_config.in_features
                    << " out=" << linear_cfg->linear_config.out_features;
                LOG_DEBUG(oss.str());
            }
            auto linear_layout = std::make_shared<LinearLayerWeightLayout>();
            const size_t in_features = linear_cfg->linear_config.in_features;
            const size_t out_features = linear_cfg->linear_config.out_features;
            linear_layout->linear_weight = Tensor(
                in_features * out_features, 
                static_cast<void*>(static_cast<char*>(weights) + offset),
                {in_features, out_features}, 
                DataType::FLOAT16
            );
            offset += linear_layout->linear_weight.size;
            layer_weights.push_back(linear_layout);
            ++cfg_idx;
            continue;
        }

        if (auto norm_cfg = std::dynamic_pointer_cast<LayerNormLayerConfig>(layer_cfg_base)) {
            {
                std::ostringstream oss;
                oss << "layout cfg_idx=" << cfg_idx << " type=LayerNorm"
                    << " size=" << (norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size);
                LOG_DEBUG(oss.str());
            }
            auto norm_layout = std::make_shared<LayerNormLayerWeightLayout>();
            const size_t norm_size = norm_cfg->norm_size == 0 ? config.hidden_size : norm_cfg->norm_size;
            norm_layout->norm_weight = Tensor(
                norm_size, 
                static_cast<void*>(static_cast<char*>(weights) + offset), 
                {norm_size}, 
                DataType::FLOAT16
            );
            norm_layout->gamma = norm_layout->norm_weight.data;
            offset += norm_layout->norm_weight.size;
            layer_weights.push_back(norm_layout);
            ++cfg_idx;
            continue;
        }

        {
            std::ostringstream oss;
            oss << "Unknown layer config type at index=" << cfg_idx;
            LOG_ERROR(oss.str());
        }
        return ErrorCode::INVALID_INPUT;
    }
    {
        std::ostringstream oss;
        oss << "build_weight_layout finished, total offset=" << offset
            << ", layer_weights=" << layer_weights.size();
        LOG_DEBUG(oss.str());
    }
    total_size = offset;
    return ErrorCode::SUCCESS;
}

std::variant<ErrorCode, size_t> ModelWeights::read_total_size(const char* model_safetensors_index_json) {
    size_t total_size = 0;
    std::ifstream infile(model_safetensors_index_json, std::ios::binary);
    if (!infile.is_open()) {
        return ErrorCode::LOAD_ERROR;
    }
    json index_json;
    try {
        infile >> index_json;
    } catch (...) {
        return ErrorCode::LOAD_ERROR;
    }
    if (!index_json.contains("metadata") || !index_json["metadata"].is_object()) {
        return ErrorCode::LOAD_ERROR;
    }
    const auto& metadata = index_json["metadata"];
    if (metadata.contains("total_size")) {
        return metadata["total_size"].get<size_t>();
    }
    return ErrorCode::LOAD_ERROR;
}
    
ErrorCode ModelWeights::init(const ModelConfig& config){
    LOG_DEBUG("ModelWeights::init begin");

    //parse header
    ErrorCode error = parse_header(config.model_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to parse model weight header");
        return error;
    }
    {
        std::ostringstream oss;
        oss << "parse_header success, headers=" << headers.size();
        LOG_DEBUG(oss.str());
    }
    //build weight names list in sequence
    error = build_weight_names(config.weight_names_path.c_str());
    if (error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight names list");
        return error;
    }
    {
        std::ostringstream oss;
        oss << "build_weight_names success, weight_names=" << weight_names.size();
        LOG_DEBUG(oss.str());
    }
    //read total size from safetensors index json
    auto total_size_or_error = read_total_size(
        config.model_safetensors_index_json.c_str()
    );
    if (std::holds_alternative<ErrorCode>(total_size_or_error)) {
        LOG_ERROR("Failed to read total size from safetensors index");
        return std::get<ErrorCode>(total_size_or_error);
    }
    {
        std::ostringstream oss;
        oss << "read_total_size success, bytes=" << std::get<size_t>(total_size_or_error);
        LOG_DEBUG(oss.str());
    }
    //allocate gpu memory for weights
    const size_t allocated_bytes = std::get<size_t>(total_size_or_error);
    cudaError_t cuda_err = cudaMalloc(&weights, allocated_bytes);
    if (cuda_err != cudaSuccess) {
        LOG_ERROR("Failed to allocate GPU memory for model weights ");
        return ErrorCode::CUDA_FAILURE;
    }
    {
        std::ostringstream oss;
        oss << "cudaMalloc success, ptr=" << weights;
        LOG_DEBUG(oss.str());
    }
    //build weight layout
    LOG_DEBUG("calling build_weight_layout");
    layout.weights = weights;
    ErrorCode build_weight_layout_error = layout.build_weight_layout(config);
    if (build_weight_layout_error != ErrorCode::SUCCESS) {
        LOG_ERROR("Failed to build weight layout");
        return build_weight_layout_error;
    }
    {
        std::ostringstream oss;
        oss << "layout.total_size=" << layout.total_size
            << ", allocated_bytes=" << allocated_bytes;
        LOG_DEBUG(oss.str());
    }
    if (layout.total_size > allocated_bytes) {
        LOG_ERROR("Weight layout exceeds allocated GPU memory; model config likely mismatches real weights");
        return ErrorCode::INVALID_INPUT;
    }
    LOG_DEBUG("build_weight_layout success");

    return ErrorCode::SUCCESS;
}
        
ErrorCode ModelWeights::parse_header(const char* file_name){
    headers.clear();

    const std::vector<std::string> shards = ResolveSafetensorShards(file_name);
    if (shards.empty()) {
        LOG_ERROR("Failed to resolve safetensors shards from path:");
        return ErrorCode::LOAD_ERROR;
    }
    {
        std::ostringstream oss;
        oss << "parse_header shard_count=" << shards.size();
        LOG_DEBUG(oss.str());
    }

    size_t layer_idx = 0;

    for (const auto& shard : shards) {
        std::ifstream infile(shard, std::ios::binary);
        if (!infile.is_open()) {
            LOG_ERROR("Failed to open model weight file shard: ");
            return ErrorCode::LOAD_ERROR;
        }

        uint64_t header_size = 0;
        infile.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
        if (!infile || header_size == 0) {
            LOG_ERROR("Failed to read safetensors header size from shard");
            return ErrorCode::LOAD_ERROR;
        }
        {
            std::ostringstream oss;
            oss << "parse_header shard=" << shard << ", header_size=" << header_size;
            LOG_DEBUG(oss.str());
        }

        std::vector<char> header_data(header_size);
        infile.read(header_data.data(), static_cast<std::streamsize>(header_size));
        if (!infile) {
            LOG_ERROR("Failed to read safetensors header data from shard");
            return ErrorCode::LOAD_ERROR;
        }

        json header_json = json::parse(header_data.data(), header_data.data() + header_size);
        for (const auto& item : header_json.items()) {
            std::string name = item.key();
            const auto& value = item.value();

            if (name == "__metadata__") {
                continue;
            }
            if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") ||
                !value.contains("data_offsets") || !value["data_offsets"].is_array() ||
                value["data_offsets"].size() != 2) {
                LOG_ERROR("Malformed tensor entry in safetensors header");
                return ErrorCode::LOAD_ERROR;
            }

            std::string dtype = value["dtype"].get<std::string>();
            std::vector<int> shape = value["shape"].get<std::vector<int>>();

            size_t offset_start = value["data_offsets"][0];
            size_t offset_end = value["data_offsets"][1];

            DataType parsed_dtype;
            if (dtype == "fp16" || dtype == "F16" || dtype == "BF16") {
                parsed_dtype = DataType::FLOAT16;
            } else if (dtype == "fp32" || dtype == "F32") {
                parsed_dtype = DataType::FLOAT32;
            } else {
                LOG_ERROR("Unsupported tensor dtype in safetensors header");
                return ErrorCode::LOAD_ERROR;
            }

            WeightHeader header = {
                layer_idx,
                shape,
                name,
                shard,
                offset_start + 8 + header_size,
                offset_end + 8 + header_size,
                parsed_dtype
            };
            headers[name] = header;
            layer_idx++;
        }
    }

    {
        std::ostringstream oss;
        oss << "parse_header finished, total headers=" << headers.size();
        LOG_DEBUG(oss.str());
    }

    return ErrorCode::SUCCESS;
}

//load to cpu
Tensor ModelWeights::load_layer(std::ifstream& file, const std::string& name) {
    if(headers.find(name) == headers.end()){
        LOG_ERROR("Weight name not found in header");
        return Tensor();
    }
    const WeightHeader& header = headers[name];
    size_t weight_size = (header.offset_end - header.offset_start);
    std::vector<size_t> shape;
    shape.reserve(header.shape.size());
    for (int dim : header.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    Tensor layer_tensor(weight_size / Tensor::element_size_bytes(header.dtype), nullptr, shape, header.dtype);
    layer_tensor.data = malloc(weight_size);
    if (layer_tensor.data == nullptr) {
        LOG_ERROR("Failed to allocate host buffer for layer");
        return Tensor();
    }
    file.seekg(header.offset_start);
    file.read((char*)layer_tensor.data, weight_size);
    if (!file) {
        LOG_ERROR("Failed to read layer tensor bytes");
        free(layer_tensor.data);
        layer_tensor.data = nullptr;
        return Tensor();
    }

    return layer_tensor;

}
ErrorCode ModelWeights::build_weight_names(const char* file_name){
    std::ifstream infile(file_name);
    if(!infile.is_open()){
        LOG_ERROR("Failed to open model weight file");
        return ErrorCode::LOAD_ERROR;
    }

    weight_names.clear();
    std::string line;
    while (std::getline(infile, line)) {
        const size_t begin = line.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) {
            continue;
        }
        const size_t end = line.find_last_not_of(" \t\r\n");
        const std::string name = line.substr(begin, end - begin + 1);
        if (name.empty()) {
            continue;
        }
        weight_names.push_back(name);
    }

    {
        std::ostringstream oss;
        oss << "build_weight_names loaded=" << weight_names.size();
        LOG_DEBUG(oss.str());
    }

    return ErrorCode::SUCCESS;
}
//concat qkv on cpu
Tensor ModelWeights::concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv){
    if (Wq.shape.size() != 2 || Wk.shape.size() != 2 || Wv.shape.size() != 2) {
        LOG_ERROR("concat_qkv expects 2D tensors");
        return Tensor();
    }
    if (Wq.dtype != Wk.dtype || Wq.dtype != Wv.dtype) {
        LOG_ERROR("concat_qkv dtype mismatch");
        return Tensor();
    }

    // HF/Qwen linear weights are [out_features, in_features].
    // Q/K/V must share in_features and be concatenated on out_features (row dimension),
    // then transposes to [in_features, out_features_total].
    const size_t q_rows = Wq.shape[0];
    const size_t k_rows = Wk.shape[0];
    const size_t v_rows = Wv.shape[0];
    const size_t in_cols = Wq.shape[1];
    if (Wk.shape[1] != in_cols || Wv.shape[1] != in_cols) {
        LOG_ERROR("concat_qkv shape mismatch on input feature dimension");
        return Tensor();
    }

    const size_t elem_size = Tensor::element_size_bytes(Wq.dtype);
    const size_t out_rows = q_rows + k_rows + v_rows;

    Tensor out(
        out_rows * in_cols,
        nullptr, 
        {out_rows, in_cols}, 
        Wq.dtype,
        "cpu"
    );

    char* data = new char[out.size];
    out.data = data;

    const char* q = static_cast<const char*>(Wq.data);
    const char* k = static_cast<const char*>(Wk.data);
    const char* v = static_cast<const char*>(Wv.data);

    const size_t q_bytes = q_rows * in_cols * elem_size;
    const size_t k_bytes = k_rows * in_cols * elem_size;
    const size_t v_bytes = v_rows * in_cols * elem_size;

    std::memcpy(data, q, q_bytes);
    std::memcpy(data + q_bytes, k, k_bytes);
    std::memcpy(data + q_bytes + k_bytes, v, v_bytes);

    return out;
}

//copy from cpu to gpu
ErrorCode ModelWeights::load_weights(const char* weight_path) {
    (void)weight_path;
    LOG_DEBUG("load_weights begin");

    Tensor tmp_layer_tensor_q;
    Tensor tmp_layer_tensor_k;
    Tensor tmp_layer_tensor_v;

    std::unordered_map<std::string, std::unique_ptr<std::ifstream>> shard_streams;

    auto get_stream_for_name = [&](const std::string& name) -> std::ifstream* {
        const auto hit = headers.find(name);
        if (hit == headers.end()) {
            return nullptr;
        }
        const std::string& shard = hit->second.shard_file;
        auto stream_it = shard_streams.find(shard);
        if (stream_it == shard_streams.end()) {
            auto stream = std::make_unique<std::ifstream>(shard, std::ios::binary);
            if (!stream->is_open()) {
                LOG_ERROR("Failed to open shard file for tensor ");
                return nullptr;
            }
            stream_it = shard_streams.emplace(shard, std::move(stream)).first;
        }
        return stream_it->second.get();
    };
    
    bool has_q = false;
    bool has_k = false;
    bool has_v = false;
    size_t idx = 0;
    for(const auto& name : weight_names){
        {
            std::ostringstream oss;
            oss << "load_weights idx=" << idx << ", name=" << name;
            LOG_DEBUG(oss.str());
        }
        ++idx;

        std::ifstream* stream = get_stream_for_name(name);
        if (stream == nullptr) {
            LOG_ERROR("load_weights stream resolve failed");
            return ErrorCode::LOAD_ERROR;
        }

        if(name.find("embed_tokens") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            if (tmp_layer_tensor.size != layout.embedding_weights.size) {
                std::ostringstream oss;
                oss << "embed_tokens tensor size mismatch, src=" << tmp_layer_tensor.size
                    << ", dst=" << layout.embedding_weights.size;
                LOG_ERROR(oss.str());
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                return ErrorCode::INVALID_INPUT;
            }
            cudaError_t copy_err = cudaMemcpy(
                layout.embedding_weights.data,
                tmp_layer_tensor.data, 
                layout.embedding_weights.size, 
                cudaMemcpyHostToDevice
            );
            if (copy_err != cudaSuccess) {
                LOG_ERROR("cudaMemcpy embed_tokens failed");
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
                return ErrorCode::CUDA_FAILURE;
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
        } else if(name.find("layers.") != std::string::npos){
            size_t pos1 = name.find("layers.") + 7;
            size_t pos2 = name.find(".", pos1);
            size_t layer_id = std::stoi(name.substr(pos1, pos2 - pos1));

            auto layer_layout = layout.get_layer_layout<TransformerLayerWeightLayout>(layer_id);
            if (!layer_layout) {
                LOG_ERROR("Transformer layer layout missing");
                continue;
            }
            if(name.find("q_proj") != std::string::npos){
                if (tmp_layer_tensor_q.data != nullptr) {
                    free(tmp_layer_tensor_q.data);
                    tmp_layer_tensor_q.data = nullptr;
                }
                Tensor loaded_q = load_layer(*stream, name);
                if(loaded_q.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_q = std::move(loaded_q);
                has_q = true;

            } else if(name.find("k_proj") != std::string::npos){
                if (tmp_layer_tensor_k.data != nullptr) {
                    free(tmp_layer_tensor_k.data);
                    tmp_layer_tensor_k.data = nullptr;
                }
                Tensor loaded_k = load_layer(*stream, name);
                if(loaded_k.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_k = std::move(loaded_k);
                has_k = true;

            } else if(name.find("v_proj") != std::string::npos){
                if (tmp_layer_tensor_v.data != nullptr) {
                    free(tmp_layer_tensor_v.data);
                    tmp_layer_tensor_v.data = nullptr;
                }
                Tensor loaded_v = load_layer(*stream, name);
                if(loaded_v.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                tmp_layer_tensor_v = std::move(loaded_v);
                has_v = true;

            } else if(name.find("o_proj") != std::string::npos){
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->attention_weights.o_proj_weight.size) {
                    std::ostringstream oss;
                    oss << "o_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->attention_weights.o_proj_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->attention_weights.o_proj_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy o_proj failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("gate_proj") != std::string::npos){
                if (layer_layout->mlp_weights.mlp_linears_weight.size() < 3) {
                    LOG_ERROR("MLP linear layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.size) {
                    std::ostringstream oss;
                    oss << "gate_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[0].linear_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy gate_proj failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("up_proj") != std::string::npos){
                if (layer_layout->mlp_weights.mlp_linears_weight.size() < 3) {
                    LOG_ERROR("MLP linear layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.size) {
                    std::ostringstream oss;
                    oss << "up_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[1].linear_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy up_proj failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("down_proj") != std::string::npos){
                if (layer_layout->mlp_weights.mlp_linears_weight.size() < 3) {
                    LOG_ERROR("MLP linear layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.size) {
                    std::ostringstream oss;
                    oss << "down_proj tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->mlp_weights.mlp_linears_weight[2].linear_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy down_proj failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("input_layernorm") != std::string::npos){
                if (layer_layout->norm_weights.size() < 2) {
                    LOG_ERROR("Norm layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->norm_weights[0].norm_weight.size) {
                    std::ostringstream oss;
                    oss << "input_layernorm tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->norm_weights[0].norm_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->norm_weights[0].norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy input_layernorm failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            } else if(name.find("post_attention_layernorm") != std::string::npos){
                if (layer_layout->norm_weights.size() < 2) {
                    LOG_ERROR("Norm layouts are insufficient");
                    return ErrorCode::LOAD_ERROR;
                }
                Tensor tmp_layer_tensor = load_layer(*stream, name);
                if(tmp_layer_tensor.data == nullptr){
                    return ErrorCode::LOAD_ERROR;
                }
                if (tmp_layer_tensor.size != layer_layout->norm_weights[1].norm_weight.size) {
                    std::ostringstream oss;
                    oss << "post_attention_layernorm tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << layer_layout->norm_weights[1].norm_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    layer_layout->norm_weights[1].norm_weight.data,
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy post_attention_layernorm failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
                free(tmp_layer_tensor.data);
                tmp_layer_tensor.data = nullptr;
            }else{
                LOG_ERROR("Unrecognized weight name, weights may be incomplete");
                continue;
            }

            if(!(has_q && has_k && has_v)){
                continue;
            }
            //concat Wq, Wk, Wv
            Tensor Wqkv = concat_qkv(tmp_layer_tensor_q, tmp_layer_tensor_k, tmp_layer_tensor_v);
            if (Wqkv.data == nullptr) {
                LOG_ERROR("concat_qkv failed");
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::LOAD_ERROR;
            }
            //transpose
            Tensor Wqkv_trans = Wqkv.transpose();
            if (Wqkv_trans.data == nullptr) {
                LOG_ERROR("transpose qkv failed");
                delete[] static_cast<char*>(Wqkv.data);
                Wqkv.data = nullptr;
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::LOAD_ERROR;
            }
            if (Wqkv_trans.size != layer_layout->attention_weights.qkv_proj_weight.size) {
                std::ostringstream oss;
                oss << "qkv_proj tensor size mismatch after concat/transpose, src=" << Wqkv_trans.size
                    << ", dst=" << layer_layout->attention_weights.qkv_proj_weight.size
                    << ", q_shape=[" << tmp_layer_tensor_q.shape[0] << "," << tmp_layer_tensor_q.shape[1] << "]"
                    << ", k_shape=[" << tmp_layer_tensor_k.shape[0] << "," << tmp_layer_tensor_k.shape[1] << "]"
                    << ", v_shape=[" << tmp_layer_tensor_v.shape[0] << "," << tmp_layer_tensor_v.shape[1] << "]"
                    << ", cfg_layer_id=" << layer_id;
                LOG_ERROR(oss.str());
                delete[] static_cast<char*>(Wqkv_trans.data);
                Wqkv_trans.data = nullptr;
                delete[] static_cast<char*>(Wqkv.data);
                Wqkv.data = nullptr;
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::INVALID_INPUT;
            }
            //copy from cpu to gpu
            cudaError_t copy_err = cudaMemcpy(
                layer_layout->attention_weights.qkv_proj_weight.data,
                Wqkv_trans.data,
                Wqkv_trans.size,
                cudaMemcpyHostToDevice
            );
            if (copy_err != cudaSuccess) {
                LOG_ERROR("cudaMemcpy qkv_proj_weight failed");
                delete[] static_cast<char*>(Wqkv_trans.data);
                Wqkv_trans.data = nullptr;
                delete[] static_cast<char*>(Wqkv.data);
                Wqkv.data = nullptr;
                free(tmp_layer_tensor_q.data);
                free(tmp_layer_tensor_k.data);
                free(tmp_layer_tensor_v.data);
                tmp_layer_tensor_q.data = nullptr;
                tmp_layer_tensor_k.data = nullptr;
                tmp_layer_tensor_v.data = nullptr;
                return ErrorCode::CUDA_FAILURE;
            }
            delete[] static_cast<char*>(Wqkv_trans.data);
            Wqkv_trans.data = nullptr;
            delete[] static_cast<char*>(Wqkv.data);
            Wqkv.data = nullptr;

            free(tmp_layer_tensor_q.data);
            free(tmp_layer_tensor_k.data);
            free(tmp_layer_tensor_v.data);
            tmp_layer_tensor_q.data = nullptr;
            tmp_layer_tensor_k.data = nullptr;
            tmp_layer_tensor_v.data = nullptr;

            has_k = false;
            has_q = false;  
            has_v = false;

        } else if(name.find("model.norm") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            std::shared_ptr<LayerNormLayerWeightLayout> final_norm_layout;
            for (size_t i = 0; i < layout.layer_weights.size(); ++i) {
                auto candidate = layout.get_layer_layout<LayerNormLayerWeightLayout>(i);
                if (candidate) {
                    final_norm_layout = candidate;
                }
            }
            if (final_norm_layout) {
                if (tmp_layer_tensor.size != final_norm_layout->norm_weight.size) {
                    std::ostringstream oss;
                    oss << "model.norm tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << final_norm_layout->norm_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    final_norm_layout->norm_weight.data,
                    tmp_layer_tensor.data,
                    final_norm_layout->norm_weight.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy model.norm failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
        } else if(name.find("lm_head") != std::string::npos){
            Tensor tmp_layer_tensor = load_layer(*stream, name);
            if(tmp_layer_tensor.data == nullptr){
                return ErrorCode::LOAD_ERROR;
            }
            std::shared_ptr<LinearLayerWeightLayout> lm_head_layout;
            for (size_t i = 0; i < layout.layer_weights.size(); ++i) {
                auto candidate = layout.get_layer_layout<LinearLayerWeightLayout>(i);
                if (candidate) {
                    lm_head_layout = candidate;
                }
            }
            if (lm_head_layout) {
                if (tmp_layer_tensor.size != lm_head_layout->linear_weight.size) {
                    std::ostringstream oss;
                    oss << "lm_head tensor size mismatch, src=" << tmp_layer_tensor.size
                        << ", dst=" << lm_head_layout->linear_weight.size;
                    LOG_ERROR(oss.str());
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::INVALID_INPUT;
                }
                cudaError_t copy_err = cudaMemcpy(
                    lm_head_layout->linear_weight.data,
                    tmp_layer_tensor.data,
                    lm_head_layout->linear_weight.size,
                    cudaMemcpyHostToDevice
                );
                if (copy_err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy lm_head failed");
                    free(tmp_layer_tensor.data);
                    tmp_layer_tensor.data = nullptr;
                    return ErrorCode::CUDA_FAILURE;
                }
            }
            free(tmp_layer_tensor.data);
            tmp_layer_tensor.data = nullptr;
        } else {
            LOG_ERROR("Unrecognized weight name, weights may be incomplete");
            continue;
        }
        
    }

    if (tmp_layer_tensor_q.data) {
        free(tmp_layer_tensor_q.data);
    }
    if (tmp_layer_tensor_k.data) {
        free(tmp_layer_tensor_k.data);
    }
    if (tmp_layer_tensor_v.data) {
        free(tmp_layer_tensor_v.data);
    }

    LOG_DEBUG("load_weights finished");
    
    return ErrorCode::SUCCESS;
}