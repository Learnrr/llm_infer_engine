#include "ModelWeights.h"

void WeightLayout::build_config(const ModelConfig& config){
    size_t offset = 0;
    embedding_weights = Tensor(config.vocab_size * config.hidden_size, nullptr, {config.vocab_size, config.hidden_size}, DTYPE);
    offset += config.vocab_size * config.hidden_size * DTYPE; // embedding weights

    for(int i = 0;i<config.num_hidden_layers;i++){
        LayerWeightLayout layer_layout;
        layer_layout.qkv_proj_weight = Tensor(config.hidden_size * config.hidden_size * DTYPE * 3, nullptr, {config.hidden_size, config.hidden_size * 3}, DTYPE);
        offset += config.hidden_size * config.hidden_size * DTYPE * 3; // q_proj weights
        /*
        layer_layout.k_proj_weight = Tensor(config.hidden_size * config.hidden_size * DTYPE, nullptr, {config.hidden_size, config.hidden_size}, DTYPE);
        offset += config.hidden_size * config.hidden_size * DTYPE; // k_proj weights

        layer_layout.v_proj_weight = Tensor(config.hidden_size * config.hidden_size * DTYPE, nullptr, {config.hidden_size, config.hidden_size}, DTYPE);
        offset += config.hidden_size * config.hidden_size * DTYPE; // v_proj weights
        */
        layer_layout.o_proj_weight = Tensor(config.hidden_size * config.hidden_size * DTYPE, nullptr, {config.hidden_size, config.hidden_size}, DTYPE);
        offset += config.hidden_size * config.hidden_size * DTYPE; // o_proj weights

        layer_layout.gate_proj_weight = Tensor(config.hidden_size * config.intermediate_size * DTYPE, nullptr, {config.hidden_size, config.intermediate_size}, DTYPE);
        offset += config.hidden_size * config.intermediate_size * DTYPE; // gate_proj weights

        layer_layout.up_proj_weight = Tensor(config.intermediate_size * config.hidden_size * DTYPE, nullptr, {config.intermediate_size, config.hidden_size}, DTYPE);
        offset += config.intermediate_size * config.hidden_size * DTYPE; // up_proj weights

        layer_layout.down_proj_weight = Tensor(config.hidden_size * config.intermediate_size * DTYPE, nullptr, {config.hidden_size, config.intermediate_size}, DTYPE);
        offset += config.hidden_size * config.intermediate_size * DTYPE; // down_proj weights

        layer_layout.attn_norm_weight = Tensor(config.hidden_size * DTYPE, nullptr, {config.hidden_size}, DTYPE);
        offset += config.hidden_size * DTYPE; // attention layer norm weights

        layer_layout.ffn_norm_weight = Tensor(config.hidden_size * DTYPE, nullptr, {config.hidden_size}, DTYPE);
        offset += config.hidden_size * DTYPE; // ffn layer norm weights

        layers.push_back(layer_layout);
    }

    llm_head_weight = Tensor(config.vocab_size * config.hidden_size * DTYPE, nullptr, {config.vocab_size, config.hidden_size}, DTYPE);
    offset += config.vocab_size * config.hidden_size * DTYPE; // lm head weights

    total_size = offset;

}

void WeightLayout::build(){
    // Implement logic to set the data pointers for each weight tensor based on the calculated offsets
    size_t offset = 0;
    embedding_weights.data = (char*)weights + offset;
    offset += embedding_weights.size;
    for(int i = 0;i<layers.size();i++){
        LayerWeightLayout& layer_layout = layers[i];
        layer_layout.q_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.qkv_proj_weight.size;
        /*
        layer_layout.k_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.k_proj_weight.size;

        layer_layout.v_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.v_proj_weight.size;
        */
        layer_layout.o_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.o_proj_weight.size;

        layer_layout.gate_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.gate_proj_weight.size;

        layer_layout.up_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.up_proj_weight.size;

        layer_layout.down_proj_weight.data = (char*)weights + offset;
        offset += layer_layout.down_proj_weight.size;

        layer_layout.attn_norm_weight.data = (char*)weights + offset;
        offset += layer_layout.attn_norm_weight.size;

        layer_layout.ffn_norm_weight.data = (char*)weights + offset;
        offset += layer_layout.ffn_norm_weight.size;
    }
    llm_head_weight.data = (char*)weights + offset;
}
    
void ModelWeights::init(const ModelConfig& config){
    layout.build_config(config);
    cudaMalloc(&weights, layout.total_size);
    layout.build();
}
        
void ModelWeights::parse_header(const char* file_name){
    std::ifstream infile(file_name, std::ios::binary);
    uint64_t header_size;
    infile.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));

    char* header_data = new char[header_size];
    infile.read(header_data, header_size);

    json header_json = json::parse(header_data);
    for (const auto& item : header_json.items()) {
        string name = item.key();
        auto value = item.value();

        char* dtype = meta["dtype"];
        vector<int> shape = meta["shape"];

        size_t offset_start = meta["data_offsets"][0];
        size_t offset_end = meta["data_offsets"][1];

        WeightHeader header = {
            layer_idx, 
            shape, 
            name, 
            offset_start + 8 + header_size, 
            offset_end + 8 + header_size, 
            dtype == "fp16" ? DataType::FLOAT16 : DataType::FLOAT32
        };
        headers[name] = header;
    }


}

//load to cpu
Tensor ModelWeights::load_layer(std::ifstream file, string name) {
    WeightHeader header = headers[name];
    size_t weight_size = (header.offset_end - header.offset_start);
    Tensor layer_tensor(weight_size,header.shape);
    layer_tensor.data = malloc(weight_size);
    file.seekg(header.offset_start);
    file.read((char*)layer_tensor.data, weight_size);

    return layer_tensor;

}
//concat qkv on cpu
Tensor ModelWeights::concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv){
    int H = Wq.shape[0];

    Tensor out;
    out.shape = {H, 3 * H};
    out.numel = H * 3 * H;

    float* data = new float[out.numel];
    out.data = data;

    float* q = (float*)Wq.data;
    float* k = (float*)Wk.data;
    float* v = (float*)Wv.data;

    for (int i = 0; i < H; i++) {

        memcpy(data + i * 3 * H,
            q + i * H,
            H * sizeof(float));

        memcpy(data + i * 3 * H + H,
            k + i * H,
            H * sizeof(float));

        memcpy(data + i * 3 * H + 2 * H,
            v + i * H,
            H * sizeof(float));
    }

    return out;
}

//copy from cpu to gpu
void ModelWeights::load_weights(const char* weight_path) {
    // Load model weights logic, e.g., read weights from file
    std::ifstream infile(weight_path, std::ios::binary);


    Tensor tmp_layer_tensor;
    Tensor tmp_layer_tensor_k;
    Tensor tmp_layer_tensor_v;            
    for(auto& item : headers){
        const string& name = item.first;
        if(name.find("embed_tokens") != string::npos){
            tmp_layer_tensor = load_layer(infile, name);
            cudaMemcpy(
                (char*)weights + layout.embedding_weights.data, 
                tmp_layer_tensor.data, 
                layout.embedding_weights.size, 
                cudaMemcpyHostToDevice
            );
        } else if(name.find("layer.") != string::npos){
            size_t pos1 = name.find("layers.") + 7;
            size_t pos2 = name.find(".", pos1);
            size_t layer_id = std::stoi(name.substr(pos1, pos2 - pos1));

            

            LayerWeightLayout& layer_layout = layout.layers[layer_id];
            if(name.find("q_proj") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.q_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("k_proj") != string::npos){
                tmp_layer_tensor_k = load_layer(infile, name);
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.k_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("v_proj") != string::npos){
                tmp_layer_tensor_v = load_layer(infile, name);
                /*
                cudaMemcpy(
                    (char*)weights + layer_layout.v_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
                */
            } else if(name.find("o_proj") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    (char*)weights + layer_layout.o_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("gate_proj") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    (char*)weights + layer_layout.gate_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("up_proj") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    (char*)weights + layer_layout.up_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("down_proj") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    (char*)weights + layer_layout.down_proj_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("attn_norm") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    (char*)weights + layer_layout.attn_norm_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            } else if(name.find("ffn_norm") != string::npos){
                tmp_layer_tensor = load_layer(infile, name);
                cudaMemcpy(
                    (char*)weights + layer_layout.ffn_norm_weight.data, 
                    tmp_layer_tensor.data, 
                    tmp_layer_tensor.size, 
                    cudaMemcpyHostToDevice
                );
            }
            //concat Wq, Wk, Wv
            Tensor Wqkv = concat_qkv(tmp_layer_tensor, tmp_layer_tensor_k, tmp_layer_tensor_v);
            //transpose
            Tensor Wqkv_trans = Wqkv.tranpose();
            //copy from cpu to gpu
            cudaMemcpy(
                (char*)weights + layer_layout.qkv_proj_weight.data,
                Wqkv_trans.data,
                Wqkv_trans.size,
                cudaMemcpyHostToDevice
            );

        } else if(name.find("lm_head") != string::npos){
            tmp_layer_tensor = load_layer(infile, name);
            cudaMemcpy(
                (char*)weights + layout.lm_head_weight.data, 
                tmp_layer_tensor.data, 
                layout.lm_head_weight.size, 
                cudaMemcpyHostToDevice
            );
        }
        free(tmp_layer_tensor.data);
        
    }
    
    
}