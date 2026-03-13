#pragma once
#include "ModelConfig.h"
#include "cuda_runtime.h"
#include "define.h"
#include "Tensor.h"
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

struct LayerWeightLayout {
    //attention
    Tensor qkv_proj_weight;
    Tensor o_proj_weight;

    //mlp
    Tensor gate_proj_weight;
    Tensor up_proj_weight;
    Tensor down_proj_weight;

    //layer norm
    Tensor attn_norm_weight;
    Tensor ffn_norm_weight;



}


struct WeightLayout{
    Tensor embedding_weights;

    std::vector<LayerWeightLayout> layers;

    Tensor llm_head_weight;

    size_t total_size;

    void build_config(const ModelConfig& config);

    void build();
    
};

struct WeightHeader {
    size_t layer_idx;
    std::vector<int> shape;
    std::string name;
    size_t offset_start;
    size_t offset_end;
    DataType dtype;

};

class ModelWeights {
    public: 
        ModelWeights(){};
        ~ModelWeights(){};

        void init(const ModelConfig& config);
        
        void parse_header(const char* file_name);

        //load to cpu
        Tensor load_layer(std::ifstream file, std::string name);
        //concat qkv on cpu
        Tensor concat_qkv(const Tensor& Wq, const Tensor& Wk, const Tensor& Wv);

        //copy from cpu to gpu
        void load_weights(const char* weight_path);

        void* weights;
        std::unordered_map<std::string, WeightHeader> headers;
        WeightLayout layout;

}