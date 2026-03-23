#pragma once
#include "Linear.h"
#include "SwiGLU.h"
#include "Tensor.h"
#include "Layer.h"
#include "ModelWeights.h"
#include "ForwardContext.h"
#include "ModelConfig.h"
#include <algorithm>
#include <vector>
class MLP: public Layer {
    public:
        MLP(const MLPLayerConfig& mlp_config,
            MLPLayerWeightLayout& layer_layout):
             layer_layout(layer_layout), 
             mlp_config(mlp_config){

            size_t linear_count = std::min(
                mlp_config.mlp_linears.size(),
                layer_layout.mlp_linears_weight.size()
            );
            linears.reserve(linear_count);
            
            //for gated MLP, first linear is the gate projection to mlp_workspace[gate| ], 
            //second linear is the up projection to mlp_workspace[ |up]. 
            //swiglu takes mlp_workspace[gate|up] as input and writes output to gate half mlp_workspace[gate| ].
            //third linear is the output projection after activation.
            for(size_t i = 0; i < linear_count; ++i){
                linears.emplace_back(std::make_unique<Linear>(
                    mlp_config.mlp_linears[i],
                    i, 
                    layer_layout.mlp_linears_weight[i].linear_weight
                ));
            }
            swiglu = std::make_unique<SwiGLU>(mlp_config.intermediate_size);
        }
        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    private:

        std::vector<std::unique_ptr<Linear>> linears;
        std::unique_ptr<SwiGLU> swiglu;

        MLPLayerWeightLayout& layer_layout;
        MLPLayerConfig mlp_config;


};