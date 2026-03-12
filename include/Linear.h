#pragma once
#include "define.h"
#include "Tensor.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"
#include "ModelWeights.h"
class Linear: public Layer {
    public:
        Linear(int input_size, int output_size, size_t layer_id, LayerWeightLayout* layer_layout){
            this->input_size = input_size;
            this->output_size = output_size;
            this->layer_id = layer_id;
            this->layer_layout = layer_layout;
        }
        void forward(Tensor& input, Tensor& output, ForwardContext& context) override {
            if(layer_id == 1){
                projection_kernel(input.data, layer_layout->gate_proj_weight.data, nullptr, output.data);
            } else if(layer_id == 2){
                projection_kernel(input.data, layer_layout->up_proj_weight.data, nullptr, output.data);
            } else if(layer_id == 3){
                projection_kernel(input.data, layer_layout->down_proj_weight.data, nullptr, output.data);
            } else {
                // Handle error case for invalid layer_id
            }

        }
    private:
        size_t input_size;
        size_t output_size;
        size_t layer_id;
        LayerWeightLayout* layer_layout;

};