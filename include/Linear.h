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
        void forward(Tensor& input, Tensor& output, ForwardContext& context) override;
    private:
        size_t input_size;
        size_t output_size;
        size_t layer_id;
        LayerWeightLayout* layer_layout;

};