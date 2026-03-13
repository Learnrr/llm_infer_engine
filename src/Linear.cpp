#include "Linear.h"

void Linear::forward(Tensor& input, Tensor& output, ForwardContext& context) override {
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