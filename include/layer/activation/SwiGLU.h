#pragma once
#include "Tensor.h"
#include "Workspace.h"
#include "Layer.h"
#include "ForwardContext.h"
class SwiGLU: public Layer {
    public:
        SwiGLU(int hidden_size){
            this->hidden_size = hidden_size;
        }
        void prefill_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
        void decode_forward(const Tensor& input, Tensor& output, ForwardContext& context) override;
    private:
        size_t hidden_size;

};