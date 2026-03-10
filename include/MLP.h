#pragma once
#include "Linear.h"
#include "SwiGLU.h"
#include "Tensor.h"
#include "Multiply.h"
#include "Layer.h"
#include "ModelWeights.h"
class MLP: public Layer {
    public:
        MLP(int hidden_size, int intermediate_size, LayerWeightLayout* layer_layout): layer_layout(layer_layout){
            linear1 = make_unique<Linear>(hidden_size, intermediate_size);
            linear2 = make_unique<Linear>(intermediate_size, hidden_size);
            swiglu = make_unique<SwiGLU>(intermediate_size);
        }
        void forward(void* input, void* output) override {

            Tensor intermediate_output(intermediate_size, {input.shape[0], intermediate_size}, input.dtype);

            linear1->forward(input, intermediate_output);
            linear2->forward(intermediate_output, output);
            swiglu->forward(intermediate_output.data, intermediate_output.data);
            

        }
    private:

        std::unique_ptr<Linear> linear1;
        std::unique_ptr<Linear> linear2;
        std::unique_ptr<Linear> linear3;

        std::unique_ptr<SwiGLU> swiglu;

        LayerWeightLayout* layer_layout;

};