#include "define.h"
#include "Tensor.h"
#include "Attention.h"
#include "MLP.h"
#include "ModelWeights.h"
#include "Workspace.h"

class TransformerLayer: public Layer {
    public:
        TransformerLayer(
            int hidden_size, 
            int num_heads, 
            LayerWeightLayout* layer_layout
        ) {
            attention = std::make_unique<Attention>(hidden_size, num_heads);
            mlp = std::make_unique<MLP>(hidden_size, INTERMEDIATE_SIZE);
            this->layer_layout = layer_layout;
        }

        void prefill_forward(const Tensor& input, Tensor& output, Workspace& workspace) override {
            // Implement the logic for the prefill forward pass of the transformer layer
            Tensor attn_output(input.size, input.data, input.shape, input.dtype);
            attention->prefill_forward(input, attn_output, workspace);
            mlp->prefill_forward(attn_output, output);
        }

        void decode_forward(const Tensor& input, Tensor& output, Workspace& workspace) override {
            // Implement the logic for the decode forward pass of the transformer layer
            Tensor attn_output(input.size, input.data, input.shape, input.dtype);
            attention->decode_forward(input, attn_output);
            mlp->decode_forward(attn_output, output);
        }

    private:
        std::unique_ptr<Attention> attention;
        std::unique_ptr<MLP> mlp;
        LayerWeightLayout* layer_layout;


}