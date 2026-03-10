#include "Layer.h"
#include "Tensor.h"
#include "Workspace.h"
#include "ModelWeights.h"
class Attention: public Layer {
    public:
        Attention(int hidden_size, int num_heads, LayerWeightLayout* layer_layout) : layer_layout(layer_layout) {}
        void prefill_forward(const Tensor& input, Tensor& output, Workspace& workspace) override{

        };
        void decode_forward(const Tensor& input, Tensor& output, Workspace& workspace) override{

        };
        void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
            // Implement logic to split the combined qkv tensor into separate q, k, v tensors

        }
        void qkv_projection(const Tensor& input, Tensor& qkv) {
            // Implement logic to project the input tensor into a combined qkv tensor
        }

        void compute_score(){

        }

        void softmax(){

        }
        void compute_context(){

        }

        void output_projection(){

        }
    private:
        LayerWeightLayout* layer_layout;

};