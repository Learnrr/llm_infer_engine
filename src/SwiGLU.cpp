#include "SwiGLU.h"

void SwiGLU::forward(Tensor& input, Tensor& output, ForwardContext& context) {
    // Implement the logic for the forward pass of the SwiGLU activation function
    // This typically involves splitting the input tensor into two parts, applying the sigmoid function to one part, and then performing element-wise multiplication with the other part.
    swiglu_kernel(input.data, output.data);
}