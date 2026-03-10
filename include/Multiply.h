#pragma once
#include "Tensor.h"

class Multiply {
    public:
        Multiply(int hidden_size);
        void forward(Tensor& input1, Tensor& input2, Tensor& output);
    private:

};