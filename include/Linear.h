#pragma once
#include "define.h"
#include "Tensor.h"

class Linear {
    public:
        Linear(int input_size, int output_size);
        void forward(Tensor& input, Tensor& output);
    private:

};