#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include "define.h"
#include "cuda_runtime.h"
#include "kernel/transpose_kernel.h"
class Tensor {
public:
    void* data;
    size_t size; // Byte size.
    DataType dtype;
    std::vector<size_t> shape;
    std::string device;
    size_t num_elements;

    Tensor() : 
        data(nullptr), 
        size(0), 
        dtype(DataType::FLOAT16), 
        shape(), device("gpu"), 
        num_elements(0) {}

    Tensor(
        size_t num_elements, 
        void* data_ptr, 
        std::vector<size_t> shape, 
        DataType dtype, 
        std::string device = "gpu")
        : data(data_ptr),
          size(num_elements * element_size_bytes(dtype)),
          dtype(dtype),
          shape(std::move(shape)),
          device(std::move(device)),
          num_elements(num_elements) {}

    ~Tensor() {}

    Tensor(const Tensor& other): 
        data(nullptr), 
        size(other.size), 
        dtype(other.dtype), 
        shape(other.shape), 
        device(other.device), 
        num_elements(other.num_elements) {
        if (other.data != nullptr && size > 0) {
            if (device == "gpu") {
                cudaError_t alloc_err = cudaMalloc(&data, size);
                if (alloc_err != cudaSuccess) {
                    data = nullptr;
                    return;
                }
                cudaError_t copy_err = cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
                if (copy_err != cudaSuccess) {
                    cudaFree(data);
                    data = nullptr;
                }
            } else {
                data = new char[size];
                std::memcpy(data, other.data, size);
            }
        }
    }

    Tensor(Tensor&& other) noexcept
        : data(other.data),
          size(other.size),
          dtype(other.dtype),
          shape(std::move(other.shape)),
          device(std::move(other.device)),
          num_elements(other.num_elements) {
        other.data = nullptr;
        other.size = 0;
        other.num_elements = 0;
    }

    static size_t element_size_bytes(DataType dtype) {
        return dtype == DataType::FLOAT16 ? 2 : 4;
    }

    size_t numel() const {
        if(data == nullptr) {
            return 0;
        }
        return num_elements;
    }

    void view(std::vector<size_t> new_shape) {
        if (data != nullptr) {
            shape = std::move(new_shape);
        }
    }

    Tensor transpose() {
        //don't transpose if dim < 2
        if (shape.size() < 2) {
            return Tensor(numel(), nullptr, shape, dtype, device);
        }
        //new shape after transpose
        std::vector<size_t> new_shape = shape;
        std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);

        //transposed tensor
        Tensor out(numel(), nullptr, std::move(new_shape), dtype, device);
        
        if (data == nullptr || out.size == 0) {
            return out;
        }
        const size_t num_dims = shape.size();
        //last 2 dimensions
        const size_t rows = shape[num_dims - 2];
        const size_t cols = shape[num_dims - 1];
        //number of elements in the last 2 dimensions
        const size_t block_elems = rows * cols;
        //number of blocks in the outer dimensions
        const size_t outer_elems = (block_elems == 0) ? 0 : (num_elements / block_elems);


        if (device == "cpu") {
            out.data = new char[out.size];

            if (dtype == DataType::FLOAT32) {
                const float* old_data = static_cast<const float*>(data);
                float* new_data = static_cast<float*>(out.data);
                for (size_t o = 0; o < outer_elems; ++o) {
                    const float* in_block = old_data + o * block_elems;
                    float* out_block = new_data + o * block_elems;
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < cols; ++j) {
                            out_block[j * rows + i] = in_block[i * cols + j];
                        }
                    }
                }

            } else {
                const uint16_t* old_data = static_cast<const uint16_t*>(data);
                uint16_t* new_data = static_cast<uint16_t*>(out.data);
                for (size_t o = 0; o < outer_elems; ++o) {
                    const uint16_t* in_block = old_data + o * block_elems;
                    uint16_t* out_block = new_data + o * block_elems;
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < cols; ++j) {
                            out_block[j * rows + i] = in_block[i * cols + j];
                        }
                    }
                }
            }
            return out;
        }

        else{
            cudaError_t alloc_err = cudaMalloc(&out.data, out.size);
            if (alloc_err != cudaSuccess) {
                return Tensor();
            }

            launch_transpose_last2d_kernel(
                data,
                out.data,
                outer_elems,
                rows,
                cols,
                dtype
            );

            cudaError_t launch_err = cudaGetLastError();
            if (launch_err != cudaSuccess) {
                cudaFree(out.data);
                out.data = nullptr;
                return Tensor();
            }
            return out;
        }
    }

    bool operator==(const Tensor& other) const {
        if (size != other.size || shape != other.shape || dtype != other.dtype) {
            return false;
        }
        if (data == nullptr || other.data == nullptr) {
            return data == other.data;
        }
        return std::memcmp(data, other.data, size) == 0;
    }
};