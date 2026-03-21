/*
cd tests
nvcc -std=c++17 -O2 -I../include test_tensor.cpp ../kernel/transpose_kernel.cu -o ../build/tests/test_tensor.exe
../build/tests/test_tensor.exe
*/

#include "Tensor.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace {

void FreeCpuTensorData(Tensor& t) {
    if (t.device == "cpu" && t.data != nullptr) {
        delete[] static_cast<char*>(t.data);
        t.data = nullptr;
    }
}

void FreeGpuTensorData(Tensor& t) {
    if (t.device == "gpu" && t.data != nullptr) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }
    return count > 0;
}

void TestViewUpdatesShape() {
    std::vector<float> storage = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(
        storage.size(),
        storage.data(),
        {2, 3},
        DataType::FLOAT32,
        "cpu"
    );

    t.view({3, 2});
    assert(t.shape.size() == 2);
    assert(t.shape[0] == 3);
    assert(t.shape[1] == 2);
}

void TestTransposeCpuFloat32_2D() {
    std::vector<float> storage = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    Tensor t(
        storage.size(),
        storage.data(),
        {2, 3},
        DataType::FLOAT32,
        "cpu"
    );

    Tensor out = t.transpose();

    assert(out.data != nullptr);
    assert(out.shape.size() == 2);
    assert(out.shape[0] == 3);
    assert(out.shape[1] == 2);

    const float* got = static_cast<const float*>(out.data);
    const std::vector<float> expected = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(got[i] == expected[i]);
    }

    FreeCpuTensorData(out);
}

void TestTransposeCpuFloat16_3DLast2DimsOnly() {
    // Shape [2, 2, 3], transpose only last 2 dims -> [2, 3, 2].
    std::vector<uint16_t> storage = {
        // block 0: [ [1,2,3], [4,5,6] ]
        1, 2, 3,
        4, 5, 6,
        // block 1: [ [7,8,9], [10,11,12] ]
        7, 8, 9,
        10, 11, 12
    };
    assert(storage.data() != nullptr);
    Tensor t(
        storage.size(),
        storage.data(),
        {2, 2, 3},
        DataType::FLOAT16,
        "cpu"
    );

    assert(t.data != nullptr);

    Tensor out = t.transpose();

    assert(out.data != nullptr);
    assert(out.shape.size() == 3);
    assert(out.shape[0] == 2);
    assert(out.shape[1] == 3);
    assert(out.shape[2] == 2);

    const uint16_t* got = static_cast<const uint16_t*>(out.data);
    const std::vector<uint16_t> expected = {
        // block 0 transposed: [ [1,4], [2,5], [3,6] ]
        1, 4,
        2, 5,
        3, 6,
        // block 1 transposed: [ [7,10], [8,11], [9,12] ]
        7, 10,
        8, 11,
        9, 12
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(got[i] == expected[i]);
    }

    FreeCpuTensorData(out);
}

void TestTransposeGpuFloat32_2D() {
    std::vector<float> host_in = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const size_t bytes = host_in.size() * sizeof(float);

    void* device_in = nullptr;
    cudaError_t alloc_err = cudaMalloc(&device_in, bytes);
    assert(alloc_err == cudaSuccess);

    cudaError_t copy_err = cudaMemcpy(device_in, host_in.data(), bytes, cudaMemcpyHostToDevice);
    assert(copy_err == cudaSuccess);

    Tensor t(
        host_in.size(),
        device_in,
        {2, 3},
        DataType::FLOAT32,
        "gpu"
    );

    Tensor out = t.transpose();
    assert(out.data != nullptr);
    assert(out.shape.size() == 2);
    assert(out.shape[0] == 3);
    assert(out.shape[1] == 2);

    std::vector<float> host_out(host_in.size(), 0.0f);
    cudaError_t back_err = cudaMemcpy(host_out.data(), out.data, bytes, cudaMemcpyDeviceToHost);
    assert(back_err == cudaSuccess);

    const std::vector<float> expected = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(host_out[i] == expected[i]);
    }

    cudaFree(device_in);
    FreeGpuTensorData(out);
}

void TestTransposeGpuFloat16_3DLast2DimsOnly() {
    std::vector<uint16_t> host_in = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    const size_t bytes = host_in.size() * sizeof(uint16_t);

    void* device_in = nullptr;
    cudaError_t alloc_err = cudaMalloc(&device_in, bytes);
    assert(alloc_err == cudaSuccess);

    cudaError_t copy_err = cudaMemcpy(device_in, host_in.data(), bytes, cudaMemcpyHostToDevice);
    assert(copy_err == cudaSuccess);

    Tensor t(
        host_in.size(),
        device_in,
        {2, 2, 3},
        DataType::FLOAT16,
        "gpu"
    );

    Tensor out = t.transpose();
    assert(out.data != nullptr);
    assert(out.shape.size() == 3);
    assert(out.shape[0] == 2);
    assert(out.shape[1] == 3);
    assert(out.shape[2] == 2);

    std::vector<uint16_t> host_out(host_in.size(), 0);
    cudaError_t back_err = cudaMemcpy(host_out.data(), out.data, bytes, cudaMemcpyDeviceToHost);
    assert(back_err == cudaSuccess);

    const std::vector<uint16_t> expected = {
        1, 4,
        2, 5,
        3, 6,
        7, 10,
        8, 11,
        9, 12
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(host_out[i] == expected[i]);
    }

    cudaFree(device_in);
    FreeGpuTensorData(out);
}

} // namespace

int main() {
    TestViewUpdatesShape();
    TestTransposeCpuFloat32_2D();
    TestTransposeCpuFloat16_3DLast2DimsOnly();

    if (HasCudaDevice()) {
        TestTransposeGpuFloat32_2D();
        TestTransposeGpuFloat16_3DLast2DimsOnly();
    } else {
        std::cout << "GPU tests skipped: no CUDA device available\n";
    }

    std::cout << "test_tensor passed\n";
    return 0;
}
