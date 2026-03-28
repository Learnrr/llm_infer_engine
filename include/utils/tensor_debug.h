#pragma once

#include "Tensor.h"
#include "utils/logger.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include "half_float/half.hpp"

inline void log_tensor_anomaly(const Tensor& tensor, const std::string& description) {
    if (tensor.data == nullptr || tensor.num_elements == 0) {
        return;
    }

    void* host_buf = std::malloc(tensor.size);
    if (host_buf == nullptr) {
        return;
    }

    cudaError_t e = cudaMemcpy(host_buf, tensor.data, tensor.size, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::free(host_buf);
        return;
    }

    size_t nan_count = 0;
    size_t inf_count = 0;
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    bool has_finite = false;
    if (tensor.dtype == DataType::FLOAT32) {
        const float* p = static_cast<const float*>(host_buf);
        for (size_t i = 0; i < tensor.num_elements; ++i) {
            const float v = p[i];
            if (std::isnan(v)) {
                ++nan_count;
            } else if (!std::isfinite(v)) {
                ++inf_count;
            } else {
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                has_finite = true;
            }
        }
    } else if (tensor.dtype == DataType::FLOAT16) {
        const half_float::half* p = static_cast<const half_float::half*>(host_buf);
        for (size_t i = 0; i < tensor.num_elements; ++i) {
            const float v = static_cast<float>(p[i]);
            if (std::isnan(v)) {
                ++nan_count;
            } else if (!std::isfinite(v)) {
                ++inf_count;
            } else {
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                has_finite = true;
            }
        }
    } else if (tensor.dtype == DataType::BF16) {
        const uint16_t* p = static_cast<const uint16_t*>(host_buf);
        for (size_t i = 0; i < tensor.num_elements; ++i) {
            const uint32_t fp32_bits = static_cast<uint32_t>(p[i]) << 16;
            float v = 0.0f;
            std::memcpy(&v, &fp32_bits, sizeof(v));
            if (std::isnan(v)) {
                ++nan_count;
            } else if (!std::isfinite(v)) {
                ++inf_count;
            } else {
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                has_finite = true;
            }
        }
    }

    if (nan_count > 0 || inf_count > 0) {
        std::ostringstream oss;
        oss << "Tensor stats [" << (description.empty() ? "" : description) << "]"
            << ": total=" << tensor.num_elements
            << " min=" << (has_finite ? min_val : 0.0f)
            << " max=" << (has_finite ? max_val : 0.0f)
            << " nan=" << nan_count
            << " inf=" << inf_count;
        LOG_INFO(oss.str());
    }

    std::free(host_buf);
}
