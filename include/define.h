
#pragma once

#include <cstddef>

enum class DataType {
    FLOAT32,
    FLOAT16,
    BF16
};

inline constexpr size_t DataTypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::BF16:
            return 2;
    }
    return 0;
}

inline constexpr const char* DataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return "FLOAT32";
        case DataType::FLOAT16:
            return "FLOAT16";
        case DataType::BF16:
            return "BF16";
    }
    return "UNKNOWN";
}

