#pragma once
#include <chrono>


inline size_t current_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

inline size_t ns_to_ms(size_t ns) {
    return ns / 1'000'000;
}

