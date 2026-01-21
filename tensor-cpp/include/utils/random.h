#pragma once
#include <cstdint>
#include <cstddef>

class FastRandom {
private:
    uint64_t state;
    
    uint64_t xorshift64star() noexcept {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }
    
public:
    FastRandom(uint64_t seed = 88172645463325252ULL) noexcept
        : state(seed) {}
    

    uint64_t next_uint64() noexcept {
        return xorshift64star();
    }

    // 生成 [0.0, 1.0) 之间的双精度浮点数
    double next_double() noexcept {
        // 使用高53位，保证均匀分布
        return (xorshift64star() >> 11) * (1.0 / (1ULL << 53));
    }

    // 生成 [0.0, 1.0) 之间的单精度浮点数
    float next_float() noexcept {
        return (xorshift64star() >> 40) * 0x1.0p-24f;  // 1.0f / (1 << 24)
    }
    
    // 批量生成
    void fill_uint64_array(uint64_t* arr, size_t n) noexcept {
        for (size_t i = 0; i < n; ++i) {
            arr[i] = next_uint64();
        }
    }

    void fill_double_array(double* arr, size_t n) noexcept {
        for (size_t i = 0; i < n; ++i) {
            arr[i] = next_double();
        }
    }

    void fill_float_array(float* arr, size_t n) noexcept {
        for (size_t i = 0; i < n; ++i) {
            arr[i] = next_float();
        }
    }
};
