#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(error), __LINE__, __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function declarations for CUDA vector addition
extern "C" void vectorAdd(const float* A, const float* B, float* C, int numElements);

#endif // CUDA_UTILS_H
