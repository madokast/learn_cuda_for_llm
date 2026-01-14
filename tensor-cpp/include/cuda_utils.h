#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>

// Function declarations for vector addition (both CPU and CUDA implementations)
extern "C" void vectorAdd(const float* A, const float* B, float* C, int numElements);

// CUDA-specific code - only included when compiling with CUDA
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    
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
#endif

#endif // CUDA_UTILS_H
