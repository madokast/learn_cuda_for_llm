#include "cuda_utils.h"

// CUDA Kernel function to add the elements of two vectors
__global__ void vectorAddKernel(float* A, float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Host function to call the CUDA kernel for vector addition
extern "C" void vectorAdd(const float* h_A, const float* h_B, float* h_C, int numElements) {
    int size = numElements * sizeof(float);
    
    // Allocate device memory
    float* d_A = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    
    float* d_B = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    
    float* d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    
    // Copy input vectors from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
