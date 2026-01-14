#include <iostream>
#include <vector>
#include <cstdlib>
#include "cuda_utils.h"

int main() {
    // Define vector size
    const int numElements = 1000000;
    std::cout << "Vector size: " << numElements << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);
    std::vector<float> h_C_expected(numElements);
    
    // Initialize host vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C_expected[i] = h_A[i] + h_B[i];
    }
    
    // Call CUDA vector addition function
    vectorAdd(h_A.data(), h_B.data(), h_C.data(), numElements);
    
    // Verify the result
    bool resultCorrect = true;
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C[i] - h_C_expected[i]) > 1e-5) {
            resultCorrect = false;
            break;
        }
    }
    
    // Print verification result
    if (resultCorrect) {
        std::cout << "Vector addition completed successfully!" << std::endl;
        std::cout << "Results match expected values." << std::endl;
    } else {
        std::cerr << "ERROR: Vector addition results do not match expected values!" << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
