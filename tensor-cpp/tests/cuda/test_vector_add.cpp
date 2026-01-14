#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "cuda_utils.h"

int main() {
    // Define test vector size (smaller for faster testing)
    const int numElements = 10000;
    std::cout << "Testing vectorAdd with " << numElements << " elements..." << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);
    std::vector<float> h_C_expected(numElements);
    
    // Initialize host vectors with known values
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
        h_C_expected[i] = h_A[i] + h_B[i];
    }
    
    // Call the CUDA vectorAdd function
    vectorAdd(h_A.data(), h_B.data(), h_C.data(), numElements);
    
    // Verify the result
    bool testPassed = true;
    float epsilon = 1e-5f; // Allow small floating point error
    
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C[i] - h_C_expected[i]) > epsilon) {
            std::cerr << "Test failed at index " << i << ": Expected " << h_C_expected[i] 
                      << ", got " << h_C[i] << std::endl;
            testPassed = false;
            break; // Stop after first failure to save time
        }
    }
    
    if (testPassed) {
        std::cout << "Test passed: vectorAdd produced correct results" << std::endl;
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
