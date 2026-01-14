#include <iostream>
#include <string>
#include "greetings.h"

int main() {
    // Call the getGreeting function
    auto [message, source] = getGreeting();
    
    // Expected values
    const std::string expectedMessage = "Hello, World!";
    const std::string expectedSource = "tensor-cpp project";
    
    // Test if the returned values match the expected values
    bool testPassed = true;
    
    if (message != expectedMessage) {
        std::cerr << "Test failed: Expected message '" << expectedMessage << "', got '" << message << "'" << std::endl;
        testPassed = false;
    }
    
    if (source != expectedSource) {
        std::cerr << "Test failed: Expected source '" << expectedSource << "', got '" << source << "'" << std::endl;
        testPassed = false;
    }
    
    if (testPassed) {
        std::cout << "Test passed: getGreeting() returned expected values" << std::endl;
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
