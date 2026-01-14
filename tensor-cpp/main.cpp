#include <iostream>
#include "greetings.h"

int main() {
    // Use structured binding to unpack the returned struct
    auto [message, source] = getGreeting();
    
    // Print the greeting information
    std::cout << message << std::endl;
    std::cout << "Greeting from: " << source << std::endl;
    
    return 0;
}
