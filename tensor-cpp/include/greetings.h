#ifndef GREETINGS_H
#define GREETINGS_H

#include <string>

// Define a struct to hold greeting information
struct GreetingInfo {
    std::string message;
    std::string source;
};

// Function declaration to get greeting information
GreetingInfo getGreeting();

#endif // GREETINGS_H
