# Set the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(tensor-cpp VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CUDA detection and configuration
option(ENABLE_CUDA "Enable CUDA support (AUTO, ON, OFF)" AUTO)

# Initialize CUDA availability flag
set(CUDA_AVAILABLE OFF)

# Check if CUDA should be enabled
if(ENABLE_CUDA STREQUAL "AUTO")
    # Find nvcc compiler
    find_program(NVCC_EXECUTABLE nvcc)
    if(NVCC_EXECUTABLE)
        message(STATUS "Found nvcc: ${NVCC_EXECUTABLE}")
        set(CUDA_AVAILABLE ON)
    else()
        message(STATUS "nvcc not found, disabling CUDA support")
        set(CUDA_AVAILABLE OFF)
    endif()
elseif(ENABLE_CUDA)
    # User explicitly enabled CUDA
    message(STATUS "CUDA explicitly enabled by user")
    set(CUDA_AVAILABLE ON)
else()
    # User explicitly disabled CUDA
    message(STATUS "CUDA explicitly disabled by user")
    set(CUDA_AVAILABLE OFF)
endif()

# If CUDA is available, enable CUDA language support
if(CUDA_AVAILABLE)
    enable_language(CUDA)
    
    # Specify the CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    
    # Set CUDA minimum version
    set(CUDA_MIN_VERSION 11.0)
endif()

# Add executable target for the original C++ program
add_executable(hello_world main.cpp src/greetings.cpp)

# Set include directories for the original program
target_include_directories(hello_world PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add CUDA executable target for vector_add only if CUDA is available
if(CUDA_AVAILABLE)
    # Add CUDA executable target for vector_add
    add_executable(cuda_vector_add main_cuda.cu src/cuda_vector_add.cu)
    
    # Set include directories for CUDA program
    target_include_directories(cuda_vector_add PUBLIC ${PROJECT_SOURCE_DIR}/include)
    
    # Set CUDA architecture (optional, but recommended)
    set_target_properties(cuda_vector_add PROPERTIES CUDA_ARCHITECTURES native)
    
    # Link CUDA runtime library if needed
    # target_link_libraries(cuda_vector_add PUBLIC cudart)
endif()

# Enable testing with CTest
option(BUILD_TESTS "Build tests" ON)
option(RUN_TESTS "Run tests after build" ON)

if(BUILD_TESTS)
    # Enable CTest
    enable_testing()
    
    # Add C++ test for getGreeting function
    add_executable(test_greetings tests/cpp/test_greetings.cpp src/greetings.cpp)
    target_include_directories(test_greetings PUBLIC ${PROJECT_SOURCE_DIR}/include)
    
    # Add test to CTest
    add_test(NAME test_greetings COMMAND test_greetings)
    
    # Initialize list of test targets to run
    set(TEST_TARGETS test_greetings)
    
    # Add CUDA test for vectorAdd function only if CUDA is available
    if(CUDA_AVAILABLE)
        add_executable(test_vector_add tests/cuda/test_vector_add.cu src/cuda_vector_add.cu)
        target_include_directories(test_vector_add PUBLIC ${PROJECT_SOURCE_DIR}/include)
        set_target_properties(test_vector_add PROPERTIES CUDA_ARCHITECTURES native)
        
        # Add test to CTest
        add_test(NAME test_vector_add COMMAND test_vector_add)
        
        # Add to test targets list
        list(APPEND TEST_TARGETS test_vector_add)
    endif()
    
    # Run tests after build if RUN_TESTS is ON
    if(RUN_TESTS)
        add_custom_command(
            TARGET ${TEST_TARGETS} POST_BUILD
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running tests..."
        )
    endif()
endif()

