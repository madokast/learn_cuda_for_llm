# 引入 googletest
include(FetchContent)

# 2. Google Benchmark
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Disable benchmark tests and install to speed up build
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(matmul_naive matmul_naive.cpp)
add_executable(matmul_naive_ijk matmul_naive_ijk.cpp)

# 链接你的核心库 和 benchmark 库
target_link_libraries(matmul_naive PRIVATE 
    tensor-core      # 你的业务代码
    benchmark::benchmark # Google Benchmark
)

target_link_libraries(matmul_naive_ijk PRIVATE 
    tensor-core      # 你的业务代码
    benchmark::benchmark # Google Benchmark
)
