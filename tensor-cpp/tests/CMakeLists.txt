# 引入 googletest
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# 定义测试可执行文件
add_executable(unit_tests test_tensor.cpp)

# 链接核心库 和 GTest
# PRIVATE：仅当前目标使用，不传播给依赖它的其他目标
# 如果其他目标链接到 unit_tests，它们不会自动获得这些库
target_link_libraries(unit_tests PRIVATE tensor-core GTest::gtest_main)

# 注册到 CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)