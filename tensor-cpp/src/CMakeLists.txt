# 定义库，包含源文件
add_library(tensor-core tensor.cpp)

# 设置头文件搜索路径
# PUBLIC 关键字意味着：
# 任何连接到 tensor-core 的目标（比如测试程序），
# 都能自动获得这些头文件路径。
target_include_directories(tensor-core PUBLIC ${CMAKE_SOURCE_DIR}/include)