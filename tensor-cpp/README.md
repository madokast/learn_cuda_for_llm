# tensor-cpp

一个基于CMake的C++项目，演示了C++17特性和CUDA编程，实现了向量加法的CPU和CUDA双版本实现，并支持条件编译。

## 项目结构

```
tensor-cpp/
├── include/                # 头文件目录
│   ├── cuda_utils.h        # CUDA工具函数和向量加法声明
│   └── greetings.h         # 问候语相关的结构体和函数声明
├── src/                    # 源文件目录
│   ├── cpu_vector_add.cpp  # 向量加法的CPU实现
│   ├── cuda_vector_add.cu  # 向量加法的CUDA实现
│   └── greetings.cpp       # 问候语相关函数的实现
├── tests/                  # 测试文件目录
│   ├── cpp/                # C++测试
│   │   └── test_greetings.cpp  # 测试问候语功能
│   └── cuda/               # CUDA测试
│       └── test_vector_add.cpp  # 测试向量加法功能
├── main.cpp                # 主程序入口，演示C++17结构化绑定
├── main_vector.cpp         # 向量加法主程序
└── CMakeLists.txt          # CMake构建配置文件
```

## 文件说明

### 1. 头文件 (include/)

#### include/greetings.h
- 定义了`GreetingInfo`结构体，包含问候信息
- 声明了`getGreeting()`函数，用于获取问候信息

#### include/cuda_utils.h
- 包含了向量加法的统一声明，同时支持CPU和CUDA实现
- 条件包含CUDA运行时头文件，仅在CUDA编译时生效
- 定义了`CUDA_CHECK`宏，用于CUDA错误检查

### 2. 源文件 (src/)

#### src/greetings.cpp
- 实现了`getGreeting()`函数，返回包含问候语和来源的结构体

#### src/cpu_vector_add.cpp
- 实现了`vectorAdd`函数的CPU版本
- 使用标准C++17实现向量加法

#### src/cuda_vector_add.cu
- 实现了`vectorAdd`函数的CUDA版本
- 包含CUDA核函数`vectorAddKernel`
- 处理设备内存分配、数据传输和核函数启动

### 3. 主程序文件

#### main.cpp
- 演示了C++17的结构化绑定特性
- 调用`getGreeting()`函数并打印结果

#### main_vector.cpp
- 向量加法的主程序
- 根据编译选项自动选择CPU或CUDA实现
- 生成随机向量，执行向量加法，验证结果

### 4. 测试文件 (tests/)

#### tests/cpp/test_greetings.cpp
- 测试`getGreeting()`函数的返回值
- 验证返回的结构体内容是否符合预期

#### tests/cuda/test_vector_add.cpp
- 测试向量加法功能
- 在CPU和CUDA环境下都能编译运行
- 生成已知值的向量，执行向量加法，验证结果

### 5. 构建配置文件

#### CMakeLists.txt
- 配置了C++17和CUDA 17标准
- 实现了CUDA自动检测功能
- 支持三种CUDA模式：AUTO、ON、OFF
- 条件编译逻辑，根据CUDA可用性选择不同的实现
- 配置了CTest测试框架
- 生成了两个主程序和三个测试目标

## 构建和运行

### 1. 基本构建

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### 2. 构建选项

- **ENABLE_CUDA**: 控制CUDA支持，可选值：AUTO（默认）、ON、OFF
  ```bash
  cmake .. -DENABLE_CUDA=ON  # 强制启用CUDA
  cmake .. -DENABLE_CUDA=OFF # 强制禁用CUDA
  ```

- **BUILD_TESTS**: 控制是否构建测试，默认ON
- **RUN_TESTS**: 控制构建后是否自动运行测试，默认ON

### 3. 运行程序

#### 运行问候语程序
```bash
./hello_world
```

#### 运行向量加法程序
```bash
./main_vector
```

### 4. 运行测试

#### 自动运行
构建后会自动运行测试（如果`RUN_TESTS=ON`）

#### 手动运行
```bash
ctest
```

## CUDA支持

项目支持自动检测CUDA环境：
- 当检测到`nvcc`编译器时，自动启用CUDA支持
- 否则，使用CPU版本的实现

### CUDA版本特性

- 使用CUDA 17标准
- 支持CUDA架构自动检测
- 实现了完整的CUDA错误检查
- 高效的核函数设计，使用256线程/块

## 技术特性

1. **C++17特性**
   - 结构化绑定
   - 自动类型推导

2. **模块化设计**
   - 分离的头文件和实现文件
   - 清晰的代码组织

3. **跨平台支持**
   - 基于CMake的构建系统
   - 支持Linux系统

4. **测试驱动开发**
   - 使用CTest测试框架
   - 全面的测试覆盖

5. **高性能计算**
   - 优化的CPU实现
   - 高效的CUDA实现

## 示例输出

### 问候语程序
```
Greeting: Hello, World!
Source: tensor-cpp
```

### 向量加法程序
```
Vector size: 1000000
Vector addition completed successfully!
Results match expected values.
```

## 测试结果

```
Test project /path/to/tensor-cpp/build
    Start 1: test_greetings
1/2 Test #1: test_greetings ...................   Passed    0.00 sec
    Start 2: test_vector_add_cpu
2/2 Test #2: test_vector_add_cpu ..............   Passed    0.00 sec

100% tests passed, 0 tests failed out of 2
```

在CUDA环境下，会额外运行CUDA版本测试：

```
Test project /path/to/tensor-cpp/build
    Start 1: test_greetings
1/3 Test #1: test_greetings ...................   Passed    0.00 sec
    Start 2: test_vector_add_cpu
2/3 Test #2: test_vector_add_cpu ..............   Passed    0.00 sec
    Start 3: test_vector_add_cuda
3/3 Test #3: test_vector_add_cuda .............   Passed    1.06 sec

100% tests passed, 0 tests failed out of 3
```

## 开发环境要求

- **编译器**: GCC 13.3.0 或更高版本
- **CUDA**: 可选，CUDA 11.0 或更高版本
- **CMake**: 3.10 或更高版本

## 许可证

MIT License
