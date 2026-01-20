#include <benchmark/benchmark.h>
#include "tensor.h" // 你的头文件

// 1. 定义测试函数，参数必须是 benchmark::State&
static void BM_Tensor_MatMul_Naive(benchmark::State& state) {
    // A. 准备阶段 (Setup)
    // 这里的代码只会运行一次，不计入计时
    int N = state.range(0); // 从参数里获取矩阵大小
    Tensor2D A(N, N);
    Tensor2D B(N, N);
    Tensor2D C(N, N);
    // 初始化数据...
    A.random(N);
    B.random(N+1);

    // B. 计时循环 (The Loop)
    // 这里的代码会被运行成千上万次
    for (auto _ : state) {
        // 这里是你要测的核心代码
        matmul_naive(A, B, C);
        
        // C. 防优化 (见下文)
        benchmark::DoNotOptimize(C);
    }
}

// // 2. 注册测试用例
// // ->Arg(N) 表示传入参数 N
// BENCHMARK(BM_Tensor_MatMul_Naive)
//     ->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256);


// // 3. 生成 main 函数
// BENCHMARK_MAIN();

// # 通过环境变量控制
// export BENCH_MIN_SIZE=16
// export BENCH_MAX_SIZE=1024
// ./your_benchmark
// 在main函数中读取环境变量
int main(int argc, char** argv) {
    const char* min_env = std::getenv("BENCH_MIN_SIZE");
    const char* max_env = std::getenv("BENCH_MAX_SIZE");
    
    int min_size = min_env ? std::atoi(min_env) : 2;
    int max_size = max_env ? std::atoi(max_env) : 256;
    
    // 动态注册
    BENCHMARK(BM_Tensor_MatMul_Naive)
        ->RangeMultiplier(2)
        ->Range(min_size, max_size);
    
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}