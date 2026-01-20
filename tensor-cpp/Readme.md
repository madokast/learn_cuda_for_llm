

## 构建

cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

cmake --build build

sudo apt install linux-tools-common linux-tools-generic

export BENCH_MIN_SIZE=512 ; export BENCH_MAX_SIZE=512 ; perf stat ./build/bin/tensor_matmul_naive