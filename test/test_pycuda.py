import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# 1. 编写 CUDA 核函数（C 语法）
cuda_code = """
__device__ void add(float *a, float *b, float *c) {
  // *c = *a + *b;
  float va, vb, vc;
  asm("ld.global.f32 %0, [%1];" : "=f"(va) : "l"(a)); // va = *a;
  asm("ld.global.f32 %0, [%1];" : "=f"(vb) : "l"(b)); // vb = *b;
  asm("add.f32 %0, %1, %2;" : "=f"(vc) : "f"(va), "f"(vb)); // vc = va + vb;
  asm("st.global.f32 [%0], %1;" : : "l"(c), "f"(vc)); // *c = vc;
}

__global__ void vec_add(float *a, float *b, float *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
      add(a+idx, b+idx, c+idx);
  }
}
"""

# 2. 编译核函数
mod = SourceModule(cuda_code)
vec_add_kernel = mod.get_function("vec_add")

# 3. 准备数据（主机端 -> 设备端）
n = 1024
np.random.seed(42)
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.empty_like(a)

# 4. 配置线程块/网格，执行核函数
block_size = 256
grid_size = (n + block_size - 1) // block_size
vec_add_kernel(
    cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(n),
    block=(block_size, 1, 1), grid=(grid_size, 1)
)

# 验证结果
print("cuda result: ", c)
print("numpy result:", a + b)
assert np.allclose(c, a + b)