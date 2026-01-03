import pycuda.autoinit as _
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pathlib import Path

cur_dir = Path(__file__).parent / "m8n8.cu"

# 1. 编写 CUDA 核函数
with open(cur_dir) as f:
    cuda_code = f.read()

# 2. 编译核函数
mod = SourceModule(cuda_code, no_extern_c=True)
test_kernel = mod.get_function("test")

# 3. 准备数据（主机端 -> 设备端）
M = np.arange(8*8, dtype=np.float16) # 8*8
M = np.concatenate([M, M+100, M+200, M+300])
N = np.empty_like(M)

# 4. 配置线程块/网格，执行核函数
block_size = 32
grid_size = 1

test_kernel(cuda.In(M), cuda.Out(N),
            block=(block_size, 1, 1), grid=(grid_size, 1))
cuda.Context.synchronize()

print("============ input m ==============")
print(M.reshape((4,8,8)))
print("============ input n ==============")
print(N.reshape((4,8,8)))

assert np.allclose(M, N)
