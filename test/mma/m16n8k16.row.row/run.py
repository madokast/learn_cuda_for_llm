import pycuda.autoinit as _
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pathlib import Path

cur_dir = Path(__file__).parent / "code.cu"

# 1. 编写 CUDA 核函数
with open(cur_dir) as f:
    cuda_code = f.read()

# 2. 编译核函数
mod = SourceModule(cuda_code, no_extern_c=True)
test_kernel = mod.get_function("test_mma")

# 3. 准备数据（主机端 -> 设备端）
shape_A, shape_B, shape_C = (16, 16), (16, 8), (16, 8)
dtype_A, dtype_B, dtype_C = np.float16, np.float16, np.float32

A = np.arange(np.prod(shape_A), dtype=dtype_A).reshape(shape_A)
B = np.arange(np.prod(shape_B), dtype=dtype_B).reshape(shape_B)
C = np.empty(shape=shape_C, dtype=dtype_C)

# 4. 配置线程块/网格，执行核函数
block_size = 32
grid_size = 1

test_kernel(cuda.In(A), cuda.In(B), cuda.Out(C),
            block=(block_size, 1, 1), grid=(grid_size, 1))
cuda.Context.synchronize()

np.set_printoptions(linewidth=200)
print("=== A\n", A.astype(np.int32))
print("=== B\n", B.astype(np.int32))
print("=== C\n", C.astype(np.int32))

AB = A.astype(np.float32) @ B.astype(np.float32)
print("=== A @ B\n", AB.astype(np.int32))

assert np.allclose(AB, C)
print("SUCCESS: CUDA result matches Numpy result.")
