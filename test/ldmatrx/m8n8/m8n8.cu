#include <cuda_fp16.h> // half half2
#include <stdint.h> // Add this line to define uint32_t
#include <stdio.h>
const int CHECK = 1;
extern "C" {

__global__ void test(const half __restrict__ *m, half __restrict__ *n) {
  int tid = threadIdx.x;
  if (CHECK) printf("m[%d] = %.3f, m[%d] = %.3f\n", tid*2, float(m[tid*2]), tid*2+1, float(m[tid*2+1]));

  // 复制到 SMEM。先转为 uint32_t 类型
  __shared__ half2 sm[32];
  sm[tid] = ((half2 *)m)[tid];
  __syncthreads(); // 同步，供后续 ldmatrix

  // 检查一下
  if (CHECK) printf("tid = %d, sm[%d] = [%f, %f]\n", tid, tid, float(sm[tid].x), float(sm[tid].y));

  // 从 SMEM 加载 8*8 矩阵
  // 1. 找到本线程需要加载的元素索引，这里就是 sm + tid 执行的 half2 中的两个元素，存储到 m_ele
  uint32_t m_ele; // 这里必须用 uint32_t 而不是 half2 否则报错 an asm operand must have scalar type
  // 2. ldmatrix 实际提供行首指针，即 sm / sm+4 / sm+8...,一共 8 行，在 x1 模式下由 8 个线程负责指出位置
  //    实际搬运由 SM 内控制、加载硬件完成，且按照固定的 order 分配到寄存器文件中
  // 3. 将行首指针转为 shared memory 的 32 位指针。这里后续的 8-31 号线程等于重复工作
  uint32_t s_ptr = __cvta_generic_to_shared(sm + (tid % 8) * 4);
  // 3. 调用 ldmatrix 加载
  asm (
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
    : "=r"(m_ele)
    : "r"(s_ptr)
  );
  
  // 写入 n 矩阵中
  half2 m_ele_h2 = *((half2 *)(&m_ele));
  if (CHECK) printf("tid = %d, m_ele = [%f, %f]\n", tid, float(m_ele_h2.x), float(m_ele_h2.y));
  n[tid*2] = m_ele_h2.x, n[tid*2+1] = m_ele_h2.y;
}

} // end extern "C"