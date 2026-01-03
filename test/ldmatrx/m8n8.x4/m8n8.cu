#include <cuda_fp16.h> // half half2
#include <stdint.h> // Add this line to define uint32_t
#include <stdio.h>

extern "C" {

// m 和 n 长度 4*8*8，即 4 个 8*8 矩阵
__global__ void test(const half __restrict__ *m, half __restrict__ *n) {
  constexpr int m_num = 4*8*8; // m 矩阵元素数目
  constexpr int cp_num = m_num / 32; // 每个线程负责拷贝元素数目

  int tid = threadIdx.x;

  // 声明共享内存，强制 16 字节对齐，满足 cp.async 和 ldmatrix 要求
  __shared__ alignas(16) half sm[m_num];
  {
    // 复制到 SMEM
    // 计算本线程负责的复制起始地址
    const half* g_ptr = &m[tid * cp_num]; 
    uint32_t s_ptr = __cvta_generic_to_shared(&sm[tid * cp_num]);
    asm volatile (
      // [1] 异步提交拷贝任务
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      // [2] 提交当前批次
      "cp.async.commit_group;\n"
      // [3] 等待直到只剩下 0 组未完成 (即全部完成)
      "cp.async.wait_group 0;\n"
      // [4] 线程同步 (防止部分线程读到未写完的数据)
      "bar.sync 0;\n"
      : // 写入操作数为空
      : "r"(s_ptr), "l"(g_ptr) // 输入操作数
      : "memory"               // 告诉编译器内存已被修改
    );
  }

  // 从 SMEM 加载 4 个 8*8 矩阵
  // 1. 声明本线程需要加载的元素，m1_ele 即 sm + tid*2 指向的两个 half 元素，m2_ele 后续即下一个矩阵同位置
  uint32_t m1_ele, m2_ele, m3_ele, m4_ele; // 这里必须用 uint32_t 而不是 half2 否则报错 an asm operand must have scalar type
  // 2. ldmatrix 实际提供行首指针，即首行 sm、下一行 sm+8、再下一行 sm+16...，每行长度 8 个元素，32 个线程一起完成行指针的指定
  //    实际搬运由 SM 内其他硬件完成，且按照固定的 order 分配到寄存器文件中，所以指令中源地址和目标地址不是直接匹配的
  // 3. 将行首指针转为 shared memory 的 32 位指针
  uint32_t s_ptr = __cvta_generic_to_shared(sm + tid * 8);
  // 3. 调用 ldmatrix 加载
  asm (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(m1_ele), "=r"(m2_ele), "=r"(m3_ele), "=r"(m4_ele)
    : "r"(s_ptr)
  );
  
  // 写入 n 矩阵中
  half2 h2; int offset;
  h2 = *((half2 *)(&m1_ele)); offset = 0;
  n[offset + tid*2] = h2.x, n[offset + tid*2+1] = h2.y;

  h2 = *((half2 *)(&m2_ele)); offset = 8*8;
  n[offset + tid*2] = h2.x, n[offset + tid*2+1] = h2.y;

  h2 = *((half2 *)(&m3_ele)); offset = 8*8*2;
  n[offset + tid*2] = h2.x, n[offset + tid*2+1] = h2.y;

  h2 = *((half2 *)(&m4_ele)); offset = 8*8*3;
  n[offset + tid*2] = h2.x, n[offset + tid*2+1] = h2.y;
}

} // end extern "C"