#include <cuda_fp16.h> // half half2
#include <stdint.h> // Add this line to define uint32_t
#include <stdio.h> // print

const int CHECK = 1;

extern "C" {

// A 16*16 矩阵，B 和 C 16*8 矩阵，完成 C = AB 计算
__global__ void test_mma(const half *__restrict__ A, const half *__restrict__ B, float* __restrict__ C) {
    const int tid = threadIdx.x;

    constexpr int a_num = 16 * 16, b_num = 16 * 8, c_num = 16 * 8;
    constexpr int share_a_num = a_num * sizeof(half) / sizeof(half2); // 128
    constexpr int share_b_num = b_num * sizeof(half) / sizeof(half2); // 64
    __shared__ alignas(16) half2 s[share_a_num + share_b_num]; // 192

    {   // 复制数据到 share，共 192 个 u32 元素，正好复制 192/32=6 轮次，A 4 轮，B 2 轮
        for (int lid = 0; lid < 4; lid++) {
            const half2 *src = (const half2*)(A) + tid + lid*32;
            half2 *des = s + tid + lid*32;
            *des = *src;
            if (CHECK) printf("A tid(%d) ele(%f, %f)\n", tid, float(des->x), float(des->y));
        }
        for (int lid = 0; lid < 2; lid++) {
            const half2 *src = (const half2*)(B) + tid + lid*32;
            half2 *des = s + share_a_num + tid + lid*32;
            *des = *src;
            if (CHECK) printf("B tid(%d) ele(%f, %f)\n", tid, float(des->x), float(des->y));
        }
        __syncthreads(); // 同步，供后续 ldmatrix
    }

    // ldmatrix 将 A 矩阵数据复制到寄存器
    uint32_t a1, a2, a3, a4; // 每个线程负责 128/32=4 个 half2 元素，这里用 u32 避免 PTX 报错
    {   // 看起来需要采用 m8n8.x4 复制 4 次，正好把 16*16 矩阵切分为 4 个 8*8 矩阵，按照左上、左下、右上、右下的顺序
        // 线程 0-7 负责 A 矩阵左上角 8*8，拷贝时给出行首元素地址 [0-7, 0]
        // 线程 8-15 负责左下角，首元素地址 [8-15, 0] // 4 因为 8 个 f16 元素合并为 4 个 u32 元素
        // 16-23 负责右上角，首元素地址 [0-7, 4]
        // 24-31 负责右下角角，首元素地址 [8-15, 4]
        const int x = tid % 16; // 首元素行号 x
        const int y = (tid / 16) * 4; //  首元素 y
        const int offset = x * 8 + y; // 首元素偏移，一行有 16/2=8个 u32
        uint32_t s_ptr = __cvta_generic_to_shared(s + offset); // 每个线程负责拷贝 A 矩阵的一行元素
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
            : "r"(s_ptr)
        );
        
        if (CHECK) {
            half2 h1=*((half2*)(&a1)), h2=*((half2*)(&a2)), h3=*((half2*)(&a3)), h4=*((half2*)(&a4));
            printf("A tid(%d) ld (%f, %f), (%f, %f), (%f, %f), (%f, %f)\n", tid, 
                float(h1.x), float(h1.y), float(h2.x), float(h2.y), float(h3.x), float(h3.y), float(h4.x), float(h4.y));
        }
    }

    // ldmatrix 复制 B 矩阵
    uint32_t b1, b2; // 每个线程负责 64/32=2 个 half2 元素，这里用 u32 避免 PTX 报错
    {   // 需要用 m8n8.x2，将 16*8 拆分为上下两个 8*8
        // 这里让 16-31 号线程，携带和 0-15 相同的参数
        const int x = tid % 16; // 首元素行号 x
        const int y = 0; //  首元素列号 y
        const int offset = x * 4 + y; // 首元素偏移，一行有 8/2=4个 u32
        uint32_t s_ptr = __cvta_generic_to_shared(s + share_a_num + offset); // 每个线程负责拷贝 B 矩阵的一行元素
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
            : "=r"(b1), "=r"(b2)
            : "r"(s_ptr)
        );
        
        if (CHECK) {
            half2 h1=*((half2*)(&b1)), h2=*((half2*)(&b2));
            printf("B tid(%d) ld (%f, %f), (%f, %f)\n", tid, 
                float(h1.x), float(h1.y), float(h2.x), float(h2.y));
        }
    }

    // mma 矩阵乘法
    float c1=0, c2=0, c3=0, c4=0;
    { 
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            " { %0, %1, %2, %3 }, "
            " { %4, %5, %6, %7 }, "
            " { %8, %9 }, "
            " { %10, %11, %12, %13 }; "
            : "=f"(c1), "=f"(c2), "=f"(c3), "=f"(c4)
            : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
            "f"(c1), "f"(c2), "f"(c3), "f"(c4));
        if (CHECK) printf("mma tid(%d) %f, %f, %f, %f\n", tid, c1, c2, c3, c4);
    }

    // 将结果写回 C
    {
        // c1 和 c2 是连续的，视为 f64，填补 C 的上面 8*8 部分，则一行 4 个线程负责放置 4 个，共 8 组线程完成 8 行的放置
        const int x = tid / 4; // 行
        const int y = (tid % 4) * 2; // 列
        const int offset_up = x * 8 + y;
        C[offset_up] = c1; C[offset_up+1] = c2;

        // c3 和 c4 连续，填补 C 的下面 8*8 部分
        const int offset_down = 8*8 + offset_up;
        C[offset_down] = c3; C[offset_down+1] = c4;
    }
}

} // extern "C"

