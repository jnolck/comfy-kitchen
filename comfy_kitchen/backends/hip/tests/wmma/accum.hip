// Quick diagnostic for accumulator layout
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <rocwmma/rocwmma.hpp>

__global__ void test_acc_layout()
{
        if (threadIdx.x >= 32) return;

        using A_frag = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major>;
        using B_frag = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major>;
        using Acc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

        __shared__ __half smem_a[256], smem_b[256];
        __shared__ float smem_out[256];

        // A: identity-like (each thread gets unique values)
        // B: all ones
        for (int i = threadIdx.x; i < 256; i += 32)
        {
                smem_a[i] = __float2half((float)i);
                smem_b[i] = __float2half(1.0f);
        }
        __syncthreads();

        A_frag a;
        B_frag b;
        Acc acc;
        for (int i = 0; i < 8; i++)
                acc.x[i] = 0.0f;

        rocwmma::load_matrix_sync(a, smem_a, 16);
        rocwmma::load_matrix_sync(b, smem_b, 16);
        rocwmma::mma_sync(acc, a, b, acc);
        rocwmma::store_matrix_sync(smem_out, acc, 16, rocwmma::mem_row_major);
        __syncthreads();

        if (threadIdx.x == 0)
        {
                printf("16x16 output matrix:\n");
                for (int r = 0; r < 16; r++)
                {
                        for (int c = 0; c < 16; c++)
                        {
                                printf("%6.0f ", smem_out[r * 16 + c]);
                        }
                        printf("\n");
                }
        }
}

int main()
{
        hipLaunchKernelGGL(test_acc_layout, dim3(1), dim3(32), 0, 0);
        hipDeviceSynchronize();
        return 0;
}
