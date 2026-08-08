// test_awq_kernel.cpp - standalone test
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <rocwmma/rocwmma.hpp>
#include <vector>

// Copy your to_fp32/from_fp32 and the awq_w4a16_mma_kernel here
// Or just include the relevant parts

// Known test dimensions
constexpr int M = 16;
constexpr int N = 128;
constexpr int K = 256;
constexpr int G = 64;

constexpr int kGemvMThreshold = 8;  // M ≤ 8 routes to gemv (naive) kernel

template <typename T>
__device__ __forceinline__ float to_fp32(T v);

template <>
__device__ __forceinline__ float to_fp32<__hip_bfloat16>(__hip_bfloat16 v)
{
        return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_fp32<__half>(__half v)
{
        return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_fp32(float v);

template <>
__device__ __forceinline__ __hip_bfloat16 from_fp32<__hip_bfloat16>(float v)
{
        return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __half from_fp32<__half>(float v)
{
        return __float2half(v);
}

// awq_w4a16_mma_kernel - rocWMMA version for AMD GPUs
template <typename T, typename WmmaT>
__global__ void awq_w4a16_mma_kernel(const T* __restrict__ x,             // (M, K)
                                     const int8_t* __restrict__ qweight,  // (N, K/2)
                                     const T* __restrict__ wscales,       // (K/G, N)
                                     const T* __restrict__ wzeros,        // (K/G, N)
                                     T* __restrict__ out,                 // (M, N)
                                     int M, int N, int K, int G)
{
        // Tile: 16×128 output per CTA, K=64 per iteration
        constexpr int BLOCK_M = 16;
        constexpr int BLOCK_N = 128;
        constexpr int BLOCK_K = 64;
        constexpr int NUM_WARPS = 4;                 // 128 threads = 4 warps
        constexpr int WARP_N = BLOCK_N / NUM_WARPS;  // 32 N per warp

        static_assert(BLOCK_K == 64, "one quantization group per K-tile");
        static_assert(BLOCK_N == 128, "BLOCK_N must be 128");

        // Shared memory for dequantized weights (fp16/bf16) and activations
        __shared__ T w_sh[BLOCK_N * BLOCK_K];  // 128×64 = 8192 elements
        __shared__ T x_sh[BLOCK_M * BLOCK_K];  // 16×64 = 1024 elements

        const int cta_n = blockIdx.x * BLOCK_N;
        const int cta_m = blockIdx.y * BLOCK_M;
        const int tid = threadIdx.x;
        const int K_half = K / 2;
        const int n_groups = K / G;

        // Accumulator fragment for this warp's portion
        // Each warp handles 16 M × 32 N output
        // Using 16×16 WMMA, that's 2 N-tiles per warp
        using Acc_frag = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
        Acc_frag acc[2];  // 2 N-tiles of 16 columns each

// Initialize accumulators to zero
#pragma unroll
        for (int n = 0; n < 2; n++)
        {
#pragma unroll
                for (int i = 0; i < 8; i++)
                {
                        acc[n].x[i] = 0.0f;
                }
        }

        for (int g = 0; g < n_groups; g++)
        {
                const int k_base = g * G;
                const int kh_base = k_base / 2;

                // Step 1: Dequantize int4 weights to fp16/bf16 in shared memory
                // 128 threads, each handles one N column (BLOCK_N = 128 = num threads)
                const int n_in_cta = tid;
                const int n_global = cta_n + n_in_cta;

                if (n_global < N)
                {
                        const float scale = to_fp32(wscales[g * N + n_global]);
                        const float zero = to_fp32(wzeros[g * N + n_global]);
                        const int8_t* qw_row = qweight + n_global * K_half + kh_base;

#pragma unroll
                        for (int kh = 0; kh < BLOCK_K / 2; kh++)
                        {
                                uint8_t byte = static_cast<uint8_t>(qw_row[kh]);
                                int lo = byte & 0xF;
                                int hi = (byte >> 4) & 0xF;

                                w_sh[n_in_cta * BLOCK_K + 2 * kh] =
                                    from_fp32<T>((float(lo) - 8.0f) * scale + zero);
                                w_sh[n_in_cta * BLOCK_K + 2 * kh + 1] =
                                    from_fp32<T>((float(hi) - 8.0f) * scale + zero);
                        }
                }
                else
                {
// Out of bounds - zero out
#pragma unroll
                        for (int kk = 0; kk < BLOCK_K; kk++)
                        {
                                w_sh[n_in_cta * BLOCK_K + kk] = T(0.0f);
                        }
                }

// Step 2: Load activations into shared memory
// 16×64 = 1024 elements, 128 threads = 8 elements per thread
#pragma unroll
                for (int i = tid; i < BLOCK_M * BLOCK_K; i += BLOCK_N)
                {
                        int mm = i / BLOCK_K;
                        int kk = i % BLOCK_K;
                        int m_global = cta_m + mm;
                        int k_global = k_base + kk;
                        x_sh[i] = (m_global < M) ? x[m_global * K + k_global] : T(0.0f);
                }

                __syncthreads();

                // Step 3: WMMA computation
                // Each warp: 16 M × 32 N output
                // Using 16×16×16 WMMA, need 2 N-tiles per warp
                // K=64 = 4 iterations of K=16

                int warp_id = tid / 32;
                int lane = tid % 32;
                int warp_n_base = warp_id * WARP_N;  // 0, 32, 64, or 96

                using A_frag =
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, WmmaT, rocwmma::row_major>;
                using B_frag =
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, WmmaT, rocwmma::col_major>;

#pragma unroll
                for (int k_slice = 0; k_slice < 4; k_slice++)
                {
                        // Load A: 16×16 tile of activations
                        A_frag a_frag;
                        rocwmma::load_matrix_sync(
                            a_frag,
                            //&x_sh[k_slice * 16],  // column offset for this K-slice
                            reinterpret_cast<const WmmaT*>(&x_sh[k_slice * 16]),
                            BLOCK_K);  // leading dimension

// Load B: two 16×16 tiles of weights (N dimension)
#pragma unroll
                        for (int n_tile = 0; n_tile < 2; n_tile++)
                        {
                                B_frag b_frag;
                                int n_offset = warp_n_base + n_tile * 16;
                                rocwmma::load_matrix_sync(
                                    b_frag,
                                    reinterpret_cast<const WmmaT*>(
                                        &w_sh[n_offset * BLOCK_K + k_slice * 16]),
                                    BLOCK_K);

                                rocwmma::mma_sync(acc[n_tile], a_frag, b_frag, acc[n_tile]);
                        }
                }

                __syncthreads();
        }

        // Step 4: Store each warp's accumulators to shared memory
        __shared__ float smem_out[BLOCK_M * BLOCK_N];  // 16×128 = 2048 floats

        int warp_id = tid / 32;
        int lane = tid % 32;

// Each warp stores two 16×16 tiles
#pragma unroll
        for (int n_tile = 0; n_tile < 2; n_tile++)
        {
                rocwmma::store_matrix_sync(
                    &smem_out[warp_id * WARP_N + n_tile * 16],  // destination column
                    acc[n_tile],
                    BLOCK_N,  // leading dimension of smem_out
                    rocwmma::mem_row_major);
        }
        __syncthreads();

        // Step 5: Cooperative write to global memory
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += BLOCK_N)
        {
                int mm = i / BLOCK_N;
                int nn = i % BLOCK_N;
                int m_global = cta_m + mm;
                int n_global = cta_n + nn;
                if (m_global < M && n_global < N)
                {
                        out[m_global * N + n_global] = from_fp32<T>(smem_out[i]);
                }
        }
}

template <typename T, typename WmmaT>
void launch_mma(const void* x, const void* qweight, const void* wscales, const void* wzeros,
                void* out, int M, int N, int K, int G, hipStream_t stream)
{
        constexpr int BLOCK_M = 16;
        constexpr int BLOCK_N = 128;
        constexpr int CTA_THREADS = 128;
        dim3 block(CTA_THREADS);
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        awq_w4a16_mma_kernel<T, WmmaT><<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(x), reinterpret_cast<const int8_t*>(qweight),
            reinterpret_cast<const T*>(wscales), reinterpret_cast<const T*>(wzeros),
            reinterpret_cast<T*>(out), M, N, K, G);
}

// CPU reference
template <typename T>
void awq_cpu_reference(const T* x, const int8_t* qweight, const T* wscales, const T* wzeros, T* out,
                       int M, int N, int K, int G)
{
        int K_half = K / 2;
        int n_groups = K / G;

        for (int m = 0; m < M; m++)
        {
                for (int n = 0; n < N; n++)
                {
                        float acc = 0.0f;
                        for (int g = 0; g < n_groups; g++)
                        {
                                float scale = wscales[g * N + n];
                                float zero = wzeros[g * N + n];
                                int k_base = g * G;
                                int kh_base = k_base / 2;

                                for (int kk = 0; kk < G; kk += 2)
                                {
                                        uint8_t byte = qweight[n * K_half + kh_base + kk / 2];
                                        int lo = byte & 0xF;
                                        int hi = (byte >> 4) & 0xF;
                                        float w_lo = (float(lo) - 8.0f) * scale + zero;
                                        float w_hi = (float(hi) - 8.0f) * scale + zero;
                                        acc += (float)x[m * K + k_base + kk] * w_lo;
                                        acc += (float)x[m * K + k_base + kk + 1] * w_hi;
                                }
                        }
                        out[m * N + n] = (T)acc;
                }
        }
}

int main()
{
        printf("=== AWQ Kernel Test ===\n");
        printf("M=%d N=%d K=%d G=%d\n\n", M, N, K, G);

        // Create known test data: all ones
        std::vector<__half> x(M * K, __float2half(1.0f));
        std::vector<int8_t> qweight(N * K / 2, 0x88);  // 0x88 = two int4 values of 8
        std::vector<__half> wscales(N * K / G, __float2half(0.5f));
        std::vector<__half> wzeros(N * K / G, __float2half(0.0f));
        std::vector<__half> out_gpu(M * N, __float2half(0.0f));
        std::vector<__half> out_cpu(M * N, __float2half(0.0f));

        // Expected: w = (8-8)*0.5 + 0 = 0, so output should be all zeros
        printf("Test 1: Zero weights (qweight=0x88, scale=0.5, zero=0)\n");
        printf("  Expected output: all zeros\n");

        // Allocate GPU memory
        __half *d_x, *d_wscales, *d_wzeros, *d_out;
        int8_t* d_qweight;
        hipMalloc(&d_x, M * K * sizeof(__half));
        hipMalloc(&d_qweight, N * K / 2 * sizeof(int8_t));
        hipMalloc(&d_wscales, N * K / G * sizeof(__half));
        hipMalloc(&d_wzeros, N * K / G * sizeof(__half));
        hipMalloc(&d_out, M * N * sizeof(__half));

        hipMemcpy(d_x, x.data(), M * K * sizeof(__half), hipMemcpyHostToDevice);
        hipMemcpy(d_qweight, qweight.data(), N * K / 2 * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_wscales, wscales.data(), N * K / G * sizeof(__half), hipMemcpyHostToDevice);
        hipMemcpy(d_wzeros, wzeros.data(), N * K / G * sizeof(__half), hipMemcpyHostToDevice);

        // Launch kernel
        dim3 block(128);
        dim3 grid(1, 1);  // Just one 16x128 tile
        awq_w4a16_mma_kernel<__half, rocwmma::float16_t>
            <<<grid, block, 0, 0>>>(d_x, d_qweight, d_wscales, d_wzeros, d_out, M, N, K, G);
        hipDeviceSynchronize();

        hipMemcpy(out_gpu.data(), d_out, M * N * sizeof(__half), hipMemcpyHostToDevice);

        // Check results
        float max_error = 0.0f;
        int error_count = 0;
        for (int i = 0; i < M * N; i++)
        {
                float gpu_val = __half2float(out_gpu[i]);
                if (fabsf(gpu_val) > 0.01f)
                {
                        if (error_count < 10)
                        {
                                printf("  ERROR at [%d,%d]: got %.4f, expected 0.0\n", i / N, i % N,
                                       gpu_val);
                        }
                        error_count++;
                        max_error = fmaxf(max_error, fabsf(gpu_val));
                }
        }
        if (error_count == 0)
        {
                printf("  PASS: all zeros\n");
        }
        else
        {
                printf("  FAIL: %d errors, max error = %.4f\n", error_count, max_error);
        }

        // Test 2: Simple known values
        printf("\nTest 2: Simple values (x=1, qweight=0x09 => lo=9,hi=0 => w=0.5,-4.0)\n");
        // qweight = 0x09: lo nibble = 9, hi nibble = 0
        // w_lo = (9-8)*0.5 + 0 = 0.5, w_hi = (0-8)*0.5 + 0 = -4.0
        // With x=1: sum of 0.5 + (-4.0) per pair = -3.5 per pair
        // 32 pairs per group, 4 groups = 128 pairs
        // Total per output = -3.5 * 128 = -448.0
        float expected = -448.0f;
        printf("  Expected: %.1f\n", expected);

        std::fill(qweight.begin(), qweight.end(), 0x09);
        hipMemcpy(d_qweight, qweight.data(), N * K / 2 * sizeof(int8_t), hipMemcpyHostToDevice);

        awq_w4a16_mma_kernel<__half, rocwmma::float16_t>
            <<<grid, block, 0, 0>>>(d_x, d_qweight, d_wscales, d_wzeros, d_out, M, N, K, G);
        hipDeviceSynchronize();

        hipMemcpy(out_gpu.data(), d_out, M * N * sizeof(__half), hipMemcpyHostToDevice);

        // Also run CPU reference
        awq_cpu_reference(x.data(), qweight.data(), wscales.data(), wzeros.data(), out_cpu.data(),
                          M, N, K, G);

        error_count = 0;
        max_error = 0.0f;
        for (int i = 0; i < M * N; i++)
        {
                float gpu_val = __half2float(out_gpu[i]);
                float cpu_val = __half2float(out_cpu[i]);
                float error = fabsf(gpu_val - expected);
                if (error > 1.0f)
                {  // Allow some fp16 precision loss
                        if (error_count < 10)
                        {
                                printf("  ERROR at [%d,%d]: got %.2f, expected %.2f (cpu: %.2f)\n",
                                       i / N, i % N, gpu_val, expected, cpu_val);
                        }
                        error_count++;
                        max_error = fmaxf(max_error, error);
                }
        }
        if (error_count == 0)
        {
                printf("  PASS: all values ~%.1f\n", expected);
        }
        else
        {
                printf("  FAIL: %d errors, max error = %.2f\n", error_count, max_error);
        }

        // Cleanup
        hipFree(d_x);
        hipFree(d_qweight);
        hipFree(d_wscales);
        hipFree(d_wzeros);
        hipFree(d_out);

        return error_count > 0 ? 1 : 0;
}
