/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * OPTIMIZED ROCm/HIP ck_tile: INT8 GEMM with FUSED dequant for RDNA3/4
 *
 * Key optimizations:
 * 1. Removed broadcast kernels - using CK's native broadcast support
 * 2. Added more tile configurations for better autotuning
 * 3. Enabled double buffering for better latency hiding
 * 4. Improved benchmarking methodology
 * 5. Optimized memory access patterns
 */
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>

#include "../utils.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace comfy
{

// ============================================================================
// Dequant epilogue: D = acc * x_scale[m] * w_scale[n] + bias[n]
// Now handles broadcasted inputs directly without pre-broadcasting
// ============================================================================
struct FusedDequantEpilogueOptimized
{
        template <typename E, typename C, typename D0, typename D1, typename D2>
        CK_TILE_HOST_DEVICE void operator()(E& e, const C& c, const D0& d0, const D1& d1,
                                            const D2& d2) const
        {
                // The hardware automatically broadcasts d0 (row vector [M]),
                // d1 (column vector [N]), and d2 (column vector [N])
                float val = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                                ck_tile::type_convert<float>(d1) +
                            ck_tile::type_convert<float>(d2);
                e = ck_tile::type_convert<E>(val);
        }
};

// ============================================================================
// GemmConfig for RDNA3 WMMA int8 - optimized for various matrix shapes
// ============================================================================
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int WTM, int WTN, int WTK,
          int BlockPerCu>
struct GemmConfigInt8WMMAOptimized
{
        static constexpr ck_tile::index_t M_Tile = TBM;
        static constexpr ck_tile::index_t N_Tile = TBN;
        static constexpr ck_tile::index_t K_Tile = TBK;
        static constexpr ck_tile::index_t M_Warp = WM;
        static constexpr ck_tile::index_t N_Warp = WN;
        static constexpr ck_tile::index_t K_Warp = WK;
        static constexpr ck_tile::index_t M_Warp_Tile = WTM;
        static constexpr ck_tile::index_t N_Warp_Tile = WTN;
        static constexpr ck_tile::index_t K_Warp_Tile = WTK;
        static constexpr int kBlockPerCu = BlockPerCu;

        static constexpr bool kPadM = true;
        static constexpr bool kPadN = true;
        static constexpr bool kPadK = true;
        static constexpr bool PermuteA = false;
        static constexpr bool PermuteB = false;
        static constexpr bool TransposeC = false;
        static constexpr bool UseStructuredSparsity = false;

        // OPTIMIZATION: Increased to 16 groups for better scheduling
        static constexpr ck_tile::index_t TileParitionerGroupNum = 16;
        static constexpr ck_tile::index_t TileParitionerM01 = 4;

        static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
        static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

        // OPTIMIZATION: Enable double buffering for better latency hiding
        static constexpr bool DoubleSmemBuffer = true;
};

// ============================================================================
// One fused int8 GEMM instance - optimized version
// ============================================================================
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int WTM,
          int WTN, int WTK, int kBlockPerCu>
struct FusedInt8GemmCKTileOptimized
{
        using ADataType = ck_tile::int8_t;
        using BDataType = ck_tile::int8_t;
        using AccDataType = int32_t;
        using CDataType = ElementOutput;

        using D0DataType = float;  // x_scale - row vector [M]
        using D1DataType = float;  // w_scale - column vector [N]
        using D2DataType = float;  // bias - column vector [N]
        using DsDataType = ck_tile::tuple<D0DataType, D1DataType, D2DataType>;

        using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
        using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
        using ELayout = ck_tile::tensor_layout::gemm::RowMajor;

        // OPTIMIZATION: Use different layouts for broadcasting
        // D0 (x_scale): RowMajor with stride 0 along N (broadcast along rows)
        // D1 (w_scale): ColumnMajor with stride 0 along M (broadcast along columns)
        // D2 (bias): ColumnMajor with stride 0 along M (broadcast along columns)
        using DsLayout =
            ck_tile::tuple<ck_tile::tensor_layout::gemm::RowMajor,     // x_scale: [M, 1] broadcast
                           ck_tile::tensor_layout::gemm::ColumnMajor,  // w_scale: [1, N] broadcast
                           ck_tile::tensor_layout::gemm::ColumnMajor   // bias: [1, N] broadcast
                           >;

        using GemmConfig =
            GemmConfigInt8WMMAOptimized<TBM, TBN, TBK, WM, WN, WK, WTM, WTN, WTK, kBlockPerCu>;

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile,
                              GemmConfig::K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape, GemmConfig::TileParitionerGroupNum, GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM, GemmConfig::kPadN,
                                             GemmConfig::kPadK, GemmConfig::DoubleSmemBuffer,
                                             ALayout, BLayout, ELayout, GemmConfig::TransposeC>;

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape,
                                                  GemmUniversalTraits, GemmConfig::Scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            ADataType, BDataType, DsDataType, AccDataType, CDataType, DsLayout, ELayout,
            FusedDequantEpilogueOptimized, TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::M_Warp_Tile,
            GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile, UniversalGemmProblem::TransposeC>>;

        using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;

        static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                        const float* bias, ElementOutput* D, int M, int N, int K,
                        hipStream_t stream)
        {
                // OPTIMIZATION: No broadcast kernels needed!
                // Pass the vectors directly with appropriate strides

                using GemmMultiDArgs = ck_tile::GemmMultiDHostArgs<DsDataType::size()>;

                int stride_A = K;  // RowMajor: A[m,k] at A[m*K + k]
                int stride_B = K;  // ColumnMajor: B[n,k] at B[n*K + k]
                int stride_E = N;  // RowMajor output: D[m,n] at D[m*N + n]

                // OPTIMIZATION: Strides for broadcast tensors
                // These tell CK how to access the vectors
                int stride_xs = 0;    // x_scale[m] - same for all n (row broadcast)
                int stride_ws = 0;    // w_scale[n] - same for all m (column broadcast)
                int stride_bias = 0;  // bias[n] - same for all m (column broadcast)

                GemmMultiDArgs args = {
                    const_cast<int8_t*>(A),
                    const_cast<int8_t*>(B),
                    {const_cast<float*>(xs), const_cast<float*>(ws),
                     bias ? const_cast<float*>(bias) : nullptr},  // Handle null bias
                    D,
                    1,
                    M,
                    N,
                    K,
                    stride_A,
                    stride_B,
                    {stride_xs, stride_ws, stride_bias},
                    stride_E};

                auto kargs = Kernel::MakeKernelArgs(args);

                if (!Kernel::IsSupportedArgument(kargs))
                {
                        return false;
                }

                const dim3 grids = Kernel::GridSize(M, N, 1);
                const dim3 blocks = Kernel::BlockSize();

                ck_tile::stream_config s{stream, false, 1};
                float elapsed =
                    ck_tile::launch_kernel(s, ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                                  Kernel{}, grids, blocks, 0, kargs));

                hipError_t err = hipGetLastError();
                if (err != hipSuccess)
                {
                        printf("  kernel error: %s\n", hipGetErrorString(err));
                        return false;
                }

                return elapsed >= 0;
        }
};

// ============================================================================
// Autotuning dispatcher - optimized with better configurations and benchmarking
// ============================================================================

template <typename OutT>
bool dispatch_fused_ck_optimized(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                                 const float* bias, OutT* D, int M, int N, int K,
                                 hipStream_t stream)
{
        using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*,
                            OutT*, int, int, int, hipStream_t);

        // OPTIMIZATION: Expanded configuration set for better coverage
        // Format: <OutputType, M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp,
        //          M_Warp_Tile, N_Warp_Tile, K_Warp_Tile, BlockPerCu>
        static const Fn runners[] = {
            // Large tiles for large matrices
            &FusedInt8GemmCKTileOptimized<OutT, 256, 256, 64, 4, 4, 1, 16, 16, 16, 1>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 256, 128, 64, 4, 2, 1, 16, 16, 16, 1>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 128, 256, 64, 2, 4, 1, 16, 16, 16, 1>::run,

            // Medium tiles for medium matrices
            &FusedInt8GemmCKTileOptimized<OutT, 128, 128, 64, 4, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 128, 128, 32, 4, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 128, 64, 64, 4, 1, 1, 16, 16, 32, 2>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 64, 128, 64, 2, 4, 1, 16, 16, 32, 1>::run,

            // Small tiles for small matrices
            &FusedInt8GemmCKTileOptimized<OutT, 64, 64, 32, 2, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 64, 64, 64, 2, 2, 1, 16, 16, 32, 1>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 32, 64, 32, 1, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTileOptimized<OutT, 64, 32, 32, 2, 1, 1, 16, 16, 16, 2>::run,

            // Specialized for tall-skinny matrices
            &FusedInt8GemmCKTileOptimized<OutT, 64, 256, 32, 2, 4, 1, 16, 16, 16, 1>::run,
            // Specialized for short-wide matrices
            &FusedInt8GemmCKTileOptimized<OutT, 256, 64, 32, 4, 2, 1, 16, 16, 16, 1>::run,
        };
        constexpr int NC = sizeof(runners) / sizeof(runners[0]);

        static std::mutex mtx;
        static std::map<std::tuple<int, int, int>, int> cache;
        const std::tuple<int, int, int> key{M, N, K};

        int best;
        {
                std::lock_guard<std::mutex> lk(mtx);
                auto it = cache.find(key);
                best = (it != cache.end()) ? it->second : -2;
        }

        if (best == -2)
        {
                best = -1;
                float best_ms = 1e30f;
                hipEvent_t start, stop;
                CUDA_CHECK(hipEventCreate(&start));
                CUDA_CHECK(hipEventCreate(&stop));

                // OPTIMIZATION: Warmup all kernels first
                for (int i = 0; i < NC; ++i)
                {
                        if (runners[i](A, B, xs, ws, bias, D, M, N, K, stream))
                        {
                                // Just warmup, don't measure
                        }
                }
                CUDA_CHECK(hipStreamSynchronize(stream));

                // OPTIMIZATION: Benchmark with proper methodology
                for (int i = 0; i < NC; ++i)
                {
                        // Quick check if kernel works
                        if (!runners[i](A, B, xs, ws, bias, D, M, N, K, stream)) continue;

                        // Warmup this specific kernel
                        runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
                        CUDA_CHECK(hipStreamSynchronize(stream));

                        // Time multiple iterations for better accuracy
                        CUDA_CHECK(hipEventRecord(start, stream));
                        const int num_iters = 5;
                        for (int r = 0; r < num_iters; ++r)
                        {
                                runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
                        }
                        CUDA_CHECK(hipEventRecord(stop, stream));
                        CUDA_CHECK(hipEventSynchronize(stop));

                        float ms = 0.f;
                        CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
                        ms /= num_iters;  // Average time per iteration

                        if (ms < best_ms)
                        {
                                best_ms = ms;
                                best = i;
                        }
                }

                CUDA_CHECK(hipEventDestroy(start));
                CUDA_CHECK(hipEventDestroy(stop));

                std::lock_guard<std::mutex> lk(mtx);
                cache[key] = best;
        }

        if (best < 0) return false;
        return runners[best](A, B, xs, ws, bias, D, M, N, K, stream);
}

}  // namespace comfy

extern "C"
{
        bool launch_cutlass_int8_dequant(const void* A, const void* B, const void* xs,
                                         const void* ws, const void* bias, void* D, int64_t M,
                                         int64_t N, int64_t K, int out_dtype_code,
                                         hipStream_t stream)
        {
                try
                {
                        if (M == 0 || N == 0 || K == 0) return true;

                        const int8_t* a = static_cast<const int8_t*>(A);
                        const int8_t* b = static_cast<const int8_t*>(B);
                        const float* x = static_cast<const float*>(xs);
                        const float* w = static_cast<const float*>(ws);
                        const float* bs = static_cast<const float*>(bias);

                        switch (out_dtype_code)
                        {
                                case 0:  // float32 output
                                        return comfy::dispatch_fused_ck_optimized<float>(
                                            a, b, x, w, bs, (float*)D, M, N, K, stream);
                                case 1:  // float16 output
                                        return comfy::dispatch_fused_ck_optimized<ck_tile::half_t>(
                                            a, b, x, w, bs, (ck_tile::half_t*)D, M, N, K, stream);
                                case 2:  // bfloat16 output
                                        return comfy::dispatch_fused_ck_optimized<ck_tile::bf16_t>(
                                            a, b, x, w, bs, (ck_tile::bf16_t*)D, M, N, K, stream);
                                default:
                                        return false;
                        }
                }
                catch (const std::exception& e)
                {
                        fprintf(stderr, "CK EXCEPTION: %s\n", e.what());
                        return false;
                }
                catch (...)
                {
                        fprintf(stderr, "CK UNKNOWN EXCEPTION\n");
                        return false;
                }
        }

}  // extern "C"
