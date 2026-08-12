/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef COMFY_FLOAT_UTILS_CUH_
#define COMFY_FLOAT_UTILS_CUH_

#include <hip/hip_fp4.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace comfy
{

// FP8 type traits for max values
template <typename T>
struct FP8LimitsTrait;

template <>
struct FP8LimitsTrait<__hip_fp8_e4m3_fnuz>
{
        static constexpr float max = 448.0f;
        static constexpr float max_inverse = 1.0 / max;
};

template <>
struct FP8LimitsTrait<__hip_fp8_e5m2_fnuz>
{
        static constexpr float max = 57344.0f;
        static constexpr float max_inverse = 1.0 / max;
};

//  FP4 type traits
template <typename T>
struct FP4LimitsTrait;

template <>
struct FP4LimitsTrait<__hip_fp4x2_storage_t>
{
        static constexpr float max = 6.0f;
        static constexpr float max_inverse = 1.0 / max;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
// Vectorized half-precision loads
template <typename IType>
__forceinline__ __device__ const IType* load_f16x2(const IType* val)
{
        float vals = *reinterpret_cast<const float*>(val);
        return reinterpret_cast<const IType*>(&vals);
}
template <typename IType>
__forceinline__ __device__ const IType* load_f16x4(const IType* val)
{
        float2 vals = *reinterpret_cast<const float2*>(val);
        return reinterpret_cast<const IType*>(&vals);
}
template <typename IType>
__forceinline__ __device__ const IType* load_f16x8(const IType* val)
{
        float4 vals = *reinterpret_cast<const float4*>(val);
        return reinterpret_cast<const IType*>(&vals);
}

template <typename IType>
__forceinline__ __device__ const IType* load_f8x4(const IType* val)
{
        float vals = *reinterpret_cast<const float*>(val);
        return reinterpret_cast<const IType*>(&vals);
}
template <typename IType>
__forceinline__ __device__ const IType* load_f8x8(const IType* val)
{
        float4 vals = *reinterpret_cast<const float4*>(val);
        return reinterpret_cast<const IType*>(&vals);
}

// Store 2 FP4 values (1 __nv_fp4x2)
// hi_first=true: val0 in high nibble, val1 in low nibble (default, matches
// cuBLAS convention) hi_first=false: val0 in low nibble, val1 in high nibble
template <typename OType, bool hi_first = true>
__forceinline__ __device__ void store_fp4x2(OType* output, size_t idx, float val0, float val1)
{
        float2 args = hi_first ? float2{val1, val0} : float2{val0, val1};
        *reinterpret_cast<__hip_fp4x2_storage_t*>(&output[idx]) =
            __hip_cvt_float2_to_fp4x2(args, __HIP_E2M1, hipRoundNearest);
}

// Store 4 FP4 values (2 __nv_fp4x2) using single store
template <typename OType, bool hi_first = true>
__forceinline__ __device__ void store_fp4x4(OType* output, size_t idx, float val0, float val1,
                                            float val2, float val3)
{
        union
        {
                uint16_t u16;
                __hip_fp4x2_storage_t fp4x2[2];
        } packed;

        float2 args0 = hi_first ? float2{val1, val0} : float2{val0, val1};
        float2 args1 = hi_first ? float2{val3, val2} : float2{val2, val3};
        packed.fp4x2[0] = __hip_cvt_float2_to_fp4x2(args0, __HIP_E2M1, hipRoundNearest);
        packed.fp4x2[1] = __hip_cvt_float2_to_fp4x2(args1, __HIP_E2M1, hipRoundNearest);

        *reinterpret_cast<uint16_t*>(&output[2 * idx]) = packed.u16;
}

#pragma clang diagnostic pop

// cuBLAS swizzled scale factor layout offset calculation
__device__ __forceinline__ size_t scale_factor_swizzled_offset(size_t row_idx, size_t col_idx,
                                                               uint32_t col_length)
{
        constexpr uint32_t kTotalRowsPerBaseBlock = 128;
        constexpr uint32_t kRowsPerBaseBlockCol = 32;
        constexpr uint32_t kColsPerBaseBlockCol = 4;

        const size_t rb = row_idx / kTotalRowsPerBaseBlock;
        const size_t rem = row_idx % kTotalRowsPerBaseBlock;
        const size_t d4 = rem / kRowsPerBaseBlockCol;
        const size_t d3 = rem % kRowsPerBaseBlockCol;
        const size_t cbg = col_idx / kColsPerBaseBlockCol;
        const size_t d5 = col_idx % kColsPerBaseBlockCol;

        const size_t cbg_cnt = (col_length + kColsPerBaseBlockCol - 1) / kColsPerBaseBlockCol;
        return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol +
               d5;
}

__forceinline__ __device__ uint8_t encode_std_e4m3(float val)
{
        constexpr float kMax = 448.0f;
        val = fminf(fmaxf(val, -kMax), kMax);

        uint8_t sign = (val < 0.0f) ? 0x80 : 0x00;
        float abs_val = fabsf(val);

        if (abs_val == 0.0f) return 0x00;

        int exp;
        float sig = frexpf(abs_val, &exp);
        int exp_bits = exp + 6;  // bias=7: frexp gives sig*2^exp, we need 2^(exp_bits-7)*(1+mant/8)

        if (exp_bits <= 0)
        {
                // Subnormal: 2^-6 * mant/8 = abs_val
                int mant = (int)roundf(abs_val * 64.0f * 8.0f);  // / (2^-6/8) = * 512
                if (mant > 7) mant = 7;
                if (mant == 0) return 0x00;
                return sign | mant;
        }

        if (exp_bits >= 15)
        {
                return sign | (15 << 3) | 6;  // max: 0x7E or 0xFE
        }

        // Normal: abs_val = (1 + mant/8) * 2^(exp_bits - 7)
        // sig * 2^exp = (1 + mant/8) * 2^(exp_bits - 7)
        // sig * 2 = 1 + mant/8  →  mant = (sig * 2 - 1) * 8
        int mant = (int)roundf((sig * 2.0f - 1.0f) * 8.0f + 1e-7f);
        if (mant == 8)
        {
                mant = 0;
                exp_bits++;
                if (exp_bits >= 15) return sign | (15 << 3) | 6;
        }

        return sign | (exp_bits << 3) | mant;
}

}  // namespace comfy

#endif  // COMFY_FLOAT_UTILS_CUH_
