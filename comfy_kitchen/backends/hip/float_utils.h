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

__device__ __forceinline__ float warp_reduce_fmax(float v)
{
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xfffffffful, v, off));
        return v;
}

__device__ __forceinline__ int8_t float_to_int8_rn(float value)
{
        int32_t quantized;
        // asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(quantized) : "f"(value));
        return static_cast<int8_t>(quantized);
}

__device__ __forceinline__ int8_t quant_int8_rcp(float v, float inv_scale)
{
        return float_to_int8_rn(v * inv_scale);
}

__device__ __forceinline__ int8_t quant_int8(float v, float scale)
{
        return quant_int8_rcp(v, 1.f / scale);
}

// ptr must be 4-byte aligned. Callers must preserve that alignment at every
// row offset, for example by requiring the row stride to be a multiple of 4.
__device__ __forceinline__ void store4_i8(int8_t* ptr, int8_t a, int8_t b, int8_t c, int8_t d)
{
        *reinterpret_cast<int32_t*>(ptr) = (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
                                           ((uint32_t)(uint8_t)c << 16) |
                                           ((uint32_t)(uint8_t)d << 24);
}

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

// From 0xDELUXA hip port fp8_utils.h
namespace hip_backend
{

// dtype codes match comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE
constexpr int kFp8E4M3Code = 5;

__forceinline__ __device__ float fp8_clamp(float v, float lo, float hi)
{
        return fminf(hi, fmaxf(lo, v));
}

// The kernels are compiled with -ffast-math, which implies -ffinite-math-only:
// isnan() folds to false there, and fp8_clamp() drops a NaN operand outright.
// Test the bits instead, so a NaN input still encodes as a NaN.
__forceinline__ __device__ bool fp8_is_nan(float v)
{
        return (__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u;
}

// Round to nearest, ties to even. OCP FP8 and torch's cast both round this way, so
// the packed code has to agree at the exact midpoints that floorf(x + 0.5f) would
// otherwise push up (e.g. e4m3 1.0625 must land on 1.0, not 1.125). x is finite and
// non-negative here, so the -ffast-math finiteness assumption is not in play.
__forceinline__ __device__ int round_to_even(float x)
{
        const float floor_x = floorf(x);
        const float frac = x - floor_x;
        int n = static_cast<int>(floor_x);
        if (frac > 0.5f)
        {
                n += 1;
        }
        else if (frac == 0.5f)
        {
                n += (n & 1);  // tie to the even neighbour
        }
        return n;
}

template <typename T>
__forceinline__ __device__ float to_float(T value)
{
        return static_cast<float>(value);
}

template <>
__forceinline__ __device__ float to_float<__half>(__half value)
{
        return __half2float(value);
}

// e4m3fn: 1-4-3, bias 7, no infinities. e5m2: 1-5-2, bias 15.
__forceinline__ __device__ uint8_t pack_fp8(float value, int dtype_code)
{
        const bool e4m3 = dtype_code == kFp8E4M3Code;
        const int mantissa_bits = e4m3 ? 3 : 2;
        const int exponent_bias = e4m3 ? 7 : 15;
        const int mantissa_levels = 1 << mantissa_bits;
        const int max_exponent_field = e4m3 ? 15 : 30;
        const int max_mantissa_field = e4m3 ? 6 : (mantissa_levels - 1);
        const float fp8_max = e4m3 ? 448.0f : 57344.0f;

        const uint8_t sign_bit = (__float_as_uint(value) >> 31) ? 0x80 : 0x00;
        float abs_value = fabsf(value);
        // Exponent all ones, mantissa all ones: the NaN torch encodes for both formats.
        if (fp8_is_nan(value))
        {
                return static_cast<uint8_t>(sign_bit | 0x7f);
        }

        // Non-finite magnitudes saturate here, matching eager's clamp-then-cast: a
        // raw torch cast would keep e5m2's 0x7c, the op does not.
        abs_value = fp8_clamp(abs_value, 0.0f, fp8_max);
        if (abs_value == 0.0f)
        {
                return sign_bit;
        }

        const float min_normal = exp2f(1 - exponent_bias);
        int exponent_field = 0;
        int mantissa_field = 0;
        if (abs_value < min_normal)
        {
                const float subnormal_scale = exp2f(1 - exponent_bias - mantissa_bits);
                mantissa_field = round_to_even(abs_value / subnormal_scale);
                if (mantissa_field >= mantissa_levels)
                {
                        exponent_field = 1;
                        mantissa_field = 0;
                }
        }
        else
        {
                exponent_field = static_cast<int>(floorf(log2f(abs_value))) + exponent_bias;
                const float exponent_scale = exp2f(exponent_field - exponent_bias);
                mantissa_field =
                    round_to_even((abs_value / exponent_scale - 1.0f) * mantissa_levels);
                if (mantissa_field >= mantissa_levels)
                {
                        mantissa_field = 0;
                        exponent_field += 1;
                }
        }

        if (exponent_field > max_exponent_field ||
            (exponent_field == max_exponent_field && mantissa_field > max_mantissa_field))
        {
                exponent_field = max_exponent_field;
                mantissa_field = max_mantissa_field;
        }

        return static_cast<uint8_t>(sign_bit | (exponent_field << mantissa_bits) | mantissa_field);
}

__forceinline__ __device__ float unpack_fp8(uint8_t value, int dtype_code = kFp8E4M3Code)
{
        const bool e4m3 = dtype_code == kFp8E4M3Code;
        const int mantissa_bits = e4m3 ? 3 : 2;
        const int exponent_bits = e4m3 ? 4 : 5;
        const int exponent_bias = e4m3 ? 7 : 15;
        const int mantissa_mask = (1 << mantissa_bits) - 1;
        const int exponent_mask = (1 << exponent_bits) - 1;

        const float sign = (value & 0x80) ? -1.0f : 1.0f;
        const int exponent = (value >> mantissa_bits) & exponent_mask;
        const int mantissa = value & mantissa_mask;

        // e4m3fn has no infinities: an all-ones exponent with an all-ones mantissa is
        // its only NaN. e5m2 follows IEEE: all-ones exponent is inf, or NaN when the
        // mantissa is non-zero. pack_fp8 emits only the NaN code, since it saturates
        // infinities, but a tensor from any other producer can carry either. The
        // result is assembled from bits, not float literals: -ffast-math assumes no
        // NaN and would fold a branch that only ever returns one.
        if (exponent == exponent_mask && (!e4m3 || mantissa == mantissa_mask))
        {
                const uint32_t sign_bits = (value & 0x80) ? 0x80000000u : 0u;
                const uint32_t payload = (!e4m3 && mantissa == 0) ? 0x7f800000u   // e5m2 inf
                                                                  : 0x7fc00000u;  // quiet NaN
                return __uint_as_float(sign_bits | payload);
        }

        if (exponent == 0)
        {
                if (mantissa == 0)
                {
                        return sign * 0.0f;
                }
                return sign * exp2f(1 - exponent_bias) *
                       (static_cast<float>(mantissa) / (1 << mantissa_bits));
        }
        return sign * exp2f(exponent - exponent_bias) *
               (1.0f + static_cast<float>(mantissa) / (1 << mantissa_bits));
}

// Fast e4m3fn decode for the small-M GEMV and, on gfx11, for widening the WMMA
// operands to bf16. Both feed arbitrary user tensors, so the specials have to
// survive: gfx12 hardware propagates a NaN operand through the fp8 WMMA, and the
// widened gfx11 path has to agree with it. Results are assembled from bits, not
// float literals, because -ffast-math would fold a branch that only ever returns
// a NaN.
__forceinline__ __device__ float fp8_to_float(uint8_t b)
{
        const uint32_t sign_bits = (b & 0x80) ? 0x80000000u : 0u;
        const int exp = (b >> 3) & 0xF;
        const int man = b & 0x7;

        // e4m3fn has no infinities: 0x7f/0xff is its only NaN.
        if (exp == 0xF && man == 0x7)
        {
                return __uint_as_float(sign_bits | 0x7fc00000u);
        }

        const float sign = (b & 0x80) ? -1.0f : 1.0f;
        if (exp == 0) return sign * man * 0.001953125f;  // subnormal step: 2^-9

        const uint32_t bits =
            (static_cast<uint32_t>(exp - 7 + 127) << 23) | (static_cast<uint32_t>(man) << 20);
        return sign * __uint_as_float(bits);
}

// e5m2 counterpart. Unlike e4m3fn it follows IEEE: an all-ones exponent is an
// infinity when the mantissa is zero and a NaN otherwise.
__forceinline__ __device__ float bf8_to_float(uint8_t b)
{
        const uint32_t sign_bits = (b & 0x80) ? 0x80000000u : 0u;
        const int exp = (b >> 2) & 0x1F;
        const int man = b & 0x3;

        if (exp == 0x1F)
        {
                return __uint_as_float(sign_bits | (man == 0 ? 0x7f800000u : 0x7fc00000u));
        }

        const float sign = (b & 0x80) ? -1.0f : 1.0f;
        if (exp == 0) return sign * man * 0.0000152587890625f;  // subnormal step: 2^-16

        const uint32_t bits =
            (static_cast<uint32_t>(exp - 15 + 127) << 23) | (static_cast<uint32_t>(man) << 21);
        return sign * __uint_as_float(bits);
}

}  // namespace hip_backend
}  // namespace comfy

#endif  // COMFY_FLOAT_UTILS_CUH_
