# File: tests/comprehensive_fp8_format_test.py
"""
Comprehensive test to diagnose AMD FNUZ vs standard FP8 format issues
"""

import torch
import sys
import numpy as np
from typing import Tuple, Optional

# Setup path to find the compiled module
sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

# ============================================================================
# FORMAT DIAGNOSTIC FUNCTIONS
# ============================================================================


def inspect_fp8_bits(x: torch.Tensor, label: str = "") -> None:
    """Show the raw bits of FP8 tensors to compare formats."""
    if x.dtype not in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ]:
        print(f"{label}: dtype={x.dtype} (not fp8)")
        return

    raw = x.view(torch.uint8)
    print(f"\n{label} (dtype={x.dtype}):")
    print(f"  Raw bytes: {' '.join(f'0x{b:02x}' for b in raw.flatten()[:16].tolist())}")

    # Decode bits
    for i in range(min(4, raw.numel())):
        b = raw.flatten()[i].item()
        sign = (b >> 7) & 1
        if "e5m2" in str(x.dtype):
            exp = (b >> 2) & 0x1F
            mant = b & 0x03
            bias = 15
        else:  # e4m3
            exp = (b >> 3) & 0x0F
            mant = b & 0x07
            bias = 7
        is_nan = (exp == ((1 << (5 if "e5m2" in str(x.dtype) else 4)) - 1)) and (mant != 0)
        print(f"  [{i}] byte=0x{b:02x} sign={sign} exp={exp} mant={mant} bias={bias} NaN={is_nan}")


def compare_formats(values: list) -> None:
    """Compare the same values in different FP8 formats."""
    print("\n" + "=" * 70)
    print("FP8 FORMAT COMPARISON")
    print("=" * 70)

    t_f16 = torch.tensor(values, dtype=torch.float16)
    print(f"\nInput (FP16): {t_f16.tolist()}")

    # Convert to different FP8 formats
    try:
        t_e4m3fn = t_f16.to(torch.float8_e4m3fn)
        inspect_fp8_bits(t_e4m3fn, "Standard E4M3")
    except Exception as e:
        print(f"Standard E4M3: ERROR - {e}")

    try:
        t_e4m3fnuz = t_f16.to(torch.float8_e4m3fnuz)
        inspect_fp8_bits(t_e4m3fnuz, "AMD E4M3 FNUZ")
    except Exception as e:
        print(f"AMD E4M3 FNUZ: ERROR - {e}")

    try:
        t_e5m2 = t_f16.to(torch.float8_e5m2)
        inspect_fp8_bits(t_e5m2, "Standard E5M2")
    except Exception as e:
        print(f"Standard E5M2: ERROR - {e}")

    try:
        t_e5m2fnuz = t_f16.to(torch.float8_e5m2fnuz)
        inspect_fp8_bits(t_e5m2fnuz, "AMD E5M2 FNUZ")
    except Exception as e:
        print(f"AMD E5M2 FNUZ: ERROR - {e}")

    # Check round-trip conversion
    print("\nRound-trip conversions (FP16 -> FP8 -> FP16):")
    for fmt_name, fmt in [("E4M3", torch.float8_e4m3fn), ("E4M3 FNUZ", torch.float8_e4m3fnuz)]:
        try:
            fp8 = t_f16.to(fmt)
            back = fp8.to(torch.float16)
            print(f"  {fmt_name}: {back.tolist()}")
        except Exception as e:
            print(f"  {fmt_name}: ERROR - {e}")


# ============================================================================
# KERNEL INTERFACE TESTS
# ============================================================================


def test_quantize_with_format(
    input_values: list,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    scale_val: float = 1.0,
    dtype_code_map: dict = None,
) -> Tuple[bool, str]:
    """
    Test quantization with specific dtype codes.
    Returns (success, message).
    """
    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Create tensors
    x = torch.tensor(input_values, device=device, dtype=input_dtype)
    scale = torch.tensor([scale_val], device=device, dtype=torch.float32)

    # Determine dtype codes
    if dtype_code_map is None:
        dtype_code_map = {
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
            torch.float8_e4m3fn: 5,
            torch.float8_e5m2: 6,
            torch.float8_e4m3fnuz: 7,  # Custom codes for AMD
            torch.float8_e5m2fnuz: 8,
        }

    input_code = dtype_code_map.get(input_dtype, -1)
    output_code = dtype_code_map.get(output_dtype, -1)

    if input_code == -1 or output_code == -1:
        return False, f"Unknown dtype code for {input_dtype}->{output_dtype}"

    # Output buffer (always uint8 for FP8)
    result_uint8 = torch.empty(x.shape, device=device, dtype=torch.uint8)

    try:
        # Wrap for DLPack
        x_ptr = x.data_ptr()
        scale_ptr = scale.data_ptr()
        result_ptr = result_uint8.data_ptr()

        _C.quantize_per_tensor_fp8(
            x, scale, result_uint8, input_code, output_code, x.numel(), stream_ptr
        )
        torch.cuda.synchronize()

        # Check results
        result_bytes = result_uint8.flatten().tolist()

        # Check for NaN in FP8 output
        nan_count = 0
        for b in result_bytes:
            if output_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
                exp = (b >> 3) & 0x0F
                mant = b & 0x07
                if exp == 0x0F and mant != 0:
                    nan_count += 1
            elif output_dtype in [torch.float8_e5m2, torch.float8_e5m2fnuz]:
                exp = (b >> 2) & 0x1F
                mant = b & 0x03
                if exp == 0x1F and mant != 0:
                    nan_count += 1

        if nan_count > 0:
            return (
                False,
                f"NaN count: {nan_count}/{len(input_values)}. Bytes: {[f'0x{b:02x}' for b in result_bytes]}",
            )

        # Compare with PyTorch's conversion
        expected_fp8 = x.to(output_dtype)
        expected_bytes = expected_fp8.view(torch.uint8).flatten().tolist()

        if result_bytes != expected_bytes:
            mismatches = [
                (i, f"0x{r:02x}", f"0x{e:02x}")
                for i, (r, e) in enumerate(zip(result_bytes, expected_bytes))
                if r != e
            ]
            return False, (
                f"Format mismatch: {mismatches[:5]}. "
                f"Got: {[f'0x{b:02x}' for b in result_bytes]}, "
                f"Expected: {[f'0x{b:02x}' for b in expected_bytes]}"
            )

        return True, "OK"

    except Exception as e:
        return False, f"Exception: {e}"


def test_dequantize_with_format(
    input_values: list,
    fp8_dtype: torch.dtype,
    output_dtype: torch.dtype,
    scale_val: float = 1.0,
    dtype_code_map: dict = None,
) -> Tuple[bool, str]:
    """
    Test dequantization with specific dtype codes.
    """
    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Create FP8 input using PyTorch's conversion
    x_f16 = torch.tensor(input_values, device=device, dtype=torch.float16)
    x_fp8 = x_f16.to(fp8_dtype)
    scale = torch.tensor([scale_val], device=device, dtype=torch.float32)

    # Determine dtype codes
    if dtype_code_map is None:
        dtype_code_map = {
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
            torch.float8_e4m3fn: 5,
            torch.float8_e5m2: 6,
            torch.float8_e4m3fnuz: 7,
            torch.float8_e5m2fnuz: 8,
        }

    fp8_code = dtype_code_map.get(fp8_dtype, -1)
    output_code = dtype_code_map.get(output_dtype, -1)

    if fp8_code == -1 or output_code == -1:
        return False, f"Unknown dtype code for {fp8_dtype}->{output_dtype}"

    result = torch.empty(x_fp8.shape, device=device, dtype=output_dtype)

    try:
        _C.dequantize_per_tensor_fp8(
            x_fp8.view(torch.uint8), scale, result, fp8_code, output_code, x_fp8.numel(), stream_ptr
        )
        torch.cuda.synchronize()

        # Check results
        result_vals = result.float().flatten().tolist()

        # Check for NaN in output
        nan_indices = [i for i, v in enumerate(result_vals) if np.isnan(v)]
        if nan_indices:
            return False, f"NaN in output at indices {nan_indices}: {result_vals}"

        # Compare with expected
        expected = x_fp8.to(output_dtype).float().flatten().tolist()

        max_diff = max(abs(a - b) for a, b in zip(result_vals, expected))
        if max_diff > 0.01:
            return False, (f"Max diff: {max_diff}. Got: {result_vals}, Expected: {expected}")

        return True, f"OK (max diff: {max_diff:.6f})"

    except Exception as e:
        return False, f"Exception: {e}"


# ============================================================================
# COMPREHENSIVE DIAGNOSTIC
# ============================================================================


def run_full_diagnostic():
    """Run complete diagnostic of the FP8 format and interface issues."""

    print("=" * 70)
    print("COMPREHENSIVE FP8 FORMAT & INTERFACE DIAGNOSTIC")
    print("=" * 70)

    # Part 1: Understand the formats
    print("\n" + "=" * 70)
    print("PART 1: FORMAT UNDERSTANDING")
    print("=" * 70)

    test_values = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 448.0, -448.0]
    compare_formats(test_values)

    # Part 2: Test kernel with both format code mappings
    print("\n" + "=" * 70)
    print("PART 2: KERNEL INTERFACE TESTS")
    print("=" * 70)

    # The two possible mappings
    standard_mapping = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float8_e4m3fn: 5,
        torch.float8_e5m2: 6,
    }

    amd_mapping = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float8_e4m3fnuz: 5,
        torch.float8_e5m2fnuz: 6,
    }

    extended_mapping = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float8_e4m3fn: 5,
        torch.float8_e5m2: 6,
        torch.float8_e4m3fnuz: 7,
        torch.float8_e5m2fnuz: 8,
    }

    test_inputs = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 448.0, -448.0]

    # Test with standard FP8
    print("\n--- Quantize FP16 -> Standard E4M3 ---")
    for mapping_name, mapping in [
        ("Standard codes (5=E4M3)", standard_mapping),
        ("AMD codes (5=E4M3 FNUZ)", amd_mapping),
        ("Extended codes (5=E4M3, 7=E4M3 FNUZ)", extended_mapping),
    ]:
        success, msg = test_quantize_with_format(
            test_inputs, torch.float16, torch.float8_e4m3fn, dtype_code_map=mapping
        )
        print(f"  {mapping_name}: {'PASS' if success else 'FAIL'} - {msg}")

    print("\n--- Quantize FP16 -> AMD E4M3 FNUZ ---")
    for mapping_name, mapping in [
        ("Standard codes (5=E4M3)", standard_mapping),
        ("AMD codes (5=E4M3 FNUZ)", amd_mapping),
        ("Extended codes (5=E4M3, 7=E4M3 FNUZ)", extended_mapping),
    ]:
        try:
            success, msg = test_quantize_with_format(
                test_inputs, torch.float16, torch.float8_e4m3fnuz, dtype_code_map=mapping
            )
            print(f"  {mapping_name}: {'PASS' if success else 'FAIL'} - {msg}")
        except Exception as e:
            print(f"  {mapping_name}: ERROR - {e}")

    # Test dequantize
    print("\n--- Dequantize Standard E4M3 -> FP16 ---")
    for mapping_name, mapping in [
        ("Standard codes", standard_mapping),
        ("AMD codes", amd_mapping),
        ("Extended codes", extended_mapping),
    ]:
        success, msg = test_dequantize_with_format(
            test_inputs, torch.float8_e4m3fn, torch.float16, dtype_code_map=mapping
        )
        print(f"  {mapping_name}: {'PASS' if success else 'FAIL'} - {msg}")

    print("\n--- Dequantize AMD E4M3 FNUZ -> FP16 ---")
    for mapping_name, mapping in [
        ("Standard codes", standard_mapping),
        ("AMD codes", amd_mapping),
        ("Extended codes", extended_mapping),
    ]:
        try:
            success, msg = test_dequantize_with_format(
                test_inputs, torch.float8_e4m3fnuz, torch.float16, dtype_code_map=mapping
            )
            print(f"  {mapping_name}: {'PASS' if success else 'FAIL'} - {msg}")
        except Exception as e:
            print(f"  {mapping_name}: ERROR - {e}")

    # Part 3: Test the actual Python interface
    print("\n" + "=" * 70)
    print("PART 3: ACTUAL PYTHON INTERFACE TEST")
    print("=" * 70)

    # Import the actual functions (assuming they're available)
    try:
        from comfy_kitchen.backends.hip import quantize_per_tensor_fp8, dequantize_per_tensor_fp8

        print("\nTesting quantize_per_tensor_fp8:")
        x = torch.tensor([0.0, 1.0, -1.0, 448.0], device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        for out_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
            try:
                result = quantize_per_tensor_fp8(x, scale, out_dtype)
                raw = result.view(torch.uint8)
                print(f"  Output dtype {out_dtype}: bytes={[f'0x{b:02x}' for b in raw.tolist()]}")

                # Check vs PyTorch
                expected = x.to(out_dtype)
                expected_raw = expected.view(torch.uint8)
                if raw.tolist() == expected_raw.tolist():
                    print(f"    MATCHES PyTorch conversion")
                else:
                    print(f"    MISMATCH: expected {[f'0x{b:02x}' for b in expected_raw.tolist()]}")
            except Exception as e:
                print(f"  ERROR: {e}")

        print("\nTesting dequantize_per_tensor_fp8:")
        for in_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
            try:
                x_f16 = torch.tensor([0.0, 1.0, -1.0, 448.0], device="cuda", dtype=torch.float16)
                x_fp8 = x_f16.to(in_dtype)
                result = dequantize_per_tensor_fp8(x_fp8, scale, torch.float16)
                print(f"  Input dtype {in_dtype}: result={result.tolist()}")

                expected = x_fp8.to(torch.float16)
                max_diff = (result - expected).abs().max().item()
                print(f"    Max diff vs PyTorch: {max_diff}")
                if max_diff > 0.01:
                    print(f"    MISMATCH: expected {expected.tolist()}")
            except Exception as e:
                print(f"  ERROR: {e}")

    except ImportError as e:
        print(f"Could not import functions: {e}")

    # Part 4: Summary and recommendations
    print("\n" + "=" * 70)
    print("PART 4: SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("""
Key findings:
1. AMD uses FNUZ variants (float8_e4m3fnuz, float8_e5m2fnuz) which have
   different bit patterns than standard FP8 (float8_e4m3fn, float8_e5m2).

2. PyTorch on ROCm handles format conversion when you do:
   - fp16 -> float8_e4m3fnuz (direct conversion to AMD format)
   - fp16 -> float8_e4m3fn (might not be supported, or converts then reinterprets)

3. The C kernels likely expect AMD native FNUZ format directly.

RECOMMENDED FIX:
- DTYPE_CODE 5 should map to torch.float8_e4m3fnuz (AMD native)
- DTYPE_CODE 6 should map to torch.float8_e5m2fnuz (AMD native)
- Remove standard FP8 types from the mapping entirely
- In quantize: input -> AMD FNUZ format directly
- In dequantize: read AMD FNUZ format, output to float16/bfloat16

The conversion fp16->float8_e4m3fnuz->float16 should be exact roundtrip
for the AMD FNUZ format.
""")


if __name__ == "__main__":
    run_full_diagnostic()
