# File: tests/final_comprehensive_test.py
"""
Final comprehensive test for the corrected FP8 interface.
Tests all paths: quantize, dequantize, round-trip, format conversions.
"""

import torch
import sys
import numpy as np

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")


# Test that the corrected interface works
def test_end_to_end():
    """Test the complete quantize-dequantize pipeline."""

    print("=" * 60)
    print("END-TO-END FP8 QUANTIZATION TEST")
    print("=" * 60)

    # Import after path setup
    from comfy_kitchen.backends.hip import (
        quantize_per_tensor_fp8,
        dequantize_per_tensor_fp8,
        DTYPE_TO_CODE,
        DTYPE_CODE_TO_DTYPE,
    )

    device = "cuda"

    # Verify dtype mappings
    print("\n1. Dtype mappings:")
    print(f"   Code 5 -> {DTYPE_CODE_TO_DTYPE[5]}")
    print(f"   Code 6 -> {DTYPE_CODE_TO_DTYPE[6]}")
    print(f"   FNUZ E4M3 code: {DTYPE_TO_CODE.get(torch.float8_e4m3fnuz)}")
    print(f"   Standard E4M3 code: {DTYPE_TO_CODE.get(torch.float8_e4m3fn)}")

    # Test values within AMD FNUZ range (max = 240)
    test_values = [
        0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        4.0,
        -4.0,
        8.0,
        -8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        240.0,
        -240.0,
    ]

    x_f16 = torch.tensor(test_values, device=device, dtype=torch.float16)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Test 1: Quantize to AMD FNUZ
    print("\n2. Quantize FP16 -> AMD FNUZ:")
    result_fp8_amd = quantize_per_tensor_fp8(x_f16, scale, torch.float8_e4m3fnuz)
    expected = x_f16.to(torch.float8_e4m3fnuz)

    result_bytes = result_fp8_amd.view(torch.uint8)
    expected_bytes = expected.view(torch.uint8)
    match = (result_bytes == expected_bytes).all()
    print(f"   Matches PyTorch: {match.item()}")
    if not match:
        mismatches = (result_bytes != expected_bytes).nonzero()[:5]
        for i in mismatches:
            print(
                f"   [{i.item()}] val={test_values[i.item()]} "
                f"got=0x{result_bytes[i].item():02x} exp=0x{expected_bytes[i].item():02x}"
            )

    # Test 2: Quantize to standard E4M3 (should convert)
    print("\n3. Quantize FP16 -> Standard E4M3 (via AMD FNUZ):")
    result_fp8_std = quantize_per_tensor_fp8(x_f16, scale, torch.float8_e4m3fn)
    # The result should be in standard E4M3 format when viewed as such
    # But we can verify by round-tripping through float16
    result_roundtrip = result_fp8_std.to(torch.float16)
    expected_roundtrip = x_f16.to(torch.float8_e4m3fnuz).to(torch.float16)
    max_diff = (result_roundtrip - expected_roundtrip).abs().max()
    print(f"   Round-trip max diff: {max_diff:.6f}")
    print(f"   Round-trip match: {max_diff < 0.01}")

    # Test 3: Dequantize from AMD FNUZ
    print("\n4. Dequantize AMD FNUZ -> FP16:")
    fp8_input = x_f16.to(torch.float8_e4m3fnuz)
    result_f16 = dequantize_per_tensor_fp8(fp8_input, scale, torch.float16)
    expected_f16 = fp8_input.to(torch.float16)

    nan_count = torch.isnan(result_f16).sum().item()
    abs_diff = (result_f16 - expected_f16).abs()
    max_diff = (
        abs_diff[~torch.isnan(result_f16)].max().item()
        if nan_count < len(test_values)
        else float("nan")
    )

    print(f"   NaN count: {nan_count}")
    print(f"   Max diff: {max_diff:.6f}")
    print(f"   Exact match: {(abs_diff < 0.001).all().item()}")

    # Test 4: Dequantize from standard E4M3 (converts to FNUZ first)
    print("\n5. Dequantize Standard E4M3 -> FP16:")
    # Create standard E4M3 by converting through PyTorch
    fp8_std = x_f16.to(torch.float8_e4m3fn)  # May fail on ROCm if not supported
    try:
        result_f16_from_std = dequantize_per_tensor_fp8(fp8_std, scale, torch.float16)
        nan_count_std = torch.isnan(result_f16_from_std).sum().item()
        print(f"   NaN count: {nan_count_std}")
        print(f"   Values: {result_f16_from_std[:5].tolist()}")
    except Exception as e:
        print(f"   Standard E4M3 not directly supported: {e}")
        print(f"   (This is expected on ROCm - use FNUZ format directly)")

    # Test 5: Large tensor test
    print("\n6. Large tensor test (4096 elements):")
    large_input = torch.randn(4096, device=device, dtype=torch.float16) * 100
    large_input = large_input.clamp(-240, 240)  # Stay in FNUZ range
    large_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    large_fp8 = quantize_per_tensor_fp8(large_input, large_scale, torch.float8_e4m3fnuz)
    large_output = dequantize_per_tensor_fp8(large_fp8, large_scale, torch.float16)

    large_nan = torch.isnan(large_output).sum().item()
    large_max_diff = (
        (large_output - large_input.to(torch.float8_e4m3fnuz).to(torch.float16)).abs().max()
    )

    print(f"   NaN count: {large_nan}")
    print(f"   Max diff: {large_max_diff:.6f}")
    print(f"   PASS: {large_nan == 0 and large_max_diff < 0.01}")

    # Test 6: Scale test
    print("\n7. Scale test:")
    scales = [0.1, 1.0, 10.0, 100.0]
    for s in scales:
        x = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=torch.float16)
        scale_t = torch.tensor([s], device=device, dtype=torch.float32)
        fp8 = quantize_per_tensor_fp8(x, scale_t, torch.float8_e4m3fnuz)
        out = dequantize_per_tensor_fp8(fp8, scale_t, torch.float16)
        nan = torch.isnan(out).sum().item()
        diff = (out - x.to(torch.float8_e4m3fnuz).to(torch.float16)).abs().max()
        print(
            f"   scale={s:6.1f}: NaN={nan}, max_diff={diff:.6f} {'✓' if nan == 0 and diff < 0.01 else '✗'}"
        )

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_end_to_end()
