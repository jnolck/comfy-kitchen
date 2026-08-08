# File: tests/fixed_fp8_test.py
"""
Corrected test that understands AMD FNUZ format limitations
"""

import torch
import sys
import numpy as np

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

# ============================================================================
# AMD FNUZ FORMAT CONSTANTS
# ============================================================================
# From the diagnostic, we can determine:
# E4M3 FNUZ: exponent bias = 7 (NOT 8 like standard E4M3)
# The bit pattern 0x80 appears to be the max value (or clamp)
# 448.0 -> nan in FNUZ, so max is lower


def discover_amd_max():
    """Binary search to find the actual max value AMD FNUZ can represent."""
    print("Discovering AMD FNUZ max value...")
    lo, hi = 0.0, 500.0
    for _ in range(50):
        mid = (lo + hi) / 2
        t = torch.tensor([mid], dtype=torch.float16).to(torch.float8_e4m3fnuz)
        back = t.to(torch.float16)
        if torch.isnan(back).any():
            hi = mid
        else:
            lo = mid
    amd_max = lo
    print(f"  AMD E4M3 FNUZ max: {amd_max}")

    lo, hi = -500.0, 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        t = torch.tensor([mid], dtype=torch.float16).to(torch.float8_e4m3fnuz)
        back = t.to(torch.float16)
        if torch.isnan(back).any():
            lo = mid
        else:
            hi = mid
    amd_min = hi
    print(f"  AMD E4M3 FNUZ min: {amd_min}")

    return amd_max, amd_min


def decode_amd_e4m3fnuz(byte_val):
    """Decode AMD E4M3 FNUZ format.

    AMD FNUZ differs from standard E4M3:
    - Exponent bias = 7 (standard is 7? or 8?)
    - No negative zero (0x80 is used differently)
    - NaN/Inf patterns differ
    """
    sign = (byte_val >> 7) & 1
    exp = (byte_val >> 3) & 0x0F
    mant = byte_val & 0x07

    if exp == 0:
        # Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
        if mant == 0:
            return 0.0  # Zero
        val = (mant / 8.0) * (2**-6)
    elif exp == 0x0F:
        if mant == 0:
            return float("inf") if sign == 0 else float("-inf")
        else:
            return float("nan")
    else:
        # Normal: (-1)^sign * 2^(exp-8) * (1 + mant/8)
        # Note: bias might be 7 or 8, test to verify
        val = (1.0 + mant / 8.0) * (2 ** (exp - 7))

    return -val if sign else val


def inspect_amd_format():
    """Comprehensive inspection of AMD FNUZ format."""
    print("\n" + "=" * 60)
    print("AMD E4M3 FNUZ FORMAT INSPECTION")
    print("=" * 60)

    # Check what PyTorch does with known values
    test_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 448.0]

    print(
        f"{'Input':>10} {'Byte':>6} {'Sign':>4} {'Exp':>4} {'Mant':>4} {'Decoded':>10} {'PyTorch':>10} {'Match':>6}"
    )
    print("-" * 60)

    for val in test_values:
        t_f16 = torch.tensor([val], dtype=torch.float16)
        t_f8 = t_f16.to(torch.float8_e4m3fnuz)
        byte_val = t_f8.view(torch.uint8).item()

        sign = (byte_val >> 7) & 1
        exp = (byte_val >> 3) & 0x0F
        mant = byte_val & 0x07

        decoded = decode_amd_e4m3fnuz(byte_val)
        pytorch_val = t_f8.to(torch.float16).item()

        match = (
            "✓"
            if abs(decoded - pytorch_val) < 0.01 or (np.isnan(decoded) and np.isnan(pytorch_val))
            else "✗"
        )

        print(
            f"{val:10.1f} 0x{byte_val:02x}   {sign:4d} {exp:4d} {mant:4d} {decoded:10.4f} {pytorch_val:10.4f} {match:>6}"
        )

    # Check the NaN pattern
    print("\nNaN patterns in AMD FNUZ:")
    nan_patterns = [0x7F, 0xFF, 0x7E, 0xFE]
    for pattern in nan_patterns:
        t = torch.tensor([pattern], dtype=torch.uint8).view(torch.float8_e4m3fnuz)
        back = t.to(torch.float16)
        print(f"  0x{pattern:02x} -> {back.item()}")


def test_with_correct_format():
    """Test quantize/dequantize using ONLY AMD FNUZ format."""

    print("\n" + "=" * 60)
    print("TESTING WITH CORRECT AMD FNUZ FORMAT")
    print("=" * 60)

    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # AMD FNUZ-safe values (within the valid range)
    safe_values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 128.0, -128.0, 256.0, -256.0],
        device=device,
        dtype=torch.float16,
    )

    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Test 1: Quantize FP16 -> AMD FNUZ
    print("\n1. Quantize FP16 -> AMD FNUZ:")
    output_uint8 = torch.zeros(safe_values.shape, device=device, dtype=torch.uint8)

    # Use dtype code 1 (FP16) -> 5 (should be FNUZ)
    _C.quantize_per_tensor_fp8(
        safe_values,
        scale,
        output_uint8,
        1,
        5,  # FP16 -> E4M3 FNUZ
        safe_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Compare with PyTorch's conversion
    expected_fp8 = safe_values.to(torch.float8_e4m3fnuz)
    expected_bytes = expected_fp8.view(torch.uint8)

    output_bytes = output_uint8
    matches = output_bytes == expected_bytes

    print(f"  Matches: {matches.sum().item()}/{safe_values.numel()}")
    if not matches.all():
        print(f"  Got:      {[f'0x{b:02x}' for b in output_bytes.tolist()]}")
        print(f"  Expected: {[f'0x{b:02x}' for b in expected_bytes.tolist()]}")
        mismatches = [i for i, m in enumerate(matches) if not m]
        for i in mismatches:
            print(
                f"  [{i}] val={safe_values[i].item():.1f} "
                f"got=0x{output_bytes[i].item():02x} "
                f"exp=0x{expected_bytes[i].item():02x}"
            )

    # Test 2: Dequantize AMD FNUZ -> FP16
    print("\n2. Dequantize AMD FNUZ -> FP16:")
    fp8_input = safe_values.to(torch.float8_e4m3fnuz)
    output_f16 = torch.zeros(safe_values.shape, device=device, dtype=torch.float16)

    _C.dequantize_per_tensor_fp8(
        fp8_input.view(torch.uint8),
        scale,
        output_f16,
        5,
        1,  # E4M3 FNUZ -> FP16
        safe_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Check for NaN
    nan_mask = torch.isnan(output_f16)
    if nan_mask.any():
        nan_indices = nan_mask.nonzero().flatten()
        print(f"  NaN in output at indices: {nan_indices.tolist()}")
        for i in nan_indices:
            fp8_byte = fp8_input.view(torch.uint8)[i].item()
            print(f"    [{i}] fp8=0x{fp8_byte:02x} val={safe_values[i].item():.1f}")

    # Compare with expected
    expected_f16 = fp8_input.to(torch.float16)
    abs_diff = (output_f16 - expected_f16).abs()
    max_diff = abs_diff.max().item()
    print(f"  Max diff vs PyTorch: {max_diff}")

    if max_diff > 0.01:
        mismatches = (abs_diff > 0.01).nonzero().flatten()
        for i in mismatches[:5]:
            print(f"  [{i}] got={output_f16[i].item():.4f} exp={expected_f16[i].item():.4f}")


def test_boundary_values():
    """Test values at the boundary of AMD FNUZ format."""

    print("\n" + "=" * 60)
    print("BOUNDARY VALUE TESTING")
    print("=" * 60)

    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Test values around the AMD FNUZ max
    # We determined from the diagnostic that 448.0 -> NaN in FNUZ
    # The max is likely around 240-256
    boundary_values = torch.tensor(
        [
            0.0,
            1.0,
            -1.0,
            240.0,
            -240.0,
            248.0,
            -248.0,
            256.0,
            -256.0,
            384.0,
            -384.0,
            448.0,
            -448.0,
        ],
        device=device,
        dtype=torch.float16,
    )

    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    print("\nHow PyTorch handles these values:")
    pytorch_fp8 = boundary_values.to(torch.float8_e4m3fnuz)
    pytorch_bytes = pytorch_fp8.view(torch.uint8)
    pytorch_back = pytorch_fp8.to(torch.float16)

    for i in range(len(boundary_values)):
        val = boundary_values[i].item()
        byte_val = pytorch_bytes[i].item()
        back_val = pytorch_back[i].item()
        is_nan = np.isnan(back_val)
        print(f"  {val:8.1f} -> 0x{byte_val:02x} -> {back_val:8.1f} {'(NaN!)' if is_nan else ''}")

    print("\nHow our kernel handles these values:")
    output_uint8 = torch.zeros(boundary_values.shape, device=device, dtype=torch.uint8)

    _C.quantize_per_tensor_fp8(
        boundary_values,
        scale,
        output_uint8,
        1,
        5,  # FP16 -> E4M3 FNUZ
        boundary_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    output_bytes = output_uint8
    # Interpret as AMD FNUZ
    output_fp8 = output_bytes.view(torch.float8_e4m3fnuz)
    output_back = output_fp8.to(torch.float16)

    for i in range(len(boundary_values)):
        val = boundary_values[i].item()
        byte_val = output_bytes[i].item()
        back_val = output_back[i].item()
        expected_byte = pytorch_bytes[i].item()
        is_nan = np.isnan(back_val)
        matches = byte_val == expected_byte

        print(
            f"  {val:8.1f} -> 0x{byte_val:02x} (exp:0x{expected_byte:02x}) -> {back_val:8.1f} "
            f"{'(NaN!)' if is_nan else ''} {'✓' if matches else '✗'}"
        )


if __name__ == "__main__":
    # Discover the actual AMD FNUZ limits
    amd_max, amd_min = discover_amd_max()

    # Inspect the format
    inspect_amd_format()

    # Run corrected tests
    test_with_correct_format()
    test_boundary_values()

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print(f"""
The AMD FNUZ format has a maximum value of approximately {amd_max:.1f} (not 448.0).
When your kernel tries to represent 448.0 in AMD FNUZ, it produces 0x7f/0xff
which are NaN patterns in AMD FNUZ.

FIX NEEDED:
1. In the quantize kernel: clamp input to AMD FNUZ max ({amd_max:.1f})
   NOT the standard E4M3 max (448.0)
2. Use the correct exponent bias for AMD FNUZ (discovered from inspection)
3. Ensure dtype code 5 maps to torch.float8_e4m3fnuz everywhere
""")
