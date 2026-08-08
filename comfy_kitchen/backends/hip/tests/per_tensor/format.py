# File: tests/debug_dequant.py
"""Test exactly what dequantize produces vs expected."""

import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

# Test 1: Round-trip with standard E4M3 input (simulating a real FP8 model)
print("=" * 60)
print("TEST 1: Standard E4M3 -> dequantize -> compare with PyTorch")
print("=" * 60)

device = "cuda"
torch.manual_seed(42)

# Create values like a real model would have
x_f32 = torch.randn(4, 4, device=device, dtype=torch.float32) * 5

# PyTorch's standard E4M3 round-trip (ground truth)
x_std_e4m3 = x_f32.to(torch.float8_e4m3fn)
x_std_back = x_std_e4m3.to(torch.float32)
print(f"PyTorch std E4M3 roundtrip: {x_std_back.flatten()[:4].tolist()}")

# Add to your test:
x_std = torch.tensor([1.0], device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
print(f"Standard E4M3 byte: 0x{x_std.view(torch.uint8).item():02x}")  # Should be 0x38

# Convert to AMD FNUZ via your function
from comfy_kitchen.backends.hip import dequantize_per_tensor_fp8


# Add to your test:
# In your test, replace scale_std with:
# This is what fp8.py actually computes
# Simulate what the model checkpoint stores:
# What if dequantize should be dividing?
# The kernel does: output = fp8_val * scale
# If quantize does: fp8 = input * scale, then dequantize should do: output = fp8 / scale
# scale = max_abs / 448 (standard E4M3 convention)
max_abs = x_f32.abs().max()
scale_std = max_abs / 448.0
print(f"Max abs: {max_abs:.4f}, scale_std: {scale_std:.6f}")

scale_t = torch.tensor([scale_std], device=device, dtype=torch.float32)

# Now call the HIP dequantize (with and without scale adjustment)
from comfy_kitchen.backends.hip import dequantize_per_tensor_fp8

# WITHOUT scale adjustment (current code)
result_no_adj = dequantize_per_tensor_fp8(x_std_e4m3.clone(), scale_t.clone(), torch.float32)
print(f"\nWithout scale adjustment:")
print(f"  Result: {result_no_adj.flatten()[:4].tolist()}")
print(f"  Expected: {x_std_back.flatten()[:4].tolist()}")
print(f"  Max diff: {(result_no_adj - x_std_back).abs().max():.6f}")

# WITH scale adjustment (448/240)
scale_adj = scale_t * (448.0 / 240.0)
result_with_adj = dequantize_per_tensor_fp8(x_std_e4m3.clone(), scale_adj.clone(), torch.float32)
print(f"\nWith scale adjustment (448/240):")
print(f"  Result: {result_with_adj.flatten()[:4].tolist()}")
print(f"  Expected: {x_std_back.flatten()[:4].tolist()}")
print(f"  Max diff: {(result_with_adj - x_std_back).abs().max():.6f}")

# What scale gives the right answer?
print(f"\nFinding correct scale factor...")
for factor in [1.0, 448.0 / 240.0, 240.0 / 448.0, 448.0, 240.0]:
    scale_test = scale_t * factor
    result = dequantize_per_tensor_fp8(x_std_e4m3.clone(), scale_test.clone(), torch.float32)
    diff = (result - x_std_back).abs().max()
    print(f"  factor={factor:.4f}: max_diff={diff:.6f}")

# Test 2: Check what the kernel actually computes
print(f"\n" + "=" * 60)
print("TEST 2: Raw kernel output inspection")
print("=" * 60)

# Call kernel directly with a single value
test_val = torch.tensor([1.0], device=device, dtype=torch.float32)
test_fp8 = test_val.to(torch.float8_e4m3fn)
test_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

result_raw = torch.zeros(1, device=device, dtype=torch.float32)
_C.dequantize_per_tensor_fp8(
    test_fp8.view(torch.uint8),
    test_scale,
    result_raw,
    5,  # input_dtype_code (E4M3)
    0,  # output_dtype_code (float32)
    1,
    torch.cuda.current_stream().cuda_stream,
)
torch.cuda.synchronize()
print(f"FP8 byte: 0x{test_fp8.view(torch.uint8).item():02x}")
print(f"Kernel output (scale=1.0): {result_raw.item():.6f}")
print(f"PyTorch output (scale=1.0): {test_fp8.to(torch.float32).item():.6f}")

# What does 0x38 mean in FNUZ vs standard?
# Standard E4M3 0x38 = 0.5 (bias=7)
# AMD FNUZ 0x38 = 0.5 (bias=8, exp=7, 2^(7-8) = 0.5)
# Both are 0.5! So 1.0 -> 0x40 in both formats
