# File: tests/test_nvfp4_dequant.py
import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

device = "cuda"

# Create fake NVFP4 data: 16 rows, 32 FP4 values (16 bytes packed)
# NVFP4: 2 values per byte, block size 16
num_rows, num_cols = 16, 32
qx = torch.randint(0, 256, (num_rows, num_cols // 2), device=device, dtype=torch.uint8)
per_tensor_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

# Block scales: one E4M3 scale per 16-element block
block_scales_fp8 = torch.ones(num_rows, num_cols // 16, device=device).to(torch.float8_e4m3fn)

from comfy_kitchen.backends.hip import dequantize_nvfp4

# Test round-trip
output = dequantize_nvfp4(qx, per_tensor_scale, block_scales_fp8, torch.float16)
print(f"Output shape: {output.shape}")  # Should be (16, 32)
print(f"Output dtype: {output.dtype}")
print(f"NaN count: {torch.isnan(output).sum().item()}")
print(f"Output sample: {output[0, :4]}")

# Compare with eager backend if available
try:
    from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4 as eager_dequant

    output_eager = eager_dequant(qx, per_tensor_scale, block_scales_fp8, torch.float16)
    diff = (output - output_eager).abs()
    print(f"HIP vs EAGER max diff: {diff.max().item():.6f}")
except Exception as e:
    print(f"Eager comparison not available: {e}")
