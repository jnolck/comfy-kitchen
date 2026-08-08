# Quick debug script to find the crash point
import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

device = "cuda"
x = torch.randn(64, 64, device=device, dtype=torch.bfloat16) * 5

print(f"Input: dtype={x.dtype}, shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")

# Step 1: What does QuantizedTensor.from_float do?
# It likely computes scale = max_abs / 240.0 or similar
max_abs = x.abs().max()
print(f"Max abs: {max_abs:.4f}")

# The scale convention: fp8 = round(input / scale)
# So scale should be roughly max_abs / 240
# If max_abs=~25 (5*5 for randn*5), scale ~= 25/240 ~= 0.1
expected_scale = max_abs.item() / 240.0
print(f"Expected scale: {expected_scale:.4f}")

# Step 2: Try quantizing directly
from comfy_kitchen.backends.hip import quantize_per_tensor_fp8, dequantize_per_tensor_fp8

# We need to know what scale comfy computes internally
# Let's try a few plausible scales
for scale_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, max_abs.item() / 240.0]:
    scale = torch.tensor([scale_val], device=device, dtype=torch.float32)

    try:
        fp8 = quantize_per_tensor_fp8(x, scale, torch.float8_e4m3fnuz)
        out = dequantize_per_tensor_fp8(fp8, scale, torch.bfloat16)

        nan_count = torch.isnan(out).sum().item()
        inf_count = torch.isinf(out).sum().item()
        max_diff = (out.float() - x.float()).abs().max().item()

        status = "✓" if (nan_count == 0 and inf_count == 0) else "✗"
        print(
            f"  scale={scale_val:.4f}: NaN={nan_count} Inf={inf_count} max_diff={max_diff:.4f} {status}"
        )
    except Exception as e:
        print(f"  scale={scale_val:.4f}: ERROR - {e}")
