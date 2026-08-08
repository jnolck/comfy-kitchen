# Quick sanity check - add this to your test
import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")

from comfy_kitchen.backends.hip import quantize_per_tensor_fp8, dequantize_per_tensor_fp8

device = "cuda"
torch.manual_seed(42)

# Simulate what the model loading does
x = torch.randn(4, 4, device=device, dtype=torch.bfloat16) * 5
max_abs = x.abs().max()
fp8_max = 224.0
scale = max_abs / fp8_max

print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Max abs: {max_abs:.4f}")
print(f"Scale: {scale:.6f}")

# Quantize
scale_t = torch.tensor([scale], device=device, dtype=torch.float32)
fp8 = quantize_per_tensor_fp8(x, scale_t, torch.float8_e4m3fnuz)


# Check for NaN in FP8
fp8_bytes = fp8.view(torch.uint8)
nan_count = ((fp8_bytes >> 3) & 0x0F == 0x0F).sum().item()
print(f"FP8 NaN count: {nan_count}")

# Dequantize
out = dequantize_per_tensor_fp8(fp8, scale_t, torch.bfloat16)
out_nan = torch.isnan(out).sum().item()
print(f"Output NaN: {out_nan}")

# Check round-trip
diff = (out.float() - x.float()).abs()
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# Check if values are in the right ballpark
print(f"Input sample:  {x.flatten()[:4]}")
print(f"Output sample: {out.flatten()[:4]}")

# The key question: are input and output correlated?
# If they're completely different, the scale convention is inverted
ratio = out.flatten()[:8].float() / (x.flatten()[:8].float() + 1e-8)
print(f"Output/Input ratio: {ratio}")
