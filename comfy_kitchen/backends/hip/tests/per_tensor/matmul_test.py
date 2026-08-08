# File: tests/matmul_test.py
import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")

device = "cuda"
torch.manual_seed(42)

# Create weight and input tensors
weight = torch.randn(256, 256, device=device, dtype=torch.bfloat16) * 0.1
x = torch.randn(1, 256, device=device, dtype=torch.bfloat16) * 5

# Import and use the string name
from comfy_kitchen.tensor.base import QuantizedTensor

# Use the string name that get_layout_class expects
qt_weight = QuantizedTensor.from_float(weight, "TensorCoreFP8Layout")
qt_input = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")

print(f"Weight scale: {qt_weight._params.scale.item():.6f}")
print(f"Input scale: {qt_input._params.scale.item():.6f}")

# Check for NaN in quantized data
w_bytes = qt_weight._qdata.view(torch.uint8)
x_bytes = qt_input._qdata.view(torch.uint8)
w_nan = ((w_bytes >> 3) & 0x0F == 0x0F).sum().item()
x_nan = ((x_bytes >> 3) & 0x0F == 0x0F).sum().item()
print(f"Weight FP8 NaN: {w_nan}/{w_bytes.numel()}")
print(f"Input FP8 NaN: {x_nan}/{x_bytes.numel()}")

# FP16 baseline
output_fp16 = torch.nn.functional.linear(x, weight)
print(
    f"\nFP16 baseline: range=[{output_fp16.min():.4f}, {output_fp16.max():.4f}], NaN={torch.isnan(output_fp16).sum().item()}"
)

# FP8 linear
output_qt = torch.nn.functional.linear(qt_input, qt_weight)
print(
    f"FP8 linear: range=[{output_qt.min():.4f}, {output_qt.max():.4f}], NaN={torch.isnan(output_qt).sum().item()}"
)

if not torch.isnan(output_qt).any():
    diff = (output_qt.float() - output_fp16.float()).abs()
    print(f"Max diff vs FP16: {diff.max():.6f}")
    print(f"Mean diff vs FP16: {diff.mean():.6f}")
else:
    print("OUTPUT HAS NaN!")

# Add to your test:
# fp8_bytes = fp8.view(torch.uint8)
# expected_bytes = x.to(torch.float8_e4m3fnuz).view(torch.uint8)
# for i in range(16):
#     if fp8_bytes[i] != expected_bytes[i]:
#         print(
#             f"  [{i}] val={x.flatten()[i].item():.4f} kernel=0x{fp8_bytes[i].item():02x} pytorch=0x{expected_bytes[i].item():02x}"
#         )
