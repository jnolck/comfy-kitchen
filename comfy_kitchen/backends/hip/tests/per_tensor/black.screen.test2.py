# File: tests/comfy_trace_test.py
"""Trace exactly what ComfyUI's QuantizedTensor does."""

import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")

# Try to import QuantizedTensor
try:
    from comfy.quantization import QuantizedTensor

    print("✓ Imported QuantizedTensor from comfy.quantization")
except:
    try:
        from comfy_kitchen import QuantizedTensor

        print("✓ Imported QuantizedTensor from comfy_kitchen")
    except:
        print("✗ Could not import QuantizedTensor - need to find where it lives")
        sys.exit(1)

device = "cuda"
torch.manual_seed(42)
x = torch.randn(64, 64, device=device, dtype=torch.bfloat16) * 5

print(f"\nInput: shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")

# Step 1: Quantize
print("\n1. Quantizing...")
try:
    qt = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
    print(f"   qt.dtype: {qt.quantized_data.dtype if hasattr(qt, 'quantized_data') else 'unknown'}")
    print(f"   qt.scale: {qt.scale if hasattr(qt, 'scale') else 'unknown'}")

    # Inspect internals
    if hasattr(qt, "quantized_data"):
        qd = qt.quantized_data
        print(f"   quantized_data: dtype={qd.dtype}, shape={qd.shape}")
        raw = qd.view(torch.uint8)
        nan_count = ((raw >> 3) & 0x0F == 0x0F).sum().item()  # FNUZ NaN pattern
        print(f"   FP8 NaN count: {nan_count}")
except Exception as e:
    print(f"   ✗ Quantize failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 2: Dequantize
print("\n2. Dequantizing...")
try:
    dq = qt.dequantize()
    print(f"   dq: dtype={dq.dtype}, shape={dq.shape}")
    nan_count = torch.isnan(dq).sum().item()
    inf_count = torch.isinf(dq).sum().item()
    print(f"   NaN: {nan_count}, Inf: {inf_count}")

    max_diff = (dq.float() - x.float()).abs().max().item()
    print(f"   Max diff: {max_diff:.6f}")

    # Check allclose
    passes = torch.allclose(dq, x, rtol=0.1, atol=0.1)
    print(f"   allclose(rtol=0.1, atol=0.1): {passes}")

    if not passes:
        # Find worst offenders
        diff = (dq.float() - x.float()).abs()
        worst_idx = diff.argmax()
        print(
            f"   Worst: idx={worst_idx.item()}, x={x.flatten()[worst_idx].item():.4f}, dq={dq.flatten()[worst_idx].item():.4f}"
        )

except Exception as e:
    print(f"   ✗ Dequantize failed: {e}")
    import traceback

    traceback.print_exc()

# Step 3: Check what layout/scale convention ComfyUI uses
print("\n3. Layout inspection...")
print(f"   Layout: TensorCoreFP8Layout")
if hasattr(qt, "scale"):
    s = qt.scale
    if isinstance(s, torch.Tensor):
        print(f"   Scale value: {s.item()}")
        print(f"   Scale dtype: {s.dtype}")
    else:
        print(f"   Scale: {s}")
