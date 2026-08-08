# More verbose test
import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

# Test with known values, small enough to see everything
x = torch.arange(16, device="cuda", dtype=torch.float16)  # 0.0 to 15.0
scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
x_fp8 = x.to(torch.float8_e4m3fnuz)
output = torch.zeros(16, device="cuda", dtype=torch.float16)
stream_ptr = torch.cuda.current_stream().cuda_stream

print(f"Input: {x}")
print(f"FP8: {x_fp8}")
print(f"numel: {x.numel()}")
print(f"blocks needed: {(x.numel() + 8 * 128 - 1) // (8 * 128)}")

_C.dequantize_per_tensor_fp8(x_fp8, scale, output, 5, 1, x.numel(), stream_ptr)

torch.cuda.synchronize()
print(f"Output: {output}")
print(f"Expected: {x_fp8.to(torch.float16)}")

# Print element-by-element comparison
#
for i in range(16):
    match = "✓" if abs(output[i].item() - expected[i].item()) < 0.01 else "✗"
    print(f"  [{i}] out={output[i].item():.2f} exp={expected[i].item():.2f} {match}")
