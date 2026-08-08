import torch

# What does PyTorch produce?
val = torch.tensor([224.0], dtype=torch.float16).to(torch.float8_e4m3fnuz)
print(f"PyTorch: 224.0 -> 0x{val.view(torch.uint8).item():02x}")

# What does the HIP runtime produce? Let's check with a tiny kernel
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

# Test the quantize kernel directly with a single value
scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
test_input = torch.tensor([224.0], device="cuda", dtype=torch.float16)
output = torch.zeros(1, device="cuda", dtype=torch.uint8)

_C.quantize_per_tensor_fp8(
    test_input, scale, output, 1, 5, 1, torch.cuda.current_stream().cuda_stream
)
torch.cuda.synchronize()

print(f"Kernel:  224.0 -> 0x{output.item():02x}")

# Also test negative boundary
test_input_neg = torch.tensor([-224.0], device="cuda", dtype=torch.float16)
_C.quantize_per_tensor_fp8(
    test_input_neg, scale, output, 1, 5, 1, torch.cuda.current_stream().cuda_stream
)
torch.cuda.synchronize()
print(f"Kernel: -224.0 -> 0x{output.item():02x}")

# What about 120?
test_input_120 = torch.tensor([120.0], device="cuda", dtype=torch.float16)
_C.quantize_per_tensor_fp8(
    test_input_120, scale, output, 1, 5, 1, torch.cuda.current_stream().cuda_stream
)
torch.cuda.synchronize()
print(f"Kernel:  120.0 -> 0x{output.item():02x}")

# Check all boundary values
for v in [64, 120, 128, 192, 224, 240]:
    t = torch.tensor([float(v)], device="cuda", dtype=torch.float16)
    # PyTorch
    pt = t.to(torch.float8_e4m3fnuz).view(torch.uint8).item()
    # Kernel
    _C.quantize_per_tensor_fp8(t, scale, output, 1, 5, 1, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    k = output.item()
    match = "✓" if pt == k else "✗"
    print(f"  {v:6.0f}: PyTorch=0x{pt:02x} Kernel=0x{k:02x} {match}")
