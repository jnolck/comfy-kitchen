# Direct test: call the C++ kernel with raw data and check output
import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

device = "cuda"
stream = torch.cuda.current_stream().cuda_stream

# Simplest possible test: one value, known encoding
# NVFP4 nibble 0x4 = 2.0 (E2M1: sign=0, exp=2, mant=0 → 2.0)
# With scale=1.0, output should be 2.0
qx = torch.zeros(16, 8, device=device, dtype=torch.uint8)  # 16 rows, 8 bytes = 16 values
qx[0, 0] = 0x44  # Both nibbles = 4, both decode to 2.0

per_tensor_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
block_scales = torch.ones(16, 2, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

# Call kernel directly through the C interface
output = torch.zeros(16, 16, device=device, dtype=torch.float32)
_C.dequantize_nvfp4(
    qx,
    per_tensor_scale,
    block_scales.view(torch.uint8),
    output,
    0,  # output_dtype_code for float32
    True,  # hi_first
    stream,
)
torch.cuda.synchronize()

print(f"Input byte: 0x{qx[0, 0].item():02x}")
print(f"Block scale byte: 0x{block_scales.view(torch.uint8)[0, 0].item():02x}")
print(f"Expected: 2.0, Got: {output[0, 0].item():.4f}")
print(f"Expected: 2.0, Got: {output[0, 1].item():.4f}")
