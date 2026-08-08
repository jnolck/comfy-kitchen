# Fixes for the hipified output:
sed -i \
        -e 's/hip_bfloat16/__hip_bfloat16/g' \
        -e 's/nv_bfloat162/__hip_bfloat162/g' \
        -e 's/nv_bfloat16/__hip_bfloat16/g' \
        -e 's/__float2bfloat16_rn/__float2bfloat16/g' \
        -e 's/__float22bfloat162_rn/__float2bfloat16/g' \
        -e 's/__floats2bfloat162_rn/__lows2bfloat162/g' \
        -e 's/__dp4a/__builtin_amdgcn_sudot4/g' \
        -e 's/cudaError_t/hipError_t/g' \
        -e 's/cudaGetLastError/hipGetLastError/g' \
        -e 's/cudaSuccess/hipSuccess/g' \
        -e 's/cudaFuncSetAttribute/hipFuncSetAttribute/g' \
        -e 's/cudaFuncAttributeMaxDynamicSharedMemorySize/hipFuncAttributeMaxDynamicSharedMemorySize/g' \
        -e 's/cudaStream_t/hipStream_t/g' \
        -e 's/0xffffffff/0xffffffffull/g' \
        ./rms_rope.cu
