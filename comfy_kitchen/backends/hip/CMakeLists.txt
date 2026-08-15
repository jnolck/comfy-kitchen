cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_HIP_ARCHITECTURES "gfx1100")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Debug build enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ggdb3 -O0")
else()
        message(STATUS "Release build enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
endif()

project(comfy_kitchen_rocm LANGUAGES CXX HIP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find ROCm/HIP
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_RESULT
)
if(NOT NB_RESULT EQUAL 0)
        message(FATAL_ERROR "nanobind not found. Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

set(CPP_SOURCES
    dlpack_bindings.cpp
)

set(HIP_SOURCES
    ops/int8_linear.cu
    ops/cublas_gemm_int8.cu
    ops/ck_gemm_int8.cu
    ops/apply_rope.cu
    ops/adaln.cu
    ops/per_tensor_quantize.cu
    ops/quantize_nvfp4.cu
    ops/cublas_gemm_nvfp4.cu
    ops/convrot_w4a4.cu
    ops/rms_rope.cu
    ops/w4a8_gemm.cu
    #ops/gemv_awq.hip
    ops/svdquant_w4a4.hip
)

set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

# Create the nanobind module
nanobind_add_module(_C NB_STATIC STABLE_ABI LTO ${CPP_SOURCES} ${HIP_SOURCES})
set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000 NB_STATIC)

# Include directories
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hipblaslt
    ${ROCM_PATH}/include/hipblas
)

if(NOT DEFINED COMFY_CK_DIR)
        set(COMFY_CK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/composable_kernel/include")
endif()
if(EXISTS "${COMFY_CK_DIR}/ck_tile/core.hpp")
        message(STATUS "CK found: ${COMFY_CK_DIR}")
        target_include_directories(_C PRIVATE ${COMFY_CK_DIR})
        target_compile_definitions(_C PRIVATE COMFY_HAVE_CK)
else()
        message(STATUS "CK not found at ${COMFY_CK_DIR} - fused int8 GEMM uses hipBLASLt only")
endif()

# Link libraries
target_link_libraries(_C PRIVATE
    hip::host
    hip::device
    roc::rocblas
    ${CMAKE_DL_LIBS}
)

# After the _C target is created
if(TARGET composable_kernel::device)
        target_link_libraries(_C PRIVATE composable_kernel::device)
endif()

# RPATH
set_target_properties(_C PROPERTIES
    LINKER_LANGUAGE HIP
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "${ROCM_PATH}/lib"
)

install(TARGETS _C
    LIBRARY DESTINATION .
)
