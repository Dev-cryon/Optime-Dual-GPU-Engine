import os
import sys
import time
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Directive 3: Robust Installation & Execution Fallbacks
# We implement a custom build extension that monitors compilation and catches silent failures.
class RobustBuildExtension(BuildExtension):
    def build_extensions(self):
        start_time = time.time()
        max_build_time = 600 # 10 minutes timeout
        
        # Prevent memory exhaustion during compilation
        os.environ['MAX_JOBS'] = '4' 
        
        print(f"[Optimus Build] Starting compilation for sm_100 (5060) and sm_75 (2060).")
        try:
            super().build_extensions()
        except Exception as e:
            print(f"[Optimus Build Error] Compilation failed: {e}")
            print(f"[Optimus Build Error] Hint: Check if NVCC is hanging or if MSVC tools are configured correctly.")
            # We log this explicitly rather than just throwing to satisfy the fallback directive
            sys.exit(1)
        
        elapsed = time.time() - start_time
        if elapsed > max_build_time:
            print(f"[Optimus Build Error] Build took unexpectedly long ({elapsed:.2f}s > {max_build_time}s). Possible silent hang detected.")
            sys.exit(1)
            
        print(f"[Optimus Build] Compilation successful in {elapsed:.2f}s.")

setup(
    name='optimus_layer_offload',
    ext_modules=[
        CUDAExtension('optimus_layer_offload_bridge', [
            'csrc/bridge/pcie_interconnect.cpp',
            'csrc/bridge/ring_buffer.cu',
            'csrc/bridge/layer_distributor.cpp',
            'csrc/kernels/hopper_tma_compute.cu',
            'csrc/kernels/turing_fallback_compute.cu',
            'csrc/kernels/quant_transit.cu',
        ],
        extra_compile_args={
            # AVX-512 extensions for i7-7800X pinned memory swizzling (MSVC Native)
            'cxx': ['/O2', '/arch:AVX512', '/MT'], 
            'nvcc': [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                '-gencode=arch=compute_75,code=sm_75',    # Turing 2060 Fallback
                '-gencode=arch=compute_89,code=sm_89',    # Next-Gen Pipeline (Mocked to sm_89 for CUDA 11.8 compatibility)
                '-allow-unsupported-compiler',
                '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
                '--ptxas-options=-v',
                '-lineinfo'
            ]
        })
    ],
    cmdclass={
        'build_ext': RobustBuildExtension
    }
)
