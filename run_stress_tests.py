import os
import subprocess
import sys

def main():
    print("===========================================")
    print("[Optimus Stress Test Runner]")
    print("===========================================")
    
    os.makedirs("tests", exist_ok=True)
    
    print("Compiling tests/stress_test_quant.cu using NVCC...")
    # Targeting sm_75 as lowest common denominator for the test
    cmd_compile = [
        "nvcc", 
        "tests/stress_test_quant.cu", 
        "-o", "tests/stress_test_quant.exe", 
        "-O3", 
        "-gencode=arch=compute_75,code=sm_75",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ]
    
    cmd_compile_bridge = [
        "nvcc", 
        "tests/stress_test_bridge.cu", 
        "-o", "tests/stress_test_bridge.exe", 
        "-O3", 
        "-gencode=arch=compute_75,code=sm_75",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ]
    
    cmd_compile_compute = [
        "nvcc", 
        "tests/stress_test_compute.cu", 
        "-o", "tests/stress_test_compute.exe", 
        "-O3", 
        "-gencode=arch=compute_75,code=sm_75",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ]
    
    cmd_compile_overflow = [
        "nvcc", 
        "tests/stress_test_overflow.cu", 
        "-o", "tests/stress_test_overflow.exe", 
        "-O3", 
        "-gencode=arch=compute_75,code=sm_75",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ]
    
    cmd_compile_loader = [
        "nvcc", 
        "tests/stress_test_loader.cu", 
        "-o", "tests/stress_test_loader.exe", 
        "-O3",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ]
    
    try:
        subprocess.run(cmd_compile, check=True)
        print("Compilation of quant_transit successful.")
        
        print("Compiling tests/stress_test_bridge.cu using NVCC...")
        subprocess.run(cmd_compile_bridge, check=True)
        print("Compilation of bridge successful.")
        
        print("Compiling tests/stress_test_compute.cu using NVCC...")
        subprocess.run(cmd_compile_compute, check=True)
        print("Compilation of compute successful.")
        
        print("Compiling tests/stress_test_overflow.cu using NVCC...")
        subprocess.run(cmd_compile_overflow, check=True)
        print("Compilation of overflow successful.")
        
        print("Compiling tests/stress_test_loader.cu using NVCC...")
        subprocess.run(cmd_compile_loader, check=True)
        print("Compilation of loader successful.\n")
    except subprocess.CalledProcessError:
        print("[Fatal] NVCC Compilation failed. Ensure MSVC and CUDA Toolkit are in PATH.")
        sys.exit(1)
    except FileNotFoundError:
        print("[Fatal] NVCC not found. Please run this in an x64 Native Tools Command Prompt.")
        sys.exit(1)
        
    print("Executing tests\\stress_test_bridge.exe...\n")
    try:
        subprocess.run(["tests\\stress_test_bridge.exe"], check=True)
    except subprocess.CalledProcessError:
        print("[Fatal] Bridge stress test encountered a runtime error.")
        
    print("Executing tests\\stress_test_quant.exe...\n")
    try:
        subprocess.run(["tests\\stress_test_quant.exe"], check=True)
    except subprocess.CalledProcessError:
        print("[Fatal] Quant stress test encountered a runtime error.")
        sys.exit(1)
        
    print("Executing tests\\stress_test_compute.exe...\n")
    try:
        subprocess.run(["tests\\stress_test_compute.exe"], check=True)
    except subprocess.CalledProcessError:
        print("[Fatal] Compute stress test encountered a runtime error.")
        sys.exit(1)
        
    print("Executing tests\\stress_test_overflow.exe...\n")
    try:
        subprocess.run(["tests\\stress_test_overflow.exe"], check=True)
    except subprocess.CalledProcessError:
        print("[Fatal] Overflow stress test encountered a runtime error.")
        sys.exit(1)
        
    print("Executing tests\\stress_test_loader.exe...\n")
    try:
        subprocess.run(["tests\\stress_test_loader.exe"], check=True)
    except subprocess.CalledProcessError:
        print("[Fatal] Loader stress test encountered a runtime error.")
        sys.exit(1)

if __name__ == "__main__":
    main()
