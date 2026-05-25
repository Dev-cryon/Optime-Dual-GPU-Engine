#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>

// Include implementations for standalone testing
#include "../csrc/bridge/vram_monitor.cpp"

extern "C" {
    void init_fallback_pool(size_t size_bytes);
    extern void* g_dev0_fallback_ptr;
    extern void* g_dev1_fallback_ptr;
}

#include "../csrc/bridge/ring_buffer.cu"

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__ void zero_copy_kv_read_kernel(const half* ptr, half* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = ptr[idx];
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "[Optimus Stress Test] Hardcore VRAM Overflow Mechanism" << std::endl;
    std::cout << "===========================================" << std::endl;

    // ---------------------------------------------------------
    // 1. Test VRAM Monitor
    // ---------------------------------------------------------
    VRAMMonitor monitor(0.80f); // Set strict 80% threshold for test
    
    std::cout << "Testing active VRAM polling (Threshold 80%)..." << std::endl;
    bool risk = monitor.check_overflow_risk(0);
    std::cout << "Initial Risk (Device 0): " << (risk ? "HIGH (Spillover Triggered)" : "LOW") << std::endl;
    
    half* artificial_bloat;
    size_t free_b, total_b;
    cudaSetDevice(0);
    cudaMemGetInfo(&free_b, &total_b);
    
    // Allocate 85% of total VRAM
    size_t bloat_size = static_cast<size_t>(total_b * 0.85);
    std::cout << "Allocating " << (bloat_size / (1024*1024*1024)) << "GB of artificial bloat to trigger OOM fallback..." << std::endl;
    
    cudaError_t alloc_err = cudaMalloc(&artificial_bloat, bloat_size);
    if (alloc_err == cudaSuccess) {
        risk = monitor.check_overflow_risk(0);
        std::cout << "Post-Bloat Risk (Device 0): " << (risk ? "HIGH (Spillover Triggered)" : "LOW") << std::endl;
        
        if (!risk) {
            std::cout << "[WARNING] VRAM Monitor failed to detect the massive allocation." << std::endl;
        } else {
            std::cout << "[SUCCESS] VRAM Monitor successfully intercepted the OOM condition." << std::endl;
        }
        cudaFree(artificial_bloat);
    } else {
        std::cout << "[WARNING] Could not allocate artificial bloat (GPU may already be near full or driver blocked it)." << std::endl;
    }

    // ---------------------------------------------------------
    // 2. Test Zero-Copy Fallback Latency (5060 reading from RAM)
    // ---------------------------------------------------------
    size_t fallback_size = 256 * 1024 * 1024; // 256MB simulated KV cache chunk
    
    // Initialize both Ping-Pong AND Fallback buffer (prevents linker/extern issues)
    init_ring_buffer(1024 * 1024); // tiny 1MB dummy
    init_fallback_pool(fallback_size);
    
    int num_elements = fallback_size / sizeof(half);
    half* d_out;
    check_cuda(cudaMalloc(&d_out, fallback_size), "Alloc D_Out");
    
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    
    cudaSetDevice(0);
    zero_copy_kv_read_kernel<<<blocks, threads>>>((const half*)g_dev0_fallback_ptr, d_out, num_elements);
    check_cuda(cudaDeviceSynchronize(), "Warmup Fallback Sync");
    
    int iters = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iters; ++i) {
        zero_copy_kv_read_kernel<<<blocks, threads>>>((const half*)g_dev0_fallback_ptr, d_out, num_elements);
    }
    check_cuda(cudaDeviceSynchronize(), "Fallback Sync");
    auto end = std::chrono::high_resolution_clock::now();
    
    double bw = (fallback_size * iters) / (1024.0*1024.0*1024.0) / std::chrono::duration<double>(end - start).count();
    std::cout << "\n[Test] Zero-Copy KV Fallback Performance (5060 pulling from System RAM):" << std::endl;
    std::cout << "  -> " << bw << " GB/s" << std::endl;
    
    if (bw > 3.0) {
        std::cout << "[RESULT] Target Met: Fallback bandwidth is sufficient to prevent crashes without freezing the pipeline." << std::endl;
    } else {
        std::cout << "[WARNING] Fallback bandwidth is extremely low. Inference will stutter heavily during OOM conditions." << std::endl;
    }

    std::cout << "===========================================" << std::endl;
    return 0;
}
