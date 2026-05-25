#include <cuda_runtime.h>
#include <iostream>

// Optimus GPU Layer Offload
// Pinned Memory Transit Layer (WDDM Bypass) - Asymmetric Ping-Pong Architecture & OOM Fallback

extern "C" {

void* g_host_ptr_5060_to_2060 = nullptr;
void* g_host_ptr_2060_to_5060 = nullptr;

void* g_dev0_ptr_5060_to_2060 = nullptr;
void* g_dev0_ptr_2060_to_5060 = nullptr;

void* g_dev1_ptr_5060_to_2060 = nullptr;
void* g_dev1_ptr_2060_to_5060 = nullptr;

size_t g_buffer_size = 0;

// OOM Fallback Pool (System RAM)
void* g_host_fallback_pool = nullptr;
void* g_dev0_fallback_ptr = nullptr;
void* g_dev1_fallback_ptr = nullptr;
size_t g_fallback_size = 0;

void init_ring_buffer(size_t size_bytes) {
    // Option B: The Ping-Pong Buffer (Asymmetric Driver Fix)
    cudaError_t err = cudaHostAlloc(&g_host_ptr_5060_to_2060, size_bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] Failed to alloc 5060->2060 buffer: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    err = cudaHostAlloc(&g_host_ptr_2060_to_5060, size_bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] Failed to alloc 2060->5060 buffer: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    cudaSetDevice(0);
    cudaHostGetDevicePointer(&g_dev0_ptr_5060_to_2060, g_host_ptr_5060_to_2060, 0);
    cudaHostGetDevicePointer(&g_dev0_ptr_2060_to_5060, g_host_ptr_2060_to_5060, 0);

    cudaSetDevice(1);
    cudaHostGetDevicePointer(&g_dev1_ptr_5060_to_2060, g_host_ptr_5060_to_2060, 0);
    cudaHostGetDevicePointer(&g_dev1_ptr_2060_to_5060, g_host_ptr_2060_to_5060, 0);
    
    g_buffer_size = size_bytes;
    std::cout << "[Optimus PCIe Bridge] Initialized Asymmetric Ping-Pong Buffers (" << (size_bytes * 2 / (1024 * 1024)) << "MB Total)." << std::endl;
}

void init_fallback_pool(size_t size_bytes) {
    // Allocate heavily cached standard pinned memory for KV cache storage
    // We do NOT use WriteCombined here because both GPUs need to randomly access the KV cache over PCIe.
    cudaError_t err = cudaHostAlloc(&g_host_fallback_pool, size_bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] Failed to alloc System RAM Fallback Pool: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    cudaSetDevice(0);
    cudaHostGetDevicePointer(&g_dev0_fallback_ptr, g_host_fallback_pool, 0);
    
    cudaSetDevice(1);
    cudaHostGetDevicePointer(&g_dev1_fallback_ptr, g_host_fallback_pool, 0);
    
    g_fallback_size = size_bytes;
    std::cout << "[Optimus VRAM Monitor] Allocated " << (size_bytes / (1024 * 1024)) << "MB of Pinned System RAM for Zero-Copy KV Fallback." << std::endl;
}

void* get_host_ptr() { return g_host_ptr_5060_to_2060; }

} // extern "C"
