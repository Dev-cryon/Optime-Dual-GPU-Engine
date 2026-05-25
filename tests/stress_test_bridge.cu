#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Expose the ring buffer logic
extern "C" {
    void init_ring_buffer(size_t size_bytes);
    
    extern void* g_dev0_ptr_5060_to_2060;
    extern void* g_dev0_ptr_2060_to_5060;
    
    extern void* g_dev1_ptr_5060_to_2060;
    extern void* g_dev1_ptr_2060_to_5060;
}

// Directly include the implementation for unit testing
#include "../csrc/bridge/ring_buffer.cu"

__global__ void dummy_write_kernel(int* ptr, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        ptr[idx] = idx;
    }
}

__global__ void dummy_read_kernel(const int* ptr, int* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = ptr[idx];
    }
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "[Optimus Stress Test] Ping-Pong Bridge Bandwidth" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices < 2) {
        std::cout << "[Warning] Less than 2 GPUs detected. Cannot test asymmetric bandwidth." << std::endl;
        return 0;
    }

    size_t size_bytes = 256 * 1024 * 1024; // 256 MB test payload
    int num_elements = size_bytes / sizeof(int);
    
    init_ring_buffer(size_bytes);
    
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    int iters = 20;
    
    // --- PATH A: 5060 -> 2060 ---
    cudaSetDevice(0);
    dummy_write_kernel<<<blocks, threads>>>((int*)g_dev0_ptr_5060_to_2060, num_elements);
    check_cuda(cudaDeviceSynchronize(), "Warmup 5060 Write");
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) dummy_write_kernel<<<blocks, threads>>>((int*)g_dev0_ptr_5060_to_2060, num_elements);
    check_cuda(cudaDeviceSynchronize(), "5060 Write Sync");
    auto end = std::chrono::high_resolution_clock::now();
    double bw_5060_write = (size_bytes * iters) / (1024.0*1024.0*1024.0) / std::chrono::duration<double>(end - start).count();

    cudaSetDevice(1);
    int* d1_out; cudaMalloc(&d1_out, size_bytes);
    dummy_read_kernel<<<blocks, threads>>>((const int*)g_dev1_ptr_5060_to_2060, d1_out, num_elements);
    check_cuda(cudaDeviceSynchronize(), "Warmup 2060 Read");
    
    start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) dummy_read_kernel<<<blocks, threads>>>((const int*)g_dev1_ptr_5060_to_2060, d1_out, num_elements);
    check_cuda(cudaDeviceSynchronize(), "2060 Read Sync");
    end = std::chrono::high_resolution_clock::now();
    double bw_2060_read = (size_bytes * iters) / (1024.0*1024.0*1024.0) / std::chrono::duration<double>(end - start).count();

    // --- PATH B: 2060 -> 5060 ---
    cudaSetDevice(1);
    dummy_write_kernel<<<blocks, threads>>>((int*)g_dev1_ptr_2060_to_5060, num_elements);
    check_cuda(cudaDeviceSynchronize(), "Warmup 2060 Write");
    
    start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) dummy_write_kernel<<<blocks, threads>>>((int*)g_dev1_ptr_2060_to_5060, num_elements);
    check_cuda(cudaDeviceSynchronize(), "2060 Write Sync");
    end = std::chrono::high_resolution_clock::now();
    double bw_2060_write = (size_bytes * iters) / (1024.0*1024.0*1024.0) / std::chrono::duration<double>(end - start).count();

    cudaSetDevice(0);
    int* d0_out; cudaMalloc(&d0_out, size_bytes);
    dummy_read_kernel<<<blocks, threads>>>((const int*)g_dev0_ptr_2060_to_5060, d0_out, num_elements);
    check_cuda(cudaDeviceSynchronize(), "Warmup 5060 Read");
    
    start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) dummy_read_kernel<<<blocks, threads>>>((const int*)g_dev0_ptr_2060_to_5060, d0_out, num_elements);
    check_cuda(cudaDeviceSynchronize(), "5060 Read Sync");
    end = std::chrono::high_resolution_clock::now();
    double bw_5060_read = (size_bytes * iters) / (1024.0*1024.0*1024.0) / std::chrono::duration<double>(end - start).count();

    std::cout << "[Path A] 5060 -> 2060 (No WriteCombined)" << std::endl;
    std::cout << "  5060 Write: " << bw_5060_write << " GB/s" << std::endl;
    std::cout << "  2060 Read:  " << bw_2060_read << " GB/s" << std::endl;
    
    std::cout << "[Path B] 2060 -> 5060 (With WriteCombined)" << std::endl;
    std::cout << "  2060 Write: " << bw_2060_write << " GB/s" << std::endl;
    std::cout << "  5060 Read:  " << bw_5060_read << " GB/s" << std::endl;
    
    if (bw_2060_read > 5.0 && bw_2060_write > 5.0) {
        std::cout << "\n[RESULT] Asymmetric Buffer stabilized the bandwidth! Test Passed." << std::endl;
    } else {
        std::cout << "\n[WARNING] Still detecting bottlenecks on the 2060 side." << std::endl;
    }
    
    return 0;
}
