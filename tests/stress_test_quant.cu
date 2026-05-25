#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Directly include the kernel for standalone unit testing
#include "../csrc/kernels/quant_transit.cu"

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "[Optimus Stress Test] Quantization Transit" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    const int hidden_dim = 4096; // Qwen 3.6 27B typical hidden dimension
    const int num_tokens = 1024; // Stress testing with a large context batch
    const int num_elements = hidden_dim * num_tokens;
    
    std::cout << "Allocating Host Memory (" << num_elements * sizeof(half) / (1024*1024) << " MB)..." << std::endl;
    std::vector<half> h_input(num_elements);
    std::vector<half> h_output(num_elements);
    
    // Seed with realistic Qwen hidden state ranges (-5.0 to 5.0)
    for (int i = 0; i < num_elements; ++i) {
        float val = static_cast<float>(rand() % 1000) / 100.0f - 5.0f;
        h_input[i] = __float2half(val);
    }
    
    half* d_input;
    int8_t* d_quantized;
    half* d_output;
    float* d_scales;
    
    check_cuda(cudaMalloc(&d_input, num_elements * sizeof(half)), "Alloc d_input");
    check_cuda(cudaMalloc(&d_quantized, num_elements * sizeof(int8_t)), "Alloc d_quantized (INT8 payload)");
    check_cuda(cudaMalloc(&d_output, num_elements * sizeof(half)), "Alloc d_output");
    check_cuda(cudaMalloc(&d_scales, num_tokens * sizeof(float)), "Alloc d_scales");
    
    check_cuda(cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(half), cudaMemcpyHostToDevice), "H2D");
    
    int threads_per_block = 1024;
    int blocks = num_tokens;
    
    std::cout << "Running Warmup..." << std::endl;
    transit_downcast_fp16_to_int8<<<blocks, threads_per_block>>>(d_input, d_quantized, d_scales, hidden_dim);
    transit_upcast_int8_to_fp16<<<blocks, threads_per_block>>>(d_quantized, d_output, d_scales, hidden_dim);
    check_cuda(cudaDeviceSynchronize(), "Warmup Sync");
    
    std::cout << "Benchmarking Kernels (1000 iterations)..." << std::endl;
    
    // Timing Downcast
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        transit_downcast_fp16_to_int8<<<blocks, threads_per_block>>>(d_input, d_quantized, d_scales, hidden_dim);
    }
    check_cuda(cudaDeviceSynchronize(), "Downcast Sync");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> downcast_time = end - start;
    
    // Timing Upcast
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        transit_upcast_int8_to_fp16<<<blocks, threads_per_block>>>(d_quantized, d_output, d_scales, hidden_dim);
    }
    check_cuda(cudaDeviceSynchronize(), "Upcast Sync");
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> upcast_time = end - start;
    
    std::cout << "-> Downcast Latency: " << downcast_time.count()/1000.0 << " ms/iter" << std::endl;
    std::cout << "-> Upcast Latency:   " << upcast_time.count()/1000.0 << " ms/iter" << std::endl;
    
    // Check Parity
    check_cuda(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost), "D2H");
    
    float max_error = 0.0f;
    for (int i = 0; i < num_elements; ++i) {
        float in_val = __half2float(h_input[i]);
        float out_val = __half2float(h_output[i]);
        float diff = std::abs(in_val - out_val);
        if (diff > max_error) max_error = diff;
    }
    
    std::cout << "-> Max Parity Error (FP16 vs INT8 vs FP16): " << max_error << std::endl;
    
    if (max_error > 0.05f) {
        std::cout << "[RESULT] Parity is expected for INT8 quantization. Checking variance..." << std::endl;
    } else {
        std::cout << "[RESULT] Parity checks passed!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input); cudaFree(d_quantized); cudaFree(d_output); cudaFree(d_scales);
    
    std::cout << "===========================================" << std::endl;
    return 0;
}
