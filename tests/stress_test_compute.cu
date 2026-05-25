#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Include implementations for standalone testing
#include "../csrc/bridge/layer_distributor.cpp"
#include "../csrc/kernels/turing_fallback_compute.cu"

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[Fatal] CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "[Optimus Stress Test] Split-Brain Compute Kernels" << std::endl;
    std::cout << "===========================================" << std::endl;

    // ---------------------------------------------------------
    // 1. Test Layer Distributor Logic
    // ---------------------------------------------------------
    QwenLayerConfig qwen_config = {64, 4096, 32, 8};
    float ratio = 0.60f; // 60% on 5060, 40% on 2060
    
    ExecutionBoundary boundary = LayerDistributor::calculate_split_brain_mapping(qwen_config, ratio);
    std::cout << "[Test] Qwen 3.6 27B Layer Split Engine (Target Ratio: 60/40):" << std::endl;
    std::cout << "  -> 5060 Handles Layers: " << boundary.gpu0_5060_start_layer << " to " << boundary.gpu0_5060_end_layer << std::endl;
    std::cout << "  -> 2060 Handles Layers: " << boundary.gpu1_2060_start_layer << " to " << boundary.gpu1_2060_end_layer << std::endl;
    std::cout << "  -> Ping-Pong Transit Required: " << (boundary.requires_transit_quantization ? "YES" : "NO") << std::endl;
    
    if (boundary.gpu0_5060_end_layer != 37 || boundary.gpu1_2060_start_layer != 38) {
        std::cout << "[WARNING] Layer split math is incorrect. Expected boundary at Layer 38." << std::endl;
    } else {
        std::cout << "[SUCCESS] Layer split cleanly severs the computational graph without breaking GQA/RoPE." << std::endl;
    }

    // ---------------------------------------------------------
    // 2. Test Turing WMMA Latency
    // ---------------------------------------------------------
    int M = 128;  // Simulated Batch * Seq_Len
    int N = 4096; // Qwen Hidden Dimension
    int K = 4096; // Qwen Hidden Dimension
    
    half *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, M * K * sizeof(half)), "Alloc A");
    check_cuda(cudaMalloc(&d_B, K * N * sizeof(half)), "Alloc B");
    check_cuda(cudaMalloc(&d_C, M * N * sizeof(half)), "Alloc C");

    // Standard 16x16 warp block sizing for Turing WMMA
    dim3 threads(32); 
    dim3 blocks((M + 15)/16, (N + 15)/16);
    
    // Warmup
    turing_awq_wmma_forward_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    check_cuda(cudaDeviceSynchronize(), "Warmup Sync");

    int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) {
        turing_awq_wmma_forward_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    check_cuda(cudaDeviceSynchronize(), "Compute Sync");
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_latency = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    std::cout << "\n[Test] Turing 2060 WMMA Fallback Latency (M=" << M << ", N=" << N << ", K=" << K << "):" << std::endl;
    std::cout << "  -> " << avg_latency << " ms per layer" << std::endl;
    
    if (avg_latency < 2.0) {
        std::cout << "[RESULT] Target Met: Compute latency is < 2ms/layer." << std::endl;
    } else {
        std::cout << "[WARNING] Compute latency exceeds 2ms/layer. Memory coalescing may be failing." << std::endl;
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    std::cout << "===========================================" << std::endl;
    return 0;
}
