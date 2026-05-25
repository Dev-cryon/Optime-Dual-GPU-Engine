#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimus GPU Layer Offload
// AWQ-optimized INT8 Downcast & Upcast kernels for PCIe transit

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) 
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// Downcast Kernel (Runs on 5060)
// Compresses FP16 hidden states down to INT8 using per-token max scaling
// This immediately slashes the PCIe bandwidth requirement by 50%
__global__ void transit_downcast_fp16_to_int8(const half* input, int8_t* output, float* scales, int hidden_dim) {
    int token_idx = blockIdx.x; 
    int tid = threadIdx.x;
    
    // Step 1: Find absolute max for the token
    float local_max = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(input[token_idx * hidden_dim + i]);
        local_max = fmaxf(local_max, fabsf(val));
    }
    
    float block_max = blockReduceMax(local_max);
    
    __shared__ float s_scale;
    if (tid == 0) {
        s_scale = fmaxf(block_max / 127.0f, 1e-7f); // Avoid division by zero
        scales[token_idx] = s_scale;
    }
    __syncthreads();
    
    // Step 2: Quantize and pack
    float scale_inv = 1.0f / s_scale;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(input[token_idx * hidden_dim + i]);
        output[token_idx * hidden_dim + i] = static_cast<int8_t>(roundf(val * scale_inv));
    }
}

// Upcast Kernel (Runs on 2060)
// Decompresses the INT8 payload back to FP16 to feed the Turing WMMA fallback loop
__global__ void transit_upcast_int8_to_fp16(const int8_t* input, half* output, const float* scales, int hidden_dim) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float scale = scales[token_idx];
    
    // High-performance vectorized memory operations could be used here,
    // but a standard strided loop is sufficient for our WDDM pipeline tests.
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = static_cast<float>(input[token_idx * hidden_dim + i]) * scale;
        output[token_idx * hidden_dim + i] = __float2half(val);
    }
}
