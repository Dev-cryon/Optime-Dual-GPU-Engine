#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "vram_monitor.cpp"

// Optimus GPU Layer Offload
// KV Cache Manager - Pre-allocates static blocks to prevent VRAM fragmentation

struct KVCacheBlock {
    int layer_id;
    void* d_keys;
    void* d_values;
    bool is_spilled_to_ram;
};

class KVManager {
private:
    std::vector<KVCacheBlock> blocks_5060;
    std::vector<KVCacheBlock> blocks_2060;
    
    size_t max_context_length;
    size_t hidden_size;
    int num_kv_heads;

public:
    KVManager(size_t context_len, size_t hidden, int kv_heads) 
        : max_context_length(context_len), hidden_size(hidden), num_kv_heads(kv_heads) {}

    // Pre-allocates the entire required KV cache block on the specified device
    void pre_allocate_cache(int start_layer, int end_layer, int device_id) {
        cudaSetDevice(device_id);
        
        // Example sizing for FP16 keys/values
        size_t block_bytes = max_context_length * hidden_size * sizeof(half);
        
        std::cout << "[KV Manager] Pre-allocating KV Cache for Layers " << start_layer 
                  << " to " << end_layer << " on Device " << device_id << std::endl;
        
        size_t total_alloc = 0;
        for (int i = start_layer; i <= end_layer; ++i) {
            KVCacheBlock block;
            block.layer_id = i;
            block.is_spilled_to_ram = false;
            
            cudaError_t err_k = cudaMalloc(&block.d_keys, block_bytes);
            cudaError_t err_v = cudaMalloc(&block.d_values, block_bytes);
            
            if (err_k != cudaSuccess || err_v != cudaSuccess) {
                std::cerr << "[Fatal] KV Manager OOM during Pre-allocation! Device " << device_id << std::endl;
                throw std::runtime_error("KV Cache Pre-allocation Failed.");
            }
            
            if (device_id == 0) blocks_5060.push_back(block);
            else blocks_2060.push_back(block);
            
            total_alloc += (block_bytes * 2);
        }
        
        std::cout << "[KV Manager] Successfully reserved " << (total_alloc / (1024*1024)) << " MB of contiguous VRAM." << std::endl;
    }

    void free_cache() {
        cudaSetDevice(0);
        for (auto& block : blocks_5060) {
            cudaFree(block.d_keys);
            cudaFree(block.d_values);
        }
        blocks_5060.clear();
        
        cudaSetDevice(1);
        for (auto& block : blocks_2060) {
            cudaFree(block.d_keys);
            cudaFree(block.d_values);
        }
        blocks_2060.clear();
    }
};
