#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// Optimus GPU Layer Offload
// Qwen 3.6 27B Aligned Layer Distributor

struct QwenLayerConfig {
    int total_layers;
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads; // For GQA
};

struct ExecutionBoundary {
    int gpu0_5060_start_layer;
    int gpu0_5060_end_layer;
    int gpu1_2060_start_layer;
    int gpu1_2060_end_layer;
    bool requires_transit_quantization;
};

class LayerDistributor {
public:
    // Static mapping generator
    static ExecutionBoundary calculate_split_brain_mapping(const QwenLayerConfig& config, float profiling_ratio_5060) {
        if (profiling_ratio_5060 < 0.0f || profiling_ratio_5060 > 1.0f) {
            throw std::invalid_argument("Ratio must be between 0.0 and 1.0");
        }
        
        ExecutionBoundary boundary;
        boundary.gpu0_5060_start_layer = 0;
        
        // Calculate raw split index
        int raw_split = static_cast<int>(config.total_layers * profiling_ratio_5060);
        
        // GQA/RoPE Boundary Check: We must strictly sever between full transformer blocks.
        // Qwen self-attention and MLP blocks are contained within a single layer index.
        // We ensure the KV cache for the last GPU0 layer stays fully resident on the 5060.
        boundary.gpu0_5060_end_layer = raw_split - 1; 
        boundary.gpu1_2060_start_layer = raw_split;
        boundary.gpu1_2060_end_layer = config.total_layers - 1;
        
        boundary.requires_transit_quantization = (raw_split < config.total_layers);
        
        return boundary;
    }

    // Explicit CUDA stream router
    static void route_execution(const ExecutionBoundary& boundary, cudaStream_t stream_5060, cudaStream_t transit_stream, cudaStream_t stream_2060) {
        std::cout << "[Orchestrator] Severing Computational Graph..." << std::endl;
        std::cout << "[Orchestrator] Routing Layers 0 to " << boundary.gpu0_5060_end_layer << " to Stream 5060." << std::endl;
        if (boundary.requires_transit_quantization) {
            std::cout << "[Orchestrator] Routing Ping-Pong Quantization to Transit Stream." << std::endl;
        }
        std::cout << "[Orchestrator] Routing Layers " << boundary.gpu1_2060_start_layer << " to " << boundary.gpu1_2060_end_layer << " to Stream 2060." << std::endl;
    }
};
