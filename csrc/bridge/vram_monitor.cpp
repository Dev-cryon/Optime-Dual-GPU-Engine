#include <cuda_runtime.h>
#include <iostream>

// Optimus GPU Layer Offload
// VRAM Monitor: Prevents CUDA OOM by explicitly polling memory states

class VRAMMonitor {
private:
    float threshold_ratio;
    
public:
    VRAMMonitor(float ratio = 0.98f) : threshold_ratio(ratio) {}

    // Polls the given device and returns true if an OOM threshold is crossed
    bool check_overflow_risk(int device_id) {
        cudaSetDevice(device_id);
        size_t free_byte;
        size_t total_byte;
        cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
        
        if (err != cudaSuccess) {
            std::cerr << "[VRAM Monitor] Failed to poll device " << device_id << std::endl;
            return true; // Fail safe - assume OOM risk if we can't read it
        }

        size_t used_byte = total_byte - free_byte;
        float current_ratio = static_cast<float>(used_byte) / static_cast<float>(total_byte);
        
        if (current_ratio >= threshold_ratio) {
            std::cout << "\n[VRAM Monitor] CRITICAL: Device " << device_id << " at " << (current_ratio * 100) 
                      << "% capacity. Triggering System RAM Spillover!" << std::endl;
            return true; // Risk of OOM is high
        }
        return false;
    }
};
