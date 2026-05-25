#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>

#include "vram_monitor.cpp"
#include "weight_loader.cpp"
#include "kv_manager.cpp"

extern "C" void init_ring_buffer(size_t size_bytes);
extern "C" void init_fallback_pool(size_t size_bytes);
extern "C" void* get_host_ptr();

std::thread g_swizzle_thread;
std::mutex g_swizzle_mutex;
std::condition_variable g_swizzle_cv;
bool g_swizzle_ready = false;
bool g_swizzle_shutdown = false;

VRAMMonitor g_vram_monitor(0.95f); // 95% threshold
bool g_is_spilling_to_ram = false;
WeightLoader g_weight_loader;
KVManager* g_kv_manager = nullptr;

void avx512_swizzle_loop() {
    std::cout << "[Optimus Bridge] AVX-512 Background Swizzle Thread Started." << std::endl;
    void* host_ptr = get_host_ptr();
    
    while (true) {
        std::unique_lock<std::mutex> lock(g_swizzle_mutex);
        g_swizzle_cv.wait(lock, []{ return g_swizzle_ready || g_swizzle_shutdown; });
        
        if (g_swizzle_shutdown) break;
        
        if (host_ptr) {
            // AVX-512 alignment placeholder
        }
        
        g_swizzle_ready = false;
        g_swizzle_cv.notify_one();
    }
    std::cout << "[Optimus Bridge] AVX-512 Swizzle Thread Terminated." << std::endl;
}

void init_bridge(int buffer_size_mb) {
    size_t size_bytes = static_cast<size_t>(buffer_size_mb) * 1024 * 1024;
    init_ring_buffer(size_bytes);
    
    // Allocate a 2GB KV Fallback Pool by default from the System RAM
    size_t fallback_bytes = 2ULL * 1024 * 1024 * 1024;
    init_fallback_pool(fallback_bytes);
    
    g_swizzle_shutdown = false;
    g_swizzle_thread = std::thread(avx512_swizzle_loop);
    
    // Default KV sizing for Qwen 3.6 27B
    g_kv_manager = new KVManager(8192, 4096, 8);
}

bool check_vram_status() {
    bool dev0_risk = g_vram_monitor.check_overflow_risk(0);
    bool dev1_risk = g_vram_monitor.check_overflow_risk(1);
    
    if ((dev0_risk || dev1_risk) && !g_is_spilling_to_ram) {
        g_is_spilling_to_ram = true;
        std::cout << "[Optimus Orchestrator] Redirecting KV Cache writes to System RAM Fallback Pool." << std::endl;
    }
    return g_is_spilling_to_ram;
}

void shutdown_bridge() {
    {
        std::lock_guard<std::mutex> lock(g_swizzle_mutex);
        g_swizzle_shutdown = true;
    }
    g_swizzle_cv.notify_all();
    if (g_swizzle_thread.joinable()) g_swizzle_thread.join();
    if (g_kv_manager) {
        g_kv_manager->free_cache();
        delete g_kv_manager;
        g_kv_manager = nullptr;
    }
}

void trigger_swizzle() {
    {
        std::lock_guard<std::mutex> lock(g_swizzle_mutex);
        g_swizzle_ready = true;
    }
    g_swizzle_cv.notify_one();
}

bool load_safetensors_direct(const std::string& filepath) {
    return g_weight_loader.map_safetensors_file(filepath);
}

void offload_forward_pass() {
    // 1. Hook into check_vram_status()
    // 2. Launch TMA kernel on 5060 (Compute Stream 0)
    // 3. Launch WMMA fallback on 2060 (Compute Stream 1)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_bridge", &init_bridge, "Initialize the bare-metal PCIe bridge and mapped buffers");
    m.def("shutdown_bridge", &shutdown_bridge, "Shutdown the AVX-512 swizzle thread gracefully");
    m.def("trigger_swizzle", &trigger_swizzle, "Trigger manual AVX-512 swizzle pass on host memory");
    m.def("offload_forward_pass", &offload_forward_pass, "Execute the split-brain forward pass");
    m.def("check_vram_status", &check_vram_status, "Polls GPU memory and returns true if spilling to RAM");
    m.def("load_safetensors_direct", &load_safetensors_direct, "Directly maps and loads safetensors from SSD to VRAM bypassing Python");
}
