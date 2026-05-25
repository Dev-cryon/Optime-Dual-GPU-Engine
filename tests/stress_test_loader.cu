#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <windows.h>
#include "../csrc/bridge/weight_loader.cpp"

void create_dummy_safetensors(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to create dummy file." << std::endl;
        exit(1);
    }
    
    // Write 8-byte header length (e.g., 64 bytes)
    uint64_t header_len = 64;
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    
    // Write dummy JSON string of 64 bytes
    std::string dummy_json = "{\"__metadata__\":{\"format\":\"pt\"},\"dummy_tensor\":{\"data_offsets\":}}";
    // pad to exactly 64 bytes
    dummy_json.resize(64, ' ');
    out.write(dummy_json.c_str(), dummy_json.size());
    
    // Write 1GB of dummy binary tensor data
    size_t tensor_size = 1024 * 1024 * 1024; // 1 GB
    std::vector<char> zeros(1024 * 1024, 0); // 1 MB chunk
    for(int i = 0; i < 1024; ++i) {
        out.write(zeros.data(), zeros.size());
    }
    out.close();
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "[Optimus Stress Test] Bare-Metal Weight Loader" << std::endl;
    std::cout << "===========================================" << std::endl;

    std::string dummy_path = "dummy.safetensors";
    
    std::cout << "[Test] Generating 1GB dummy safetensors file... (This may take a moment)" << std::endl;
    create_dummy_safetensors(dummy_path);
    std::cout << "[Test] Dummy file generated." << std::endl;

    WeightLoader loader;
    
    auto start = std::chrono::high_resolution_clock::now();
    bool success = loader.map_safetensors_file(dummy_path);
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "\n[Test] Zero-Copy File Mapping Latency: " << elapsed << " ms" << std::endl;
    
    if (success && elapsed < 50.0) {
        std::cout << "[RESULT] Target Met: OS-level Memory Mapping successfully hooked SSD bypass in under 50ms!" << std::endl;
    } else if (success) {
        std::cout << "[WARNING] Mapping succeeded but was abnormally slow. SSD performance issue?" << std::endl;
    } else {
        std::cout << "[FATAL] Memory mapping failed." << std::endl;
    }

    loader.close_mapping();
    
    // Clean up
    DeleteFileA(dummy_path.c_str());
    
    std::cout << "===========================================" << std::endl;
    return 0;
}
