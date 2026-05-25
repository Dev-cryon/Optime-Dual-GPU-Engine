#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

// Optimus GPU Layer Offload
// Bare-Metal Weight Loader (Zero-Copy SSD -> VRAM)

// Minimal struct to represent a tensor found in the safetensors header
struct SafeTensorEntry {
    std::string name;
    size_t byte_offset_start;
    size_t byte_offset_end;
    int target_device_id;
};

class WeightLoader {
private:
    HANDLE hFile;
    HANDLE hMapping;
    void* pMappedView;
    size_t fileSize;
    
    // Safetensors header length is a 64-bit unsigned integer (8 bytes)
    uint64_t header_len = 0;

public:
    WeightLoader() : hFile(INVALID_HANDLE_VALUE), hMapping(NULL), pMappedView(nullptr), fileSize(0) {}

    ~WeightLoader() {
        close_mapping();
    }

    void close_mapping() {
        if (pMappedView) UnmapViewOfFile(pMappedView);
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
        pMappedView = nullptr;
        hMapping = NULL;
        hFile = INVALID_HANDLE_VALUE;
    }

    // Direct Windows API Memory Mapping
    bool map_safetensors_file(const std::string& filepath) {
        std::cout << "[Weight Loader] Initiating zero-copy file map: " << filepath << std::endl;
        
        hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "[Fatal] CreateFileA failed. Error: " << GetLastError() << std::endl;
            return false;
        }

        LARGE_INTEGER liSize;
        if (!GetFileSizeEx(hFile, &liSize)) {
            std::cerr << "[Fatal] GetFileSizeEx failed." << std::endl;
            return false;
        }
        fileSize = liSize.QuadPart;

        // Create mapping object
        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapping == NULL) {
            std::cerr << "[Fatal] CreateFileMappingA failed. Error: " << GetLastError() << std::endl;
            return false;
        }

        // Map the entire file into virtual memory
        pMappedView = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (pMappedView == nullptr) {
            std::cerr << "[Fatal] MapViewOfFile failed. Error: " << GetLastError() << std::endl;
            return false;
        }
        
        std::cout << "[Weight Loader] Successfully mapped " << (fileSize / (1024*1024)) << " MB into Host Virtual Memory." << std::endl;
        
        // Read Safetensors Header length (first 8 bytes)
        header_len = *reinterpret_cast<uint64_t*>(pMappedView);
        std::cout << "[Weight Loader] Detected Safetensors JSON Header Size: " << header_len << " bytes." << std::endl;
        
        return true;
    }

    // Streams a specific byte offset directly from mapped Host RAM to GPU VRAM using DMA
    void stream_to_vram(size_t byte_offset, size_t size_bytes, void* d_dest_ptr, int device_id, cudaStream_t stream) {
        if (!pMappedView) throw std::runtime_error("File not mapped!");
        
        cudaSetDevice(device_id);
        
        // Offset starts after the 8-byte length indicator AND the header itself
        size_t absolute_offset = 8 + header_len + byte_offset;
        
        if (absolute_offset + size_bytes > fileSize) {
            throw std::runtime_error("Out of bounds read on mapped file!");
        }

        void* host_src = static_cast<char*>(pMappedView) + absolute_offset;
        
        // Fire async copy. Since pMappedView is pageable by Windows, the CUDA driver will implicitly 
        // stage it. However, because it's a direct SSD memory map, the OS handles the page fault streaming 
        // directly from NVMe, bypassing standard Python buffer exhaustion.
        cudaError_t err = cudaMemcpyAsync(d_dest_ptr, host_src, size_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            std::cerr << "[Fatal] cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
};
