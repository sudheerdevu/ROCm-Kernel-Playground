/**
 * HIP Kernel Utilities
 * 
 * Common utilities and helper functions for HIP kernel development.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

// ============ Error Checking Macros ============

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define HIP_KERNEL_CHECK() \
    do { \
        hipError_t err = hipGetLastError(); \
        if (err != hipSuccess) { \
            std::cerr << "Kernel Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// ============ Timer Class ============

class HipTimer {
public:
    HipTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }
    
    ~HipTimer() {
        hipEventDestroy(start_);
        hipEventDestroy(stop_);
    }
    
    void start() {
        HIP_CHECK(hipEventRecord(start_, 0));
    }
    
    void stop() {
        HIP_CHECK(hipEventRecord(stop_, 0));
        HIP_CHECK(hipEventSynchronize(stop_));
    }
    
    float elapsed_ms() {
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
private:
    hipEvent_t start_, stop_;
};


// ============ Device Memory RAII ============

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t count) : count_(count), data_(nullptr) {
        HIP_CHECK(hipMalloc(&data_, count * sizeof(T)));
    }
    
    ~DeviceBuffer() {
        if (data_) {
            hipFree(data_);
        }
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Enable move
    DeviceBuffer(DeviceBuffer&& other) : count_(other.count_), data_(other.data_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    
    void copyFromHost(const T* host_data) {
        HIP_CHECK(hipMemcpy(data_, host_data, count_ * sizeof(T), hipMemcpyHostToDevice));
    }
    
    void copyToHost(T* host_data) const {
        HIP_CHECK(hipMemcpy(host_data, data_, count_ * sizeof(T), hipMemcpyDeviceToHost));
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
private:
    size_t count_;
    T* data_;
};


// ============ Kernel Launch Helpers ============

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    hipStream_t stream;
    
    LaunchConfig(int grid_x, int block_x, size_t smem = 0, hipStream_t s = 0)
        : grid(grid_x), block(block_x), shared_mem(smem), stream(s) {}
    
    LaunchConfig(dim3 g, dim3 b, size_t smem = 0, hipStream_t s = 0)
        : grid(g), block(b), shared_mem(smem), stream(s) {}
};

inline LaunchConfig make_1d_config(int n, int block_size = 256) {
    int grid_size = (n + block_size - 1) / block_size;
    return LaunchConfig(grid_size, block_size);
}

inline LaunchConfig make_2d_config(int width, int height, int block_x = 16, int block_y = 16) {
    int grid_x = (width + block_x - 1) / block_x;
    int grid_y = (height + block_y - 1) / block_y;
    return LaunchConfig(dim3(grid_x, grid_y), dim3(block_x, block_y));
}


// ============ Validation Utilities ============

template<typename T>
bool validate_results(const T* expected, const T* actual, size_t count, T epsilon = 1e-5) {
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(expected[i] - actual[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
double compute_bandwidth_gbps(size_t bytes, double ms) {
    return (bytes / 1e9) / (ms / 1e3);
}

template<typename T>
double compute_gflops(size_t flops, double ms) {
    return (flops / 1e9) / (ms / 1e3);
}


// ============ Device Info ============

inline void print_device_info(int device_id = 0) {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Compute units: " << props.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Max shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Total global memory: " << props.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Warp size: " << props.warpSize << std::endl;
}


// ============ Benchmark Helpers ============

template<typename KernelFunc>
double benchmark_kernel(KernelFunc&& kernel_launch, int warmup = 5, int iterations = 100) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel_launch();
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    HipTimer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        kernel_launch();
    }
    timer.stop();
    
    return timer.elapsed_ms() / iterations;
}
