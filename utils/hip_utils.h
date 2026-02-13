/**
 * hip_utils.h - Common HIP utilities for kernel playground
 */

#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// Error checking macro
#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            std::cerr << "HIP Error: " << hipGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel launch macro with error checking
#define LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...)            \
    do {                                                                   \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);              \
        HIP_CHECK(hipGetLastError());                                      \
    } while (0)

/**
 * Simple GPU timer using HIP events
 */
class GpuTimer {
public:
    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }
    
    ~GpuTimer() {
        hipEventDestroy(start_);
        hipEventDestroy(stop_);
    }
    
    void start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start_, stream));
    }
    
    void stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop_, stream));
    }
    
    float elapsed_ms() {
        HIP_CHECK(hipEventSynchronize(stop_));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
private:
    hipEvent_t start_, stop_;
};

/**
 * Device information printer
 */
inline void print_device_info() {
    int device;
    HIP_CHECK(hipGetDevice(&device));
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    std::cout << "====== Device Info ======\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Compute units: " << props.multiProcessorCount << "\n";
    std::cout << "Clock: " << props.clockRate / 1000 << " MHz\n";
    std::cout << "Global memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Shared memory/block: " << props.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Max threads/block: " << props.maxThreadsPerBlock << "\n";
    std::cout << "Warp size: " << props.warpSize << "\n";
    std::cout << "GCN Arch: " << props.gcnArchName << "\n";
    std::cout << "========================\n\n";
}

/**
 * Allocate device memory with initialization
 */
template<typename T>
T* allocate_device(size_t count, const T* host_data = nullptr) {
    T* d_ptr;
    HIP_CHECK(hipMalloc(&d_ptr, count * sizeof(T)));
    
    if (host_data) {
        HIP_CHECK(hipMemcpy(d_ptr, host_data, count * sizeof(T), hipMemcpyHostToDevice));
    }
    
    return d_ptr;
}

/**
 * RAII wrapper for device memory
 */
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t count) : count_(count), ptr_(nullptr) {
        HIP_CHECK(hipMalloc(&ptr_, count * sizeof(T)));
    }
    
    DeviceBuffer(size_t count, const T* host_data) : DeviceBuffer(count) {
        HIP_CHECK(hipMemcpy(ptr_, host_data, count * sizeof(T), hipMemcpyHostToDevice));
    }
    
    ~DeviceBuffer() {
        if (ptr_) hipFree(ptr_);
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    
    void copy_to_host(T* dst) const {
        HIP_CHECK(hipMemcpy(dst, ptr_, count_ * sizeof(T), hipMemcpyDeviceToHost));
    }
    
    void copy_from_host(const T* src) {
        HIP_CHECK(hipMemcpy(ptr_, src, count_ * sizeof(T), hipMemcpyHostToDevice));
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Enable move
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : count_(other.count_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
private:
    size_t count_;
    T* ptr_;
};

/**
 * Verify results with tolerance
 */
template<typename T>
bool verify_results(const T* expected, const T* actual, size_t count, 
                    T tolerance = static_cast<T>(1e-5)) {
    for (size_t i = 0; i < count; ++i) {
        T diff = std::abs(expected[i] - actual[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << actual[i] 
                      << " (diff = " << diff << ")\n";
            return false;
        }
    }
    return true;
}

/**
 * Fill array with random values
 */
template<typename T>
void fill_random(T* arr, size_t count, T min_val = 0, T max_val = 1) {
    for (size_t i = 0; i < count; ++i) {
        T r = static_cast<T>(rand()) / RAND_MAX;
        arr[i] = min_val + r * (max_val - min_val);
    }
}

/**
 * Calculate optimal grid dimensions
 */
inline dim3 calc_grid_1d(size_t total, size_t block_size) {
    return dim3((total + block_size - 1) / block_size);
}

inline dim3 calc_grid_2d(size_t width, size_t height, dim3 block) {
    return dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
}

/**
 * Benchmark wrapper
 */
template<typename Func>
float benchmark_kernel(Func kernel_launcher, int warmup_iters = 5, int bench_iters = 20) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        kernel_launcher();
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < bench_iters; ++i) {
        kernel_launcher();
    }
    timer.stop();
    
    return timer.elapsed_ms() / bench_iters;
}

#endif // HIP_UTILS_H
