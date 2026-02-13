/**
 * Vector Operations - Basic parallel patterns
 * 
 * Demonstrates:
 * - Element-wise operations
 * - Memory coalescing
 * - Grid-stride loops for arbitrary sizes
 */

#include <hip/hip_runtime.h>
#include "../../utils/hip_utils.h"
#include <vector>
#include <cmath>

// Basic vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Grid-stride loop version - handles any size with fixed grid
__global__ void vectorAddStride(const float* a, const float* b, float* c, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) {
        c[idx] = a[idx] + b[idx];
    }
}

// SAXPY: y = alpha * x + y
__global__ void saxpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// Element-wise multiply
__global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Fused multiply-add: d = a * b + c
__global__ void vectorFMA(const float* a, const float* b, const float* c, 
                          float* d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = __fmaf_rn(a[idx], b[idx], c[idx]);
    }
}

// Apply activation function (ReLU)
__global__ void vectorReLU(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

int main() {
    print_device_info();
    
    const int N = 10 * 1024 * 1024;  // 10M elements
    const int BLOCK_SIZE = 256;
    
    std::cout << "Vector Operations Demo\n";
    std::cout << "Array size: " << N << " elements (" 
              << N * sizeof(float) / (1024*1024) << " MB)\n\n";
    
    // Allocate host memory
    std::vector<float> h_a(N), h_b(N), h_c(N), h_ref(N);
    
    fill_random(h_a.data(), N, -1.0f, 1.0f);
    fill_random(h_b.data(), N, -1.0f, 1.0f);
    
    // CPU reference
    for (int i = 0; i < N; ++i) {
        h_ref[i] = h_a[i] + h_b[i];
    }
    
    // Allocate device memory
    DeviceBuffer<float> d_a(N, h_a.data());
    DeviceBuffer<float> d_b(N, h_b.data());
    DeviceBuffer<float> d_c(N);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Benchmark vector addition
    auto launcher = [&]() {
        vectorAdd<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), N);
    };
    float ms = benchmark_kernel(launcher);
    
    d_c.copy_to_host(h_c.data());
    bool passed = verify_results(h_ref.data(), h_c.data(), N);
    
    double bandwidth = (3.0 * N * sizeof(float)) / (ms * 1e6);  // 2 reads + 1 write
    
    std::cout << "Vector Addition:\n";
    std::cout << "  Time: " << ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "  Verification: " << (passed ? "PASSED" : "FAILED") << "\n\n";
    
    // Benchmark SAXPY
    float alpha = 2.0f;
    d_c.copy_from_host(h_b.data());  // Reset y
    
    auto saxpy_launcher = [&]() {
        saxpy<<<grid, block>>>(alpha, d_a.get(), d_c.get(), N);
    };
    ms = benchmark_kernel(saxpy_launcher);
    bandwidth = (3.0 * N * sizeof(float)) / (ms * 1e6);
    
    std::cout << "SAXPY (y = ax + y):\n";
    std::cout << "  Time: " << ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n\n";
    
    // Benchmark FMA
    DeviceBuffer<float> d_d(N);
    auto fma_launcher = [&]() {
        vectorFMA<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), d_d.get(), N);
    };
    ms = benchmark_kernel(fma_launcher);
    bandwidth = (4.0 * N * sizeof(float)) / (ms * 1e6);
    
    std::cout << "Fused Multiply-Add (d = a*b + c):\n";
    std::cout << "  Time: " << ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n\n";
    
    std::cout << "Key Takeaways:\n";
    std::cout << "  - Vector ops are memory-bound (low arithmetic intensity)\n";
    std::cout << "  - Bandwidth determines performance\n";
    std::cout << "  - Use grid-stride loops for flexibility\n";
    std::cout << "  - FMA instructions combine multiply+add efficiently\n";
    
    return 0;
}
