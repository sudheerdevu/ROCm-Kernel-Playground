/**
 * Parallel Reduction - Progressive optimization journey
 * 
 * This file demonstrates multiple reduction strategies from naive to optimized,
 * showing the impact of each optimization technique.
 */

#include <hip/hip_runtime.h>
#include "../../utils/hip_utils.h"
#include <vector>
#include <numeric>

constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Kernel 1: Naive Reduction
// Problem: Divergent warps, strided access, poor memory coalescing
// ============================================================================
__global__ void reduce_naive(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Naive reduction with strided access
    // Bad: Thread divergence within warps
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Kernel 2: Sequential Addressing
// Fix: Removes warp divergence by using contiguous thread mapping
// Still has: Bank conflicts in early iterations
// ============================================================================
__global__ void reduce_sequential(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - contiguous threads work together
    // Good: No warp divergence
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Kernel 3: First Add During Load
// Fix: Double the work per thread, halves number of blocks needed
// Improves arithmetic intensity
// ============================================================================
__global__ void reduce_first_add(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Each thread loads and adds TWO elements
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Kernel 4: Warp-Level Reduction with Shuffle
// Uses warp shuffle operations for final 32 elements
// Eliminates __syncthreads() within warp, faster synchronization
// ============================================================================
__device__ float warp_reduce(float val) {
    // Warp shuffle reduction - no shared memory needed for last 32 elements
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void reduce_warp_shuffle(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE / 32];  // One value per warp
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    
    // Load with first add
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    // Warp-level reduction using shuffles
    sum = warp_reduce(sum);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < blockDim.x / warpSize) {
        sum = sdata[tid];
        sum = warp_reduce(sum);
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// ============================================================================
// Kernel 5: Grid-Stride Loop + Atomic Final
// Handles arbitrary sizes, single kernel launch
// ============================================================================
__global__ void reduce_grid_stride(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE / 32];
    
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    
    // Grid-stride loop - each thread processes multiple elements
    float sum = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; 
         idx < n; 
         idx += blockDim.x * gridDim.x) {
        sum += input[idx];
    }
    
    // Warp reduction
    sum = warp_reduce(sum);
    
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid < blockDim.x / warpSize) {
        sum = sdata[tid];
        sum = warp_reduce(sum);
    }
    
    // Atomic add to global result
    if (tid == 0) {
        atomicAdd(output, sum);
    }
}

// ============================================================================
// Main: Run all variants and compare
// ============================================================================
int main() {
    print_device_info();
    
    const int N = 16 * 1024 * 1024;  // 16M elements
    
    std::cout << "Parallel Reduction Comparison\n";
    std::cout << "Array size: " << N << " elements (" << N * sizeof(float) / (1024*1024) << " MB)\n\n";
    
    // Allocate host memory
    std::vector<float> h_input(N);
    fill_random(h_input.data(), N, 0.0f, 1.0f);
    
    // CPU reference
    float cpu_sum = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    std::cout << "CPU sum: " << cpu_sum << "\n\n";
    
    // Allocate device memory
    DeviceBuffer<float> d_input(N, h_input.data());
    
    // Temporary output for block sums
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    DeviceBuffer<float> d_block_sums(num_blocks);
    DeviceBuffer<float> d_output(1);
    
    // ------------------------------------------------
    // Test each kernel
    // ------------------------------------------------
    
    auto test_reduction = [&](const char* name, auto kernel, int blocks_per_element = 1) {
        // Reset output
        float zero = 0.0f;
        HIP_CHECK(hipMemcpy(d_output.get(), &zero, sizeof(float), hipMemcpyHostToDevice));
        
        int grid_size = (N / blocks_per_element + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Benchmark
        auto launcher = [&]() {
            kernel<<<grid_size, BLOCK_SIZE>>>(d_input.get(), d_block_sums.get(), N);
        };
        
        float ms = benchmark_kernel(launcher);
        
        // For two-pass kernels, we'd need final reduction
        // For atomic version, result is directly in d_output
        
        std::cout << name << ":\n";
        std::cout << "  Time: " << ms << " ms\n";
        std::cout << "  Bandwidth: " << (N * sizeof(float)) / (ms * 1e6) << " GB/s\n\n";
    };
    
    test_reduction("1. Naive (strided)", reduce_naive);
    test_reduction("2. Sequential Addressing", reduce_sequential);
    test_reduction("3. First Add During Load", reduce_first_add, 2);
    test_reduction("4. Warp Shuffle", reduce_warp_shuffle, 2);
    
    // Grid-stride version (single pass with atomic)
    {
        float zero = 0.0f;
        HIP_CHECK(hipMemcpy(d_output.get(), &zero, sizeof(float), hipMemcpyHostToDevice));
        
        int grid_size = 256;  // Few blocks, many iterations per thread
        
        auto launcher = [&]() {
            HIP_CHECK(hipMemset(d_output.get(), 0, sizeof(float)));
            reduce_grid_stride<<<grid_size, BLOCK_SIZE>>>(d_input.get(), d_output.get(), N);
        };
        
        float ms = benchmark_kernel(launcher);
        
        float gpu_sum;
        d_output.copy_to_host(&gpu_sum);
        
        std::cout << "5. Grid-Stride + Atomic:\n";
        std::cout << "  Time: " << ms << " ms\n";
        std::cout << "  Bandwidth: " << (N * sizeof(float)) / (ms * 1e6) << " GB/s\n";
        std::cout << "  Result: " << gpu_sum << " (error: " << std::abs(gpu_sum - cpu_sum) << ")\n\n";
    }
    
    std::cout << "Key Takeaways:\n";
    std::cout << "  - Sequential addressing eliminates warp divergence\n";
    std::cout << "  - Doing work during load improves arithmetic intensity\n";
    std::cout << "  - Warp shuffles avoid shared memory for intra-warp ops\n";
    std::cout << "  - Grid-stride loops enable single-kernel arbitrary-size reduction\n";
    
    return 0;
}
