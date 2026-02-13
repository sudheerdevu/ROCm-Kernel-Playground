/**
 * ROCm Kernel Playground - 06 Histogram
 * Parallel histogram computation with various optimization techniques
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>

#define NUM_BINS 256

// Naive histogram using global atomics
__global__ void histogram_naive(
    const unsigned char* input,
    unsigned int* histogram,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        atomicAdd(&histogram[input[i]], 1);
    }
}

// Shared memory histogram with privatization
__global__ void histogram_shared(
    const unsigned char* input,
    unsigned int* histogram,
    int size
) {
    __shared__ unsigned int s_hist[NUM_BINS];
    
    // Initialize shared histogram
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Compute local histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        atomicAdd(&s_hist[input[i]], 1);
    }
    __syncthreads();
    
    // Merge to global histogram
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Per-thread privatization with sequential merge
#define PRIV_SIZE 8  // Number of bins per thread for privatization

__global__ void histogram_privatized(
    const unsigned char* input,
    unsigned int* histogram,
    int size
) {
    // Per-thread private histogram (subset of bins)
    __shared__ unsigned int s_hist[NUM_BINS];
    
    // Initialize
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Process multiple elements per thread to amortize atomic overhead
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Local accumulator for consecutive same-value optimization
    unsigned char prev_val = 0;
    unsigned int local_count = 0;
    
    for (int i = idx; i < size; i += stride) {
        unsigned char val = input[i];
        if (val == prev_val && local_count > 0) {
            local_count++;
        } else {
            if (local_count > 0) {
                atomicAdd(&s_hist[prev_val], local_count);
            }
            prev_val = val;
            local_count = 1;
        }
    }
    // Flush remaining
    if (local_count > 0) {
        atomicAdd(&s_hist[prev_val], local_count);
    }
    
    __syncthreads();
    
    // Merge to global
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Warp-aggregated histogram using warp-level primitives
__global__ void histogram_warp_aggregated(
    const unsigned char* input,
    unsigned int* histogram,
    int size
) {
    __shared__ unsigned int s_hist[NUM_BINS];
    
    // Initialize
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        unsigned char val = input[i];
        
        // Warp-level aggregation
        unsigned int mask = __ballot(1);
        unsigned int leader_mask = 0;
        
        // Find threads with same bin value in warp
        for (int b = 0; b < 32; b++) {
            if ((mask >> b) & 1) {
                unsigned int same_val_mask = __ballot(val == __shfl(val, b));
                if (__popc(same_val_mask & ((1u << (threadIdx.x & 31)) - 1)) == 0) {
                    // First thread with this value in warp
                    if ((threadIdx.x & 31) == b) {
                        atomicAdd(&s_hist[val], __popc(same_val_mask));
                    }
                    leader_mask |= same_val_mask;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Merge to global
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Multi-pass histogram for very large inputs
__global__ void histogram_partial(
    const unsigned char* input,
    unsigned int* partial_hist,
    int size,
    int chunk_size
) {
    __shared__ unsigned int s_hist[NUM_BINS];
    
    // Initialize
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    int block_start = blockIdx.x * chunk_size;
    int block_end = min(block_start + chunk_size, size);
    
    for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        atomicAdd(&s_hist[input[i]], 1);
    }
    __syncthreads();
    
    // Write partial histogram
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        partial_hist[blockIdx.x * NUM_BINS + i] = s_hist[i];
    }
}

__global__ void histogram_merge(
    const unsigned int* partial_hist,
    unsigned int* histogram,
    int num_partials
) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= NUM_BINS) return;
    
    unsigned int sum = 0;
    for (int i = 0; i < num_partials; i++) {
        sum += partial_hist[i * NUM_BINS + bin];
    }
    histogram[bin] = sum;
}

// CPU reference implementation
void histogram_cpu(const unsigned char* input, unsigned int* histogram, int size) {
    memset(histogram, 0, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < size; i++) {
        histogram[input[i]]++;
    }
}

int main() {
    std::cout << "=== ROCm Histogram Kernels ===\n\n";
    
    const int SIZE = 64 * 1024 * 1024;  // 64MB image
    const int ITERATIONS = 50;
    
    // Allocate host memory
    std::vector<unsigned char> h_input(SIZE);
    std::vector<unsigned int> h_hist_cpu(NUM_BINS);
    std::vector<unsigned int> h_hist_gpu(NUM_BINS);
    
    // Initialize with random data (biased towards certain values for realistic test)
    for (int i = 0; i < SIZE; i++) {
        // Gaussian-ish distribution centered around 128
        float r = static_cast<float>(rand()) / RAND_MAX;
        r += static_cast<float>(rand()) / RAND_MAX;
        r += static_cast<float>(rand()) / RAND_MAX;
        r /= 3.0f;
        h_input[i] = static_cast<unsigned char>(r * 255);
    }
    
    // CPU reference
    histogram_cpu(h_input.data(), h_hist_cpu.data(), SIZE);
    
    // Allocate device memory
    unsigned char* d_input;
    unsigned int* d_histogram;
    hipMalloc(&d_input, SIZE);
    hipMalloc(&d_histogram, NUM_BINS * sizeof(unsigned int));
    
    hipMemcpy(d_input, h_input.data(), SIZE, hipMemcpyHostToDevice);
    
    // Launch configuration
    int block_size = 256;
    int grid_size = 1024;
    
    // Benchmark naive
    hipMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        hipMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));
        histogram_naive<<<grid_size, block_size>>>(d_input, d_histogram, SIZE);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float naive_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    hipMemcpy(h_hist_gpu.data(), d_histogram, NUM_BINS * sizeof(unsigned int), hipMemcpyDeviceToHost);
    bool naive_correct = (h_hist_gpu == h_hist_cpu);
    
    // Benchmark shared memory
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        hipMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));
        histogram_shared<<<grid_size, block_size>>>(d_input, d_histogram, SIZE);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float shared_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    hipMemcpy(h_hist_gpu.data(), d_histogram, NUM_BINS * sizeof(unsigned int), hipMemcpyDeviceToHost);
    bool shared_correct = (h_hist_gpu == h_hist_cpu);
    
    // Benchmark privatized
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        hipMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int));
        histogram_privatized<<<grid_size, block_size>>>(d_input, d_histogram, SIZE);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float priv_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    hipMemcpy(h_hist_gpu.data(), d_histogram, NUM_BINS * sizeof(unsigned int), hipMemcpyDeviceToHost);
    bool priv_correct = (h_hist_gpu == h_hist_cpu);
    
    // Print results
    float data_size_gb = SIZE / (1024.0f * 1024.0f * 1024.0f);
    
    std::cout << "Data size: " << SIZE / (1024 * 1024) << " MB\n";
    std::cout << "Number of bins: " << NUM_BINS << "\n\n";
    
    std::cout << "Performance Results:\n";
    std::cout << "  Naive (global atomics): " << naive_time << " ms";
    std::cout << " (" << data_size_gb / (naive_time / 1000.0f) << " GB/s)";
    std::cout << " [" << (naive_correct ? "PASS" : "FAIL") << "]\n";
    
    std::cout << "  Shared memory:          " << shared_time << " ms";
    std::cout << " (" << data_size_gb / (shared_time / 1000.0f) << " GB/s)";
    std::cout << " [" << (shared_correct ? "PASS" : "FAIL") << "]\n";
    
    std::cout << "  Privatized:             " << priv_time << " ms";
    std::cout << " (" << data_size_gb / (priv_time / 1000.0f) << " GB/s)";
    std::cout << " [" << (priv_correct ? "PASS" : "FAIL") << "]\n";
    
    std::cout << "\nSpeedups vs Naive:\n";
    std::cout << "  Shared memory: " << naive_time / shared_time << "x\n";
    std::cout << "  Privatized:    " << naive_time / priv_time << "x\n";
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_histogram);
    
    return 0;
}
