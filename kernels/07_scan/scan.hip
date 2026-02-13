/**
 * ROCm Kernel Playground - 07 Parallel Scan (Prefix Sum)
 * Inclusive and exclusive scan implementations
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// Hillis-Steele scan (inclusive) - work efficient for small arrays
__global__ void scan_hillis_steele(float* data, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load to shared memory
    temp[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Hillis-Steele algorithm
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    
    // Write back
    if (idx < n) {
        data[idx] = temp[tid];
    }
}

// Blelloch scan (exclusive) - work efficient O(n) work
__global__ void scan_blelloch_upsweep(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2 + stride * 2 - 1;
    if (idx < n) {
        data[idx] += data[idx - stride];
    }
}

__global__ void scan_blelloch_downsweep(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2 + stride * 2 - 1;
    if (idx < n) {
        float temp = data[idx - stride];
        data[idx - stride] = data[idx];
        data[idx] += temp;
    }
}

// Optimized block-level scan with shared memory
__device__ void warp_scan_inclusive(float* data, int lane) {
    if (lane >= 1)  data[threadIdx.x] += data[threadIdx.x - 1];
    if (lane >= 2)  data[threadIdx.x] += data[threadIdx.x - 2];
    if (lane >= 4)  data[threadIdx.x] += data[threadIdx.x - 4];
    if (lane >= 8)  data[threadIdx.x] += data[threadIdx.x - 8];
    if (lane >= 16) data[threadIdx.x] += data[threadIdx.x - 16];
}

__global__ void scan_block_optimized(
    const float* input,
    float* output,
    float* block_sums,
    int n
) {
    __shared__ float s_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data
    s_data[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Warp-level scan
    int warp_id = tid / 32;
    int lane = tid % 32;
    
    // Scan within each warp
    for (int offset = 1; offset < 32; offset *= 2) {
        float val = __shfl_up(s_data[tid], offset);
        if (lane >= offset) {
            s_data[tid] += val;
        }
    }
    __syncthreads();
    
    // Get warp sums and scan them
    __shared__ float warp_sums[8];  // Max 8 warps per block (256 threads)
    
    if (lane == 31) {
        warp_sums[warp_id] = s_data[tid];
    }
    __syncthreads();
    
    // First warp scans the warp sums
    if (warp_id == 0 && lane < (BLOCK_SIZE / 32)) {
        float val = warp_sums[lane];
        for (int offset = 1; offset < (BLOCK_SIZE / 32); offset *= 2) {
            float temp = __shfl_up(val, offset);
            if (lane >= offset) {
                val += temp;
            }
        }
        warp_sums[lane] = val;
    }
    __syncthreads();
    
    // Add warp prefix to each element
    if (warp_id > 0) {
        s_data[tid] += warp_sums[warp_id - 1];
    }
    __syncthreads();
    
    // Write output
    if (idx < n) {
        output[idx] = s_data[tid];
    }
    
    // Store block sum for multi-block scan
    if (block_sums != nullptr && tid == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = s_data[tid];
    }
}

// Add block prefixes for multi-block scan
__global__ void add_block_prefix(
    float* output,
    const float* block_prefixes,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x > 0) {
        output[idx] += block_prefixes[blockIdx.x - 1];
    }
}

// Segmented scan - scan with segment boundaries
__global__ void segmented_scan(
    const float* input,
    const int* flags,  // 1 = start of new segment
    float* output,
    int n
) {
    __shared__ float s_data[BLOCK_SIZE];
    __shared__ int s_flags[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    s_data[tid] = (idx < n) ? input[idx] : 0.0f;
    s_flags[tid] = (idx < n) ? flags[idx] : 0;
    __syncthreads();
    
    // Segmented scan with flag propagation
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        int flag = s_flags[tid];
        
        if (tid >= stride && s_flags[tid] == 0) {
            val = s_data[tid - stride];
            // Propagate flag
            flag |= s_flags[tid - stride];
        }
        __syncthreads();
        
        s_data[tid] += val;
        s_flags[tid] = flag;
        __syncthreads();
    }
    
    if (idx < n) {
        output[idx] = s_data[tid];
    }
}

// Stream compaction using scan
__global__ void compute_predicates(
    const float* input,
    int* predicates,
    int n,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        predicates[idx] = (input[idx] > threshold) ? 1 : 0;
    }
}

__global__ void scatter(
    const float* input,
    const int* predicates,
    const int* scatter_indices,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && predicates[idx] == 1) {
        output[scatter_indices[idx] - 1] = input[idx];
    }
}

// CPU reference
void scan_cpu(const std::vector<float>& input, std::vector<float>& output) {
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); i++) {
        output[i] = output[i-1] + input[i];
    }
}

int main() {
    std::cout << "=== ROCm Parallel Scan Kernels ===\n\n";
    
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int ITERATIONS = 50;
    
    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output_cpu(N);
    std::vector<float> h_output_gpu(N);
    
    // Initialize with random values
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand() % 10) / 10.0f;
    }
    
    // CPU reference
    scan_cpu(h_input, h_output_cpu);
    
    // Allocate device memory
    float *d_input, *d_output, *d_block_sums, *d_block_prefixes;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_block_sums, num_blocks * sizeof(float));
    hipMalloc(&d_block_prefixes, num_blocks * sizeof(float));
    
    hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice);
    
    // Benchmark block-optimized scan (with block sums)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        // First pass: scan blocks
        scan_block_optimized<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, d_block_sums, N);
        
        // Second pass: scan block sums
        int num_blocks2 = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scan_block_optimized<<<num_blocks2, BLOCK_SIZE>>>(
            d_block_sums, d_block_prefixes, nullptr, num_blocks);
        
        // Third pass: add block prefixes
        add_block_prefix<<<num_blocks, BLOCK_SIZE>>>(
            d_output, d_block_prefixes, N);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float optimized_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    // Verify
    hipMemcpy(h_output_gpu.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabsf(h_output_gpu[i] - h_output_cpu[i]);
        max_error = fmaxf(max_error, error);
    }
    
    // Calculate bandwidth
    float data_gb = (2.0f * N * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);  // Read + write
    float bandwidth = data_gb / (optimized_time / 1000.0f);
    
    // Print results
    std::cout << "Array size: " << N / (1024 * 1024) << "M elements\n";
    std::cout << "Data size: " << (N * sizeof(float)) / (1024 * 1024) << " MB\n\n";
    
    std::cout << "Performance Results:\n";
    std::cout << "  Block-optimized scan: " << optimized_time << " ms\n";
    std::cout << "  Effective bandwidth:  " << bandwidth << " GB/s\n\n";
    
    std::cout << "Verification:\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  Status: " << (max_error < 1e-3f ? "PASS" : "FAIL") << "\n";
    
    // Quick demonstration of scan applications
    std::cout << "\n=== Scan Applications ===\n";
    std::cout << "- Prefix sum for parallel algorithms\n";
    std::cout << "- Stream compaction (filter)\n";
    std::cout << "- Radix sort (counting sort step)\n";
    std::cout << "- Polynomial evaluation\n";
    std::cout << "- Recurrence solving\n";
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_block_sums);
    hipFree(d_block_prefixes);
    
    return 0;
}
