/**
 * ROCm Kernel Playground - 08 Memory Access Patterns
 * Demonstrating efficient GPU memory access strategies
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

#define BLOCK_SIZE 256
#define ARRAY_SIZE (64 * 1024 * 1024)  // 64M elements

// ============================================
// 1. Coalesced vs Non-Coalesced Access
// ============================================

// Good: Coalesced access - consecutive threads access consecutive memory
__global__ void coalesced_read(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * 2.0f;
    }
}

// Bad: Strided access - threads access memory with gaps
__global__ void strided_read(const float* input, float* output, int n, int stride_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n / stride_factor; i += grid_stride) {
        int read_idx = i * stride_factor;
        output[i] = input[read_idx] * 2.0f;
    }
}

// ============================================
// 2. Structure of Arrays vs Array of Structures
// ============================================

// Array of Structures (AoS) - bad for GPU
struct ParticleAoS {
    float x, y, z;
    float vx, vy, vz;
    float mass;
    int id;
};

__global__ void update_particles_aos(ParticleAoS* particles, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    particles[idx].x += particles[idx].vx * dt;
    particles[idx].y += particles[idx].vy * dt;
    particles[idx].z += particles[idx].vz * dt;
}

// Structure of Arrays (SoA) - good for GPU
struct ParticlesSoA {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int* id;
};

__global__ void update_particles_soa(
    float* x, float* y, float* z,
    const float* vx, const float* vy, const float* vz,
    int n, float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    x[idx] += vx[idx] * dt;
    y[idx] += vy[idx] * dt;
    z[idx] += vz[idx] * dt;
}

// ============================================
// 3. Bank Conflicts in Shared Memory
// ============================================

#define SMEM_SIZE 1024

// Bad: Bank conflicts with stride of 32
__global__ void bank_conflict_kernel(float* output) {
    __shared__ float smem[SMEM_SIZE];
    
    int tid = threadIdx.x;
    
    // Initialize (no conflict)
    smem[tid] = tid;
    __syncthreads();
    
    // Bad access pattern - all threads hit same bank
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum += smem[tid * 32 % SMEM_SIZE];  // 32-way bank conflict
    }
    
    output[blockIdx.x * blockDim.x + tid] = sum;
}

// Good: No bank conflicts with padding
#define SMEM_PADDED_SIZE (SMEM_SIZE + 32)  // Add padding

__global__ void no_bank_conflict_kernel(float* output) {
    __shared__ float smem[SMEM_PADDED_SIZE];
    
    int tid = threadIdx.x;
    
    // Initialize with padding
    smem[tid + tid / 32] = tid;  // Padding to avoid conflicts
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        int idx = tid + tid / 32;  // Padded index
        sum += smem[idx];
    }
    
    output[blockIdx.x * blockDim.x + tid] = sum;
}

// ============================================
// 4. Memory Tiling for Matrix Operations
// ============================================

#define TILE_SIZE 16

__global__ void matrix_transpose_naive(
    const float* input,
    float* output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];  // Non-coalesced write!
    }
}

__global__ void matrix_transpose_tiled(
    const float* input,
    float* output,
    int width, int height
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Coalesced read into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Compute transposed indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Coalesced write from shared memory
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================
// 5. Vectorized Memory Access
// ============================================

__global__ void scalar_copy(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i];
    }
}

__global__ void vectorized_copy(const float4* input, float4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i];  // 4 floats at once
    }
}

// ============================================
// 6. Texture Memory / Cache Hints
// ============================================

__global__ void read_with_ldg(const float* __restrict__ input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        // __ldg uses texture cache for read-only data
        output[i] = __ldg(&input[i]) * 2.0f;
    }
}

int main() {
    std::cout << "=== ROCm Memory Access Patterns ===\n\n";
    
    const int N = ARRAY_SIZE;
    const int ITERATIONS = 50;
    
    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice);
    
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = min(grid_size, 65535);
    
    // =================================
    // Test 1: Coalesced vs Strided
    // =================================
    std::cout << "1. Coalesced vs Strided Access:\n";
    
    // Coalesced
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        coalesced_read<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float coalesced_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    // Strided (stride=32)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        strided_read<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N, 32);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float strided_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    float data_gb = (2.0f * N * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
    std::cout << "   Coalesced:  " << coalesced_time << " ms (" 
              << data_gb / (coalesced_time / 1000.0f) << " GB/s)\n";
    std::cout << "   Strided:    " << strided_time << " ms (" 
              << data_gb / (strided_time / 1000.0f) << " GB/s)\n";
    std::cout << "   Slowdown:   " << strided_time / coalesced_time << "x\n\n";
    
    // =================================
    // Test 2: Scalar vs Vectorized
    // =================================
    std::cout << "2. Scalar vs Vectorized Copy:\n";
    
    // Scalar
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        scalar_copy<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float scalar_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    // Vectorized (float4)
    int grid_vec = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_vec = min(grid_vec, 65535);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        vectorized_copy<<<grid_vec, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(d_input),
            reinterpret_cast<float4*>(d_output),
            N / 4);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float vec_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    std::cout << "   Scalar:     " << scalar_time << " ms (" 
              << data_gb / (scalar_time / 1000.0f) << " GB/s)\n";
    std::cout << "   Vectorized: " << vec_time << " ms (" 
              << data_gb / (vec_time / 1000.0f) << " GB/s)\n";
    std::cout << "   Speedup:    " << scalar_time / vec_time << "x\n\n";
    
    // =================================
    // Test 3: Matrix Transpose
    // =================================
    std::cout << "3. Matrix Transpose (Naive vs Tiled):\n";
    
    const int MAT_SIZE = 4096;
    float *d_mat_in, *d_mat_out;
    hipMalloc(&d_mat_in, MAT_SIZE * MAT_SIZE * sizeof(float));
    hipMalloc(&d_mat_out, MAT_SIZE * MAT_SIZE * sizeof(float));
    
    dim3 block_2d(TILE_SIZE, TILE_SIZE);
    dim3 grid_2d((MAT_SIZE + TILE_SIZE - 1) / TILE_SIZE, 
                 (MAT_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    // Naive
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_transpose_naive<<<grid_2d, block_2d>>>(d_mat_in, d_mat_out, MAT_SIZE, MAT_SIZE);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float naive_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    // Tiled
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_transpose_tiled<<<grid_2d, block_2d>>>(d_mat_in, d_mat_out, MAT_SIZE, MAT_SIZE);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float tiled_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    float mat_gb = (2.0f * MAT_SIZE * MAT_SIZE * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
    std::cout << "   Naive:  " << naive_time << " ms (" 
              << mat_gb / (naive_time / 1000.0f) << " GB/s)\n";
    std::cout << "   Tiled:  " << tiled_time << " ms (" 
              << mat_gb / (tiled_time / 1000.0f) << " GB/s)\n";
    std::cout << "   Speedup: " << naive_time / tiled_time << "x\n\n";
    
    // Summary
    std::cout << "=== Memory Access Best Practices ===\n";
    std::cout << "1. Always use coalesced memory access\n";
    std::cout << "2. Prefer Structure of Arrays (SoA) over Array of Structures (AoS)\n";
    std::cout << "3. Use shared memory tiling for data reuse\n";
    std::cout << "4. Add padding to avoid bank conflicts\n";
    std::cout << "5. Use vectorized loads (float4) when possible\n";
    std::cout << "6. Use __ldg() for read-only data\n";
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_mat_in);
    hipFree(d_mat_out);
    
    return 0;
}
