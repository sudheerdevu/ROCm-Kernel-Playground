/**
 * Matrix Multiplication - Optimization Journey
 * 
 * From naive to tiled to register-blocked implementations,
 * demonstrating memory hierarchy exploitation for GEMM.
 */

#include <hip/hip_runtime.h>
#include "../../utils/hip_utils.h"
#include <vector>

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_SIZE = TILE_SIZE;

// ============================================================================
// Kernel 1: Naive GEMM
// O(n³) global memory accesses - terrible memory efficiency
// ============================================================================
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 2: Tiled GEMM using Shared Memory (LDS)
// Loads tiles into shared memory, reducing global memory traffic by TILE_SIZE
// ============================================================================
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading of A tile
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Collaborative loading of B tile
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Wait for all threads to finish loading
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Wait before loading next tile
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 3: Bank-Conflict-Free Tiled GEMM
// Adds +1 padding to shared memory to avoid bank conflicts
// ============================================================================
__global__ void gemm_tiled_nobank(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // Padding prevents bank conflicts - 33 instead of 32 elements per row
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 4: Vectorized Load with float4
// Loads 4 elements at once for better memory bandwidth utilization
// ============================================================================
__global__ void gemm_vectorized(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    // This is a simplified version - real implementations use float4 loads
    // and register blocking for even better performance
    
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Manual unroll for ILP
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k += 4) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            sum += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
            sum += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
            sum += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    print_device_info();
    
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    
    std::cout << "Matrix Multiplication: (" << M << "x" << K << ") * (" 
              << K << "x" << N << ")\n\n";
    
    // Allocate host memory
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
    
    fill_random(h_A.data(), M * K, -1.0f, 1.0f);
    fill_random(h_B.data(), K * N, -1.0f, 1.0f);
    
    // CPU reference (only for small matrices)
    bool verify = (M <= 512 && K <= 512 && N <= 512);
    if (verify) {
        std::cout << "Computing CPU reference...\n";
        gemm_cpu(h_A.data(), h_B.data(), h_ref.data(), M, N, K);
    }
    
    // Allocate device memory
    DeviceBuffer<float> d_A(M * K, h_A.data());
    DeviceBuffer<float> d_B(K * N, h_B.data());
    DeviceBuffer<float> d_C(M * N);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Calculate theoretical metrics
    double ops = 2.0 * M * N * K;  // multiply-add = 2 ops
    double bytes_accessed = sizeof(float) * (M * K + K * N + M * N);
    
    auto benchmark_gemm = [&](const char* name, auto kernel) {
        // Clear output
        HIP_CHECK(hipMemset(d_C.get(), 0, M * N * sizeof(float)));
        
        auto launcher = [&]() {
            kernel<<<grid, block>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K);
        };
        
        float ms = benchmark_kernel(launcher);
        
        double gflops = ops / (ms * 1e6);
        double bandwidth = bytes_accessed / (ms * 1e6);
        
        std::cout << name << ":\n";
        std::cout << "  Time: " << ms << " ms\n";
        std::cout << "  GFLOPS: " << gflops << "\n";
        std::cout << "  Bandwidth: " << bandwidth << " GB/s (theoretical)\n";
        
        // Verify
        if (verify) {
            d_C.copy_to_host(h_C.data());
            if (verify_results(h_ref.data(), h_C.data(), M * N, 1e-3f)) {
                std::cout << "  Verification: PASSED\n";
            } else {
                std::cout << "  Verification: FAILED\n";
            }
        }
        std::cout << "\n";
    };
    
    benchmark_gemm("1. Naive GEMM", gemm_naive);
    benchmark_gemm("2. Tiled GEMM (LDS)", gemm_tiled);
    benchmark_gemm("3. Tiled (No Bank Conflict)", gemm_tiled_nobank);
    benchmark_gemm("4. Vectorized Unroll", gemm_vectorized);
    
    std::cout << "Optimization Summary:\n";
    std::cout << "1. Naive: Each thread does K global loads per element computed\n";
    std::cout << "2. Tiled: Reduces global loads by TILE_SIZE (" << TILE_SIZE << "x)\n";
    std::cout << "3. NoBank: Eliminates shared memory bank conflicts\n";
    std::cout << "4. Unroll: Increases ILP for better ALU utilization\n\n";
    
    std::cout << "For production GEMM, use rocBLAS which achieves >95% peak FLOPS!\n";
    
    return 0;
}
