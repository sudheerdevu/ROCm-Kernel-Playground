/**
 * ROCm Kernel Playground - 05 Convolution
 * 2D Convolution implementations with various optimizations
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// Naive 2D convolution
__global__ void conv2d_naive(
    const float* input,
    const float* kernel,
    float* output,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int output_h, int output_w
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= output_w || out_y >= output_h) return;
    
    float sum = 0.0f;
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            sum += input[in_y * input_w + in_x] * kernel[ky * kernel_w + kx];
        }
    }
    output[out_y * output_w + out_x] = sum;
}

// Shared memory convolution - tiles input and kernel
#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define SHARED_SIZE (TILE_SIZE + 2 * KERNEL_RADIUS)

__global__ void conv2d_shared(
    const float* input,
    const float* kernel,
    float* output,
    int input_h, int input_w,
    int kernel_size,
    int output_h, int output_w
) {
    __shared__ float s_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float s_kernel[5][5];  // 5x5 kernel max
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // Load kernel into shared memory (first threads only)
    if (tx < kernel_size && ty < kernel_size) {
        s_kernel[ty][tx] = kernel[ty * kernel_size + tx];
    }
    
    // Load input tile with halo
    int in_x = out_x - KERNEL_RADIUS;
    int in_y = out_y - KERNEL_RADIUS;
    
    // Each thread loads multiple elements for the halo
    for (int dy = ty; dy < SHARED_SIZE; dy += TILE_SIZE) {
        for (int dx = tx; dx < SHARED_SIZE; dx += TILE_SIZE) {
            int gx = blockIdx.x * TILE_SIZE + dx - KERNEL_RADIUS;
            int gy = blockIdx.y * TILE_SIZE + dy - KERNEL_RADIUS;
            
            if (gx >= 0 && gx < input_w && gy >= 0 && gy < input_h) {
                s_input[dy][dx] = input[gy * input_w + gx];
            } else {
                s_input[dy][dx] = 0.0f;  // Zero padding
            }
        }
    }
    
    __syncthreads();
    
    if (out_x >= output_w || out_y >= output_h) return;
    
    // Compute convolution from shared memory
    float sum = 0.0f;
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            sum += s_input[ty + ky][tx + kx] * s_kernel[ky][kx];
        }
    }
    output[out_y * output_w + out_x] = sum;
}

// Separable convolution - decompose 2D kernel into two 1D passes
__global__ void conv1d_horizontal(
    const float* input,
    const float* kernel,
    float* output,
    int height, int width,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = kernel_size / 2;
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int in_x = x + k - radius;
        if (in_x >= 0 && in_x < width) {
            sum += input[y * width + in_x] * kernel[k];
        }
    }
    output[y * width + x] = sum;
}

__global__ void conv1d_vertical(
    const float* input,
    const float* kernel,
    float* output,
    int height, int width,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = kernel_size / 2;
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int in_y = y + k - radius;
        if (in_y >= 0 && in_y < height) {
            sum += input[in_y * width + x] * kernel[k];
        }
    }
    output[y * width + x] = sum;
}

// Im2col convolution - transform to matrix multiplication
__global__ void im2col_kernel(
    const float* input,
    float* col_matrix,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int output_h, int output_w
) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = output_h * output_w;
    
    if (col_idx >= total_cols) return;
    
    int out_y = col_idx / output_w;
    int out_x = col_idx % output_w;
    
    int col_height = kernel_h * kernel_w;
    
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            int row_idx = ky * kernel_w + kx;
            col_matrix[row_idx * total_cols + col_idx] = input[in_y * input_w + in_x];
        }
    }
}

// Utility functions
void init_gaussian_kernel(std::vector<float>& kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - radius;
            float dy = y - radius;
            float val = expf(-(dx*dx + dy*dy) / (2 * sigma * sigma));
            kernel[y * size + x] = val;
            sum += val;
        }
    }
    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

void init_sobel_kernel(std::vector<float>& kernel_x, std::vector<float>& kernel_y) {
    // Sobel X (horizontal edges)
    kernel_x = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    // Sobel Y (vertical edges)
    kernel_y = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
}

int main() {
    std::cout << "=== ROCm Convolution Kernels ===\n\n";
    
    // Image dimensions
    const int INPUT_H = 1024;
    const int INPUT_W = 1024;
    const int KERNEL_SIZE = 5;
    const int OUTPUT_H = INPUT_H - KERNEL_SIZE + 1;
    const int OUTPUT_W = INPUT_W - KERNEL_SIZE + 1;
    
    // Allocate host memory
    std::vector<float> h_input(INPUT_H * INPUT_W);
    std::vector<float> h_kernel(KERNEL_SIZE * KERNEL_SIZE);
    std::vector<float> h_output_naive(OUTPUT_H * OUTPUT_W);
    std::vector<float> h_output_shared(OUTPUT_H * OUTPUT_W);
    
    // Initialize input (random grayscale image)
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Initialize Gaussian blur kernel
    init_gaussian_kernel(h_kernel, KERNEL_SIZE, 1.0f);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    hipMalloc(&d_input, INPUT_H * INPUT_W * sizeof(float));
    hipMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    hipMalloc(&d_output, OUTPUT_H * OUTPUT_W * sizeof(float));
    
    // Copy to device
    hipMemcpy(d_input, h_input.data(), INPUT_H * INPUT_W * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, h_kernel.data(), KERNEL_SIZE * KERNEL_SIZE * sizeof(float), hipMemcpyHostToDevice);
    
    // Launch configurations
    dim3 block(16, 16);
    dim3 grid_naive((OUTPUT_W + block.x - 1) / block.x, (OUTPUT_H + block.y - 1) / block.y);
    dim3 grid_shared((OUTPUT_W + TILE_SIZE - 1) / TILE_SIZE, (OUTPUT_H + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warmup
    conv2d_naive<<<grid_naive, block>>>(d_input, d_kernel, d_output, 
                                         INPUT_H, INPUT_W, KERNEL_SIZE, KERNEL_SIZE,
                                         OUTPUT_H, OUTPUT_W);
    hipDeviceSynchronize();
    
    // Benchmark naive convolution
    const int ITERATIONS = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        conv2d_naive<<<grid_naive, block>>>(d_input, d_kernel, d_output,
                                             INPUT_H, INPUT_W, KERNEL_SIZE, KERNEL_SIZE,
                                             OUTPUT_H, OUTPUT_W);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float naive_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    hipMemcpy(h_output_naive.data(), d_output, OUTPUT_H * OUTPUT_W * sizeof(float), hipMemcpyDeviceToHost);
    
    // Benchmark shared memory convolution
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        conv2d_shared<<<grid_shared, dim3(TILE_SIZE, TILE_SIZE)>>>(
            d_input, d_kernel, d_output,
            INPUT_H, INPUT_W, KERNEL_SIZE,
            OUTPUT_H, OUTPUT_W);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    float shared_time = std::chrono::duration<float, std::milli>(end - start).count() / ITERATIONS;
    
    hipMemcpy(h_output_shared.data(), d_output, OUTPUT_H * OUTPUT_W * sizeof(float), hipMemcpyDeviceToHost);
    
    // Verify results match
    float max_diff = 0.0f;
    for (int i = 0; i < OUTPUT_H * OUTPUT_W; i++) {
        float diff = fabsf(h_output_naive[i] - h_output_shared[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    
    // Print results
    std::cout << "Input size: " << INPUT_H << "x" << INPUT_W << "\n";
    std::cout << "Kernel size: " << KERNEL_SIZE << "x" << KERNEL_SIZE << "\n";
    std::cout << "Output size: " << OUTPUT_H << "x" << OUTPUT_W << "\n\n";
    
    std::cout << "Performance Results:\n";
    std::cout << "  Naive convolution:  " << naive_time << " ms\n";
    std::cout << "  Shared memory:      " << shared_time << " ms\n";
    std::cout << "  Speedup:            " << naive_time / shared_time << "x\n\n";
    
    std::cout << "Verification:\n";
    std::cout << "  Max difference: " << max_diff << "\n";
    std::cout << "  Status: " << (max_diff < 1e-5f ? "PASS" : "FAIL") << "\n";
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_output);
    
    return 0;
}
