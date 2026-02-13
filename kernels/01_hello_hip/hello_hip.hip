/**
 * Hello HIP - First kernel demonstrating basic HIP concepts
 * 
 * This is the "Hello World" of HIP programming, showing:
 * - Device enumeration and properties
 * - Memory allocation
 * - Kernel launch
 * - Error handling
 */

#include <hip/hip_runtime.h>
#include <iostream>

// Simple kernel - each thread writes its global ID
__global__ void helloKernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = idx;
    }
}

// Kernel with more interesting computation
__global__ void squareKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        output[idx] = val * val;
    }
}

void printDeviceInfo() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    std::cout << "Found " << deviceCount << " HIP device(s)\n\n";
    
    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        
        std::cout << "Device " << i << ": " << props.name << "\n";
        std::cout << "  Compute capability: " << props.major << "." << props.minor << "\n";
        std::cout << "  Total memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
        std::cout << "  Compute units: " << props.multiProcessorCount << "\n";
        std::cout << "  Max threads/block: " << props.maxThreadsPerBlock << "\n";
        std::cout << "  Warp size: " << props.warpSize << "\n";
        std::cout << "  GCN Arch: " << props.gcnArchName << "\n";
        std::cout << "\n";
    }
}

int main() {
    std::cout << "=== Hello HIP ===\n\n";
    
    // Print device information
    printDeviceInfo();
    
    // Parameters
    const int N = 1024;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // Host arrays
    int* h_output = new int[N];
    float* h_input = new float[N];
    float* h_result = new float[N];
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Device arrays
    int* d_output;
    float *d_input, *d_result;
    
    hipMalloc(&d_output, N * sizeof(int));
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_result, N * sizeof(float));
    
    // Copy input to device
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    
    // Launch hello kernel
    std::cout << "Launching helloKernel with " << gridSize << " blocks of " 
              << blockSize << " threads\n";
    helloKernel<<<gridSize, blockSize>>>(d_output, N);
    hipDeviceSynchronize();
    
    // Launch square kernel
    std::cout << "Launching squareKernel\n";
    squareKernel<<<gridSize, blockSize>>>(d_input, d_result, N);
    hipDeviceSynchronize();
    
    // Copy results back
    hipMemcpy(h_output, d_output, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_result, d_result, N * sizeof(float), hipMemcpyDeviceToHost);
    
    // Verify
    bool hello_ok = true, square_ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != i) hello_ok = false;
        if (h_result[i] != i * i) square_ok = false;
    }
    
    std::cout << "\nResults:\n";
    std::cout << "  helloKernel: " << (hello_ok ? "PASSED" : "FAILED") << "\n";
    std::cout << "  squareKernel: " << (square_ok ? "PASSED" : "FAILED") << "\n";
    
    // Show some values
    std::cout << "\nFirst 10 square results: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << "\n";
    
    // Cleanup
    hipFree(d_output);
    hipFree(d_input);
    hipFree(d_result);
    delete[] h_output;
    delete[] h_input;
    delete[] h_result;
    
    std::cout << "\nHello HIP complete!\n";
    return 0;
}
