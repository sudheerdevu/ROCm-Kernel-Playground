# ROCm Kernel Playground - Getting Started

Welcome to the ROCm Kernel Playground! This guide will help you get started with writing and optimizing HIP kernels.

## Prerequisites

- AMD GPU (MI series, Radeon RX 6000/7000 series, or similar)
- ROCm 5.0 or later
- CMake 3.16+ (optional)
- Modern C++ compiler with C++17 support

## Building the Examples

### Using Make

```bash
# Build all examples
make all

# Build a specific example
make 01_hello_hip

# Clean build files
make clean
```

### Using CMake (Alternative)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Project Structure

```
ROCm-Kernel-Playground/
├── kernels/           # Collection of HIP kernel examples
│   ├── 01_hello_hip/  # Basic HIP introduction
│   ├── 02_vector_ops/ # Vector operations
│   ├── 03_reduction/  # Parallel reduction
│   ├── 04_matrix_mul/ # Matrix multiplication
│   ├── 05_convolution/# 2D convolution
│   ├── 06_histogram/  # Histogram computation
│   ├── 07_scan/       # Prefix scan
│   └── 08_memory_patterns/ # Memory access patterns
├── src/               # Shared utilities
├── docs/              # Documentation
├── profiling/         # Profiling scripts
├── benchmarks/        # Benchmark results
└── utils/             # Helper scripts
```

## Running Your First Kernel

### 1. Check GPU Status

```bash
rocm-smi --showallinfo
```

### 2. Build and Run Hello World

```bash
cd kernels/01_hello_hip
make
./hello_hip
```

### 3. Profile the Kernel

```bash
rocprof ./hello_hip
cat results.csv
```

## Key Concepts

### HIP Programming Model

- **Grid**: Collection of thread blocks
- **Block**: Collection of threads that can cooperate
- **Thread**: Single execution unit
- **Wavefront**: 64 threads executing in lockstep (AMD)

### Memory Hierarchy

| Memory Type | Scope | Speed | Size |
|-------------|-------|-------|------|
| Registers | Thread | Fastest | Limited |
| LDS (Shared) | Block | Fast | 64KB/CU |
| L1 Cache | CU | Fast | 16KB |
| L2 Cache | Device | Medium | MB scale |
| Global Memory | Grid | Slow | GB scale |

### Launch Configuration

```cpp
// 1D kernel launch
int n = 1024;
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
kernel<<<gridSize, blockSize>>>(args...);

// 2D kernel launch
dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
kernel<<<grid, block>>>(args...);
```

## Optimization Tips

1. **Coalesce Memory Access**: Consecutive threads access consecutive memory
2. **Use Shared Memory**: Cache frequently accessed data
3. **Avoid Divergence**: Minimize conditional branching within wavefronts
4. **Sufficient Parallelism**: Launch enough threads to hide memory latency
5. **Optimal Block Size**: Use multiples of 64 (wavefront size)

## Next Steps

1. Work through the kernel examples in order
2. Read the [Kernel Optimization Guide](optimization.md)
3. Try the [Profiling Tutorial](profiling.md)
4. Experiment with modifying the examples

## Useful Commands

```bash
# Check available GPUs
rocm-smi -i

# Monitor GPU usage
watch -n1 rocm-smi --showuse

# Compile HIP code
hipcc -o program program.hip

# Profile with metrics
rocprof --stats ./program
```

## Troubleshooting

### Permission Denied

```bash
sudo chmod 666 /dev/kfd
sudo usermod -a -G render $USER
# Logout and login again
```

### Out of Memory

- Reduce problem size
- Free unused allocations
- Check for memory leaks

### Kernel Not Launching

- Verify block/grid dimensions are valid
- Check that shared memory doesn't exceed limits
- Ensure proper synchronization
