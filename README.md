# ROCm Kernel Playground 🎮

Educational HIP/ROCm kernel examples demonstrating GPU programming concepts from basic to advanced.

## Overview

This repository provides hands-on learning material for AMD GPU programming using HIP. Each kernel showcases specific optimization techniques with detailed comments explaining the "why" behind each approach.

## Project Structure

```
ROCm-Kernel-Playground/
├── README.md
├── Makefile
├── kernels/
│   ├── 01_hello_hip/          # Basic HIP setup
│   ├── 02_vector_ops/         # Simple parallel patterns
│   ├── 03_reduction/          # Parallel reduction strategies
│   ├── 04_matrix_mul/         # GEMM optimization journey
│   ├── 05_convolution/        # 2D convolution patterns
│   ├── 06_histogram/          # Atomic operations
│   ├── 07_scan/               # Prefix sum algorithms
│   └── 08_memory_patterns/    # Coalescing & bank conflicts
├── utils/
│   └── hip_utils.h            # Common utilities
└── benchmarks/
    └── run_all.sh
```

## Learning Path

### Level 1: Foundations
1. **Hello HIP** - Device queries, kernel launch, error handling
2. **Vector Ops** - Memory allocation, grid/block mapping

### Level 2: Core Patterns
3. **Reduction** - Tree reduction, warp shuffles, atomic finalization
4. **Matrix Multiply** - Tiled GEMM, LDS usage, register blocking

### Level 3: Advanced
5. **Convolution** - 2D stencils, halo regions, texture memory
6. **Histogram** - Privatization, shared memory atomics
7. **Scan** - Work-efficient prefix sum, Blelloch algorithm
8. **Memory Patterns** - Coalescing, bank conflicts, padding

## Building

```bash
# All kernels
make all

# Specific kernel
make 02_vector_ops

# With debug symbols
make DEBUG=1 all
```

## Target Architecture

Default: gfx1103 (RDNA 3)

```bash
# Override architecture
make GPU_ARCH=gfx90a all
```

## Prerequisites

- ROCm 5.x+
- HIP SDK
- hipcc compiler

## Example: Vector Addition

```cpp
// kernels/02_vector_ops/vector_add.hip

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check - essential for cases where n % blockDim != 0
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

## Performance Tips Demonstrated

1. **Memory Coalescing**: Access consecutive memory addresses from consecutive threads
2. **Occupancy**: Balance between threads/block and registers/thread
3. **LDS Tiling**: Reduce global memory traffic for data reuse patterns
4. **Warp-Level Primitives**: Use `__shfl_*` for efficient intra-warp communication
5. **Avoid Bank Conflicts**: Pad shared memory arrays when needed

## Profiling

Each kernel includes profiling commands:

```bash
cd kernels/03_reduction
rocprof --stats ./reduction_test
rocprof -i ../../profiling/counters.txt ./reduction_test
```

## License

MIT - Use freely for learning and experimentation.

## Author

Sudheer Devu - AI Performance Engineer
