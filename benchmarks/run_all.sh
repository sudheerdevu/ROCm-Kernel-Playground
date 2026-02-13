#!/bin/bash
# Run all kernel benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  ROCm Kernel Playground - Full Benchmark"
echo "=========================================="
echo ""

cd "$ROOT_DIR"

# Check for GPU
if ! rocm-smi &> /dev/null; then
    echo "Warning: ROCm not detected, tests may fail"
fi

# Build all
echo "Building all kernels..."
make all
echo ""

# Run each benchmark
KERNELS=(
    "01_hello_hip/hello_hip"
    "02_vector_ops/vector_ops_test"
    "03_reduction/reduction_test"
    "04_matrix_mul/gemm_test"
)

for kernel in "${KERNELS[@]}"; do
    path="kernels/$kernel"
    if [ -x "$path" ]; then
        echo "=========================================="
        echo "Running: $kernel"
        echo "=========================================="
        "./$path"
        echo ""
    else
        echo "Skipping $kernel (not built)"
    fi
done

echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
