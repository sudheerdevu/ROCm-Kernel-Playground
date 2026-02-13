#!/bin/bash
# Profile all kernels and generate reports
# Usage: ./profile_all.sh [output_dir]

OUTPUT_DIR="${1:-./profile_results}"
KERNELS_DIR="../kernels"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Metrics to collect
METRICS="GPUBusy,VALUInsts,VALUUtilization,MemUnitBusy,FetchSize,WriteSize"

echo "ROCm Kernel Playground - Profiling Suite"
echo "========================================="
echo ""

# Check for rocprof
if ! command -v rocprof &> /dev/null; then
    echo "Error: rocprof not found. Please ensure ROCm is installed."
    exit 1
fi

# Profile each kernel directory
for kernel_dir in "$KERNELS_DIR"/*/; do
    kernel_name=$(basename "$kernel_dir")
    echo "Profiling: $kernel_name"
    
    # Find executable
    executable=$(find "$kernel_dir" -maxdepth 1 -type f -executable | head -1)
    
    if [ -z "$executable" ]; then
        echo "  No executable found, skipping..."
        continue
    fi
    
    # Create output subdirectory
    kernel_output="$OUTPUT_DIR/$kernel_name"
    mkdir -p "$kernel_output"
    
    # Basic profiling
    echo "  Running basic profile..."
    rocprof -o "$kernel_output/results.csv" "$executable" > /dev/null 2>&1
    
    # Detailed metrics
    echo "  Collecting performance counters..."
    cat > "$kernel_output/metrics.txt" << EOF
pmc: GPUBusy
pmc: VALUInsts
pmc: VALUUtilization
pmc: MemUnitBusy
pmc: FetchSize
pmc: WriteSize
EOF
    
    rocprof -i "$kernel_output/metrics.txt" -o "$kernel_output/detailed.csv" "$executable" > /dev/null 2>&1
    
    # HIP trace
    echo "  Capturing HIP trace..."
    rocprof --hip-trace -o "$kernel_output/hip_trace.csv" "$executable" > /dev/null 2>&1
    
    echo "  Done!"
done

echo ""
echo "Profiling complete. Results in: $OUTPUT_DIR"
echo ""
echo "Summary:"
find "$OUTPUT_DIR" -name "*.csv" -exec wc -l {} \;
