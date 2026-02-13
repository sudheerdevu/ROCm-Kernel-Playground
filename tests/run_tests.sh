#!/bin/bash
# Test suite for ROCm Kernel Playground

set -e

echo "=== ROCm Kernel Playground Test Suite ==="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

# Test 1: Check kernel directories exist
echo "Test 1: Checking kernel tutorials..."
if [ -d "../kernels" ]; then
    kernel_count=$(ls -d ../kernels/*/ 2>/dev/null | wc -l)
    if [ "$kernel_count" -ge 5 ]; then
        pass "Found $kernel_count kernel tutorials"
    else
        fail "Expected at least 5 kernel tutorials, found $kernel_count"
    fi
else
    fail "kernels/ directory not found"
fi

# Test 2: Check each kernel has required files
echo "Test 2: Checking kernel structure..."
for dir in ../kernels/*/; do
    name=$(basename "$dir")
    if [ -f "${dir}Makefile" ] || [ -f "${dir}CMakeLists.txt" ]; then
        pass "$name has build file"
    else
        fail "$name missing build file"
    fi
done

# Test 3: Check HIP source files syntax (if hipcc available)
echo "Test 3: Checking HIP syntax..."
if command -v hipcc &> /dev/null; then
    for hip_file in ../kernels/*/*.hip ../src/*.hip 2>/dev/null; do
        if [ -f "$hip_file" ]; then
            name=$(basename "$hip_file")
            if hipcc -fsyntax-only "$hip_file" 2>/dev/null; then
                pass "$name syntax OK"
            else
                fail "$name has syntax errors"
            fi
        fi
    done
else
    skip "hipcc not available - skipping syntax check"
fi

# Test 4: Check utility headers
echo "Test 4: Checking utility headers..."
if [ -f "../src/hip_utils.hpp" ]; then
    pass "hip_utils.hpp exists"
else
    fail "hip_utils.hpp not found"
fi

if [ -f "../src/benchmark.hpp" ]; then
    pass "benchmark.hpp exists"
else
    fail "benchmark.hpp not found"
fi

# Test 5: Check profiling scripts
echo "Test 5: Checking profiling tools..."
if [ -f "../profiling/profile_all.sh" ]; then
    pass "profile_all.sh exists"
else
    fail "profile_all.sh not found"
fi

if [ -f "../profiling/analyze_results.py" ]; then
    pass "analyze_results.py exists"
else
    fail "analyze_results.py not found"
fi

# Test 6: Check documentation
echo "Test 6: Checking documentation..."
if [ -f "../docs/getting-started.md" ]; then
    pass "getting-started.md exists"
else
    fail "getting-started.md not found"
fi

echo ""
echo "=== Test Results ==="
echo -e "Passed:  ${GREEN}${PASSED}${NC}"
echo -e "Failed:  ${RED}${FAILED}${NC}"
echo -e "Skipped: ${YELLOW}${SKIPPED}${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
