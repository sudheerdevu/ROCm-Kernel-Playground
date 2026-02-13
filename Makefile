# ROCm Kernel Playground - Main Makefile

HIPCC ?= hipcc
GPU_ARCH ?= gfx1103

COMMON_FLAGS = -std=c++17 --offload-arch=$(GPU_ARCH)
INCLUDES = -I./utils

ifdef DEBUG
    COMMON_FLAGS += -g -O0 -DDEBUG
else
    COMMON_FLAGS += -O3
endif

KERNEL_DIRS = 01_hello_hip 02_vector_ops 03_reduction 04_matrix_mul \
              05_convolution 06_histogram 07_scan 08_memory_patterns

.PHONY: all clean $(KERNEL_DIRS)

all: $(KERNEL_DIRS)

01_hello_hip:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

02_vector_ops:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

03_reduction:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

04_matrix_mul:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

05_convolution:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

06_histogram:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

07_scan:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

08_memory_patterns:
	$(MAKE) -C kernels/$@ HIPCC=$(HIPCC) GPU_ARCH=$(GPU_ARCH)

clean:
	@for dir in $(KERNEL_DIRS); do \
		$(MAKE) -C kernels/$$dir clean; \
	done

test: all
	./benchmarks/run_all.sh
