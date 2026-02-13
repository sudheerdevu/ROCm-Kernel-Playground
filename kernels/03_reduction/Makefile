# Reduction kernels Makefile

HIPCC ?= hipcc
GPU_ARCH ?= gfx1103
CXXFLAGS = -std=c++17 --offload-arch=$(GPU_ARCH) -O3 -I../../utils

TARGET = reduction_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): reduction.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	rocprof --stats ./$(TARGET)
