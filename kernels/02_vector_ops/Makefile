# Vector Operations Makefile

HIPCC ?= hipcc
GPU_ARCH ?= gfx1103
CXXFLAGS = -std=c++17 --offload-arch=$(GPU_ARCH) -O3 -I../../utils

TARGET = vector_ops_test

.PHONY: all clean run

all: $(TARGET)

$(TARGET): vector_ops.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	rocprof --stats ./$(TARGET)
