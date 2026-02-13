# Hello HIP Makefile

HIPCC ?= hipcc
GPU_ARCH ?= gfx1103
CXXFLAGS = -std=c++17 --offload-arch=$(GPU_ARCH) -O3

TARGET = hello_hip

.PHONY: all clean run

all: $(TARGET)

$(TARGET): hello_hip.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
