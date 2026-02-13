HIPCC = hipcc
HIPFLAGS = -O3 -std=c++17
TARGET = memory_patterns

GPU_ARCH ?= gfx1103

all: $(TARGET)

$(TARGET): memory_patterns.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $<

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	rocprof --stats ./$(TARGET)

clean:
	rm -f $(TARGET) *.csv *.json *.txt

.PHONY: all run profile clean
