/**
 * Kernel Benchmarking Framework
 * 
 * Provides infrastructure for benchmarking and comparing HIP kernels.
 */

#pragma once

#include "hip_utils.hpp"
#include <functional>
#include <map>
#include <iomanip>

struct BenchmarkResult {
    std::string name;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double bandwidth_gbps;
    double gflops;
    int iterations;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << name << ":" << std::endl;
        std::cout << "  Avg time: " << avg_time_ms << " ms" << std::endl;
        std::cout << "  Min/Max:  " << min_time_ms << " / " << max_time_ms << " ms" << std::endl;
        std::cout << "  Std dev:  " << std_dev_ms << " ms" << std::endl;
        if (bandwidth_gbps > 0) {
            std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
        }
        if (gflops > 0) {
            std::cout << "  GFLOPS:   " << gflops << std::endl;
        }
    }
};


class KernelBenchmark {
public:
    using KernelFn = std::function<void()>;
    
    KernelBenchmark(const std::string& name) : name_(name) {}
    
    void add_kernel(const std::string& kernel_name, KernelFn fn) {
        kernels_[kernel_name] = fn;
    }
    
    BenchmarkResult run_kernel(const std::string& kernel_name,
                                size_t data_bytes = 0,
                                size_t flops = 0,
                                int warmup = 10,
                                int iterations = 100) {
        auto it = kernels_.find(kernel_name);
        if (it == kernels_.end()) {
            std::cerr << "Kernel not found: " << kernel_name << std::endl;
            return {};
        }
        
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            it->second();
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Collect timing data
        std::vector<double> times;
        times.reserve(iterations);
        
        for (int i = 0; i < iterations; ++i) {
            HipTimer timer;
            timer.start();
            it->second();
            timer.stop();
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        double sum = 0, min_t = times[0], max_t = times[0];
        for (double t : times) {
            sum += t;
            min_t = std::min(min_t, t);
            max_t = std::max(max_t, t);
        }
        double avg = sum / iterations;
        
        double var_sum = 0;
        for (double t : times) {
            var_sum += (t - avg) * (t - avg);
        }
        double std_dev = std::sqrt(var_sum / iterations);
        
        BenchmarkResult result;
        result.name = kernel_name;
        result.avg_time_ms = avg;
        result.min_time_ms = min_t;
        result.max_time_ms = max_t;
        result.std_dev_ms = std_dev;
        result.iterations = iterations;
        
        if (data_bytes > 0) {
            result.bandwidth_gbps = (data_bytes / 1e9) / (avg / 1e3);
        }
        if (flops > 0) {
            result.gflops = (flops / 1e9) / (avg / 1e3);
        }
        
        return result;
    }
    
    void run_all(size_t data_bytes = 0, size_t flops = 0,
                 int warmup = 10, int iterations = 100) {
        std::cout << "=== Benchmark: " << name_ << " ===" << std::endl;
        std::cout << std::endl;
        
        for (const auto& [name, fn] : kernels_) {
            auto result = run_kernel(name, data_bytes, flops, warmup, iterations);
            result.print();
            std::cout << std::endl;
        }
    }
    
    void compare(const std::string& baseline, size_t data_bytes = 0,
                 size_t flops = 0, int warmup = 10, int iterations = 100) {
        auto baseline_result = run_kernel(baseline, data_bytes, flops, warmup, iterations);
        
        std::cout << "=== Comparison (baseline: " << baseline << ") ===" << std::endl;
        std::cout << std::endl;
        
        baseline_result.print();
        std::cout << "  (baseline)" << std::endl << std::endl;
        
        for (const auto& [name, fn] : kernels_) {
            if (name == baseline) continue;
            
            auto result = run_kernel(name, data_bytes, flops, warmup, iterations);
            result.print();
            
            double speedup = baseline_result.avg_time_ms / result.avg_time_ms;
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                      << speedup << "x" << std::endl << std::endl;
        }
    }
    
private:
    std::string name_;
    std::map<std::string, KernelFn> kernels_;
};
