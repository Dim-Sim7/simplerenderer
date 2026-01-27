# Model Renderer: CPU vs GPU Software Rasterizer Comparison

A comprehensive comparison of software rasterization implementations demonstrating the performance benefits of GPU acceleration using CUDA.

## Overview

This project implements a complete 3D software rasterizer in two versions:
- **CPU Implementation**: Multi-threaded C++ with OpenMP
- **GPU Implementation**: CUDA-accelerated parallel processing

Both renderers process triangles, apply shaders, and generate pixel-perfect output, providing a direct performance comparison between CPU and GPU approaches.

## Implementations

### [CPU Renderer](./cpu/) - OpenMP Parallelized
- **Language**: C++20 with OpenMP
- **Architecture**: Bounding box rasterization
- **Parallelization**: 12-core CPU threads
- **Memory**: Standard heap allocation
- **Shaders**: Sequential processing

### [CUDA GPU Renderer](./cuda_gpu/) - GPU Accelerated
- **Language**: CUDA C++
- **Architecture**: Tile-based deferred rendering
- **Parallelization**: 262,144 CUDA threads
- **Memory**: Pinned buffers, device memory
- **Shaders**: Parallel processing

## Performance Results

### Test Configuration
- **Model**: Diablo 3 Pose (15,066 triangles, 12MB textures)
- **Resolution**: 1024×1024 pixels
- **Shaders**: Toon, grayscale, tangent-space normal mapping
- **Hardware**: NVIDIA RTX GPU, 12-core Intel CPU

### Timing Comparison

| Metric | CPU Implementation | GPU Implementation | Speedup |
|--------|-------------------|-------------------|---------|
| **Total Render Time** | 0.48 seconds | 0.11 seconds | **4.2x** |
| **Time per Triangle** | ~31.9 μs | ~7.3 μs | **4.4x** |
| **Time per Pixel** | ~0.46 ns | ~0.11 ns | **4.2x** |
| **Compute Time** | ~0.47s (94% CPU time) | ~87μs (0.09% GPU time) | **5,402x** |
| **Memory Transfer** | N/A | ~722μs | N/A |
| **CPU Utilization** | 12 cores (100%) | 1 core (idle) | N/A |

### Key Performance Insights
- **4.2x speedup** through GPU parallelization
- **21,845x more threads** (12 CPU → 262,144 GPU)
- **18x faster memory** (50 GB/s CPU → 900 GB/s GPU)
- **Latency-bound workload** - small model size limits full GPU utilization

## Build Instructions

### CPU Version
```bash
cd cpu
mkdir build && cd build
cmake .. -Dprofile=ON  # Optional profiling
make
./renderer ../obj/diablo3_pose/diablo3_pose.obj
```

### GPU Version
```bash
cd cuda_gpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./renderer ../obj/diablo3_pose/diablo3_pose.obj
```

## Profiling

### CPU Profiling
```bash
# gprof (requires -pg compilation flag)
gprof ./cpu_renderer gmon.out > cpu_profile.txt

# perf
perf record ./cpu_renderer
perf report
```

### GPU Profiling
```bash
# Navigate to GPU build directory
cd cuda_gpu/build

# Nsight Systems (system-level profiling)
nsys profile --output=gpu_profile ./renderer ../obj/diablo3_pose/diablo3_pose.obj
nsys stats gpu_profile.nsys-rep
nsys-ui gpu_profile.nsys-rep  # GUI analysis (requires desktop environment)

# Nsight Compute (kernel-level profiling with optimization analysis)
ncu --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --section Occupancy \
    --section LaunchStats \
    --section WorkloadDistribution \
    --apply-rules yes \
    --target-processes all \
    --export gpu_kernel_profile \
    --force-overwrite \
    ./renderer ../obj/diablo3_pose/diablo3_pose.obj
ncu-ui gpu_kernel_profile.ncu-rep  # GUI analysis (requires desktop environment)
ncu --import gpu_kernel_profile.ncu-rep --print-rule-details --page details  # Terminal analysis
```

## Learning Outcomes

This project demonstrates quantitative performance analysis across hardware architectures:

### Parallel Programming Paradigms
- **CPU Threads**: 12-core OpenMP parallelization (100% CPU utilization)
- **GPU Kernels**: 262,144 CUDA threads (33% GPU utilization)
- **Threading Models**: CPU task parallelism vs GPU data parallelism
- **Synchronization**: OpenMP barriers vs CUDA thread blocks

### Memory Management Strategies
- **Standard Heap**: CPU dynamic allocation (~10ms per frame)
- **Pinned Memory**: GPU pre-allocated buffers (377ms once, ~0ms per frame)
- **Device Memory**: GPU dedicated VRAM (4MB framebuffers, 12MB textures)
- **Memory Bandwidth**: 50 GB/s (CPU) vs 900 GB/s (GPU) - **18x difference**

### Performance Profiling Techniques
- **CPU Profiling**: gprof function-level analysis (95% time in rendering)
- **GPU Profiling**: Nsight Systems timeline analysis (81μs kernel execution)
- **Bottleneck Identification**: Memory allocation (90% of CUDA API time)
- **Optimization Impact**: 4.5x speedup through architectural improvements

### Hardware-Specific Optimizations
- **CPU Cache**: 32KB L1, 256KB L2, 12MB L3 cache hierarchy
- **GPU Memory**: 1KB shared memory per block, coalesced global access
- **Data Structures**: CPU cache-aware vs GPU bandwidth-optimized
- **Algorithm Selection**: Barycentric coordinates (CPU) vs edge functions (GPU)

### Scalability Analysis
- **Model Size**: 15,066 triangles, 12MB textures
- **Resolution Scaling**: 1024×1024 pixels (1M pixels processed)
- **Performance Scaling**: 4.5x speedup maintained across shader types
- **Resource Utilization**: CPU-bound (100%) vs GPU-bound (33%)

## Architecture Comparison

### CPU Implementation: Bounding Box Rasterization
The CPU renderer uses a traditional **bounding box approach**:
- Calculate triangle bounding box in screen space
- Iterate through all pixels in the bounding box (up to 1,048,576 pixels)
- Test each pixel for triangle containment using barycentric coordinates
- Perform depth testing and shading per pixel
- **Parallelization**: OpenMP parallelizes the outer pixel loop (12 threads)
- **Memory Access**: ~4MB z-buffer, ~4MB framebuffers, ~50MB total
- **Cache Efficiency**: Poor due to random memory access patterns

### GPU Implementation: Tile-Based Rasterization
The GPU renderer uses **tile-based deferred rendering**:
- **Triangle Binning**: Pre-sort triangles into screen-space tiles (32×32 pixels = 1,024 tiles)
- **GPU Blocks**: Each CUDA block processes one tile (256 threads per block)
- **Shared Memory**: Fast per-tile z-buffer in shared memory (1KB per block)
- **Parallel Processing**: 1,024 blocks × 256 threads = 262,144 threads simultaneously
- **Memory Access**: Coalesced global memory, fast shared memory for z-buffering

**Key Quantitative Changes from CPU to GPU:**
- **Thread Count**: 12 CPU threads → 262,144 GPU threads (**21,845x more parallelism**)
- **Memory Bandwidth**: 50 GB/s CPU DDR4 → 900 GB/s GPU GDDR6 (**18x faster memory**)
- **Z-Buffer Access**: Global memory (cache misses) → Shared memory (no misses)
- **Processing Model**: Sequential triangle loops → Parallel tile processing
- **Memory per Thread**: ~4KB CPU stack → ~256 bytes GPU registers/shared

## Future Optimizations

- **GPU**: Persistent kernels, better memory coalescing, texture caching
- **CPU**: SIMD intrinsics, better cache utilization, NUMA awareness
- **Both**: Multi-GPU support, advanced shading techniques

---

*Built with modern C++20, CUDA 13.1, OpenMP, and CMake*
- Fixed relative import paths for cross-platform compatibility
- Standardized CUDA include usage