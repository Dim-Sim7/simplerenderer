# CUDA GPU Software Rasterizer

A GPU-accelerated software rasterizer using CUDA to learn parallel graphics programming and GPU optimization.

## Features
- CUDA kernel-based triangle rasterization
- Parallel vertex processing and binning
- GPU memory management with pinned buffers
- Asynchronous texture uploads
- Z-buffered rendering with atomic operations
- Multiple shader types (toon, grayscale, tangent-space normal mapping)
- CUDA streams for concurrent operations
- NVIDIA Nsight profiling integration

## Performance Comparison

### Test Model: Diablo 3 Pose (5,022 faces, 2,519 vertices, 15,066 triangles, 12MB textures)

| Implementation | Render Time | Time/Triangle | Time/Pixel | Notes |
|----------------|-------------|---------------|------------|-------|
| **CPU (OpenMP)** | 0.48 seconds | ~31.9 μs | ~0.46 ns | 12 cores, sequential shaders |
| **GPU (CUDA)** | 0.11 seconds | ~7.3 μs | ~0.11 ns | 1,024 CUDA cores, parallel shaders |

**Performance Analysis**: The GPU implementation achieves 4.2x speedup over CPU, processing ~7.3 μs per triangle vs ~31.9 μs on CPU.

### Profiling Breakdown (GPU Implementation)

#### CUDA API Time Distribution
- **cudaHostAlloc**: 85.5% (377ms) - Pinned memory allocation (128MB buffer)
- **cudaFreeHost**: 12.6% (55ms) - Memory cleanup
- **cuLibraryLoadData**: 3.6% (16ms) - CUDA runtime initialization
- **cudaStreamCreate**: 0.7% (3ms) - Async operations setup
- **cudaMalloc**: 0.5% (2ms) - Device memory allocation (4MB framebuffer)
- **cudaMemcpyAsync**: 0.4% (2ms) - Texture uploads (12MB total)

#### GPU Kernel Performance
- **raster_GPU kernel**: 81μs (92.7% of GPU compute time) - 5,022 triangles processed
- **clear_z kernel**: 6μs (7.3% of GPU compute time) - 1M pixels cleared

#### Memory Operations
- **Host→Device copies**: 467μs (64.3% of transfer time) - 12MB at 25.6 GB/s
- **Device→Host copy**: 255μs (35.1% of transfer time) - 4MB at 15.7 GB/s
- **CUDA memset**: 4μs (0.6% of transfer time) - Memory initialization

#### GPU Resource Utilization
- **CUDA Cores Used**: 1,024 (out of 3,072 available - 33% utilization)
- **Memory Bandwidth**: ~40 GB/s peak utilization
- **Shared Memory per Block**: 1KB (256 threads × 16×16 pixel tile)
- **Thread Blocks**: 1,024 (32×32 grid for 1024×1024 resolution)
- **Threads per Block**: 256 (16×16 tile processing)

### Nsight Compute Kernel Analysis

#### Performance Assessment Summary
The GPU implementation achieves excellent performance for software rasterization, with the primary limitation being **workload size rather than algorithmic inefficiency**. Nsight Compute analysis reveals this is a latency-bound workload where the GPU hardware cannot be fully saturated due to the relatively small problem size (15K triangles).

#### Key Performance Metrics
- **Memory Throughput**: 317.35 GB/s (72.28% of peak sustained)
- **Compute Throughput**: 64.71% of peak
- **Achieved Occupancy**: 45.33% (limited by register pressure)
- **Theoretical Occupancy**: 66.67% (register-limited)
- **Issue Slots Busy**: 32.22%
- **SM Busy**: 34.46%

#### Optimization Analysis Results

**Primary Finding**: The implementation is **well-optimized** for its workload size. The low throughput percentages (below 80% of peak) indicate latency issues from insufficient parallelism rather than bottlenecks.

**Rule-Based Recommendations**:
1. **Pipeline Under-utilization** (Est. Local Speedup: 95.17%)
   - ALU pipeline: 4.83% of peak utilization
   - FMA pipeline: 7.15% of peak utilization
   - **Cause**: Insufficient warps per scheduler (workload too small)

2. **Occupancy Limitations** (Est. Local Speedup: 53.09%)
   - Theoretical limit: 66.67% (register-constrained)
   - Achieved: 45.33-58.73% across kernels
   - **Cause**: Register pressure from complex shader calculations

3. **Workload Distribution** (Est. Speedup: 5-6%)
   - Minor imbalances across SMs/SMSPs/L2 slices
   - **Impact**: Negligible for current workload size

#### Memory Workload Analysis
- **No memory spilling**: Zero local/shared memory spilling requests
- **Efficient bandwidth utilization**: 72.28% of peak memory throughput
- **Coalesced memory access**: No DRAM bottlenecks detected
- **L2 cache efficiency**: No sector promotion misses

#### Kernel-Specific Performance

**raster_GPU Kernel** (Main rendering, 81μs):
- Memory throughput: 89.32 GB/s
- Compute throughput: 48.02%
- Issue slots busy: 27.00%
- **Bottleneck**: Latency from small workload size

**clear_z Kernel** (Z-buffer clear, 6μs):
- Memory throughput: 342.89 GB/s (78.27% utilization)
- Compute throughput: 64.69%
- Issue slots busy: 32.21%
- **Assessment**: Memory-bound but efficient

### Optimizations Implemented
1. **Pre-allocated pinned buffers** - Eliminates per-frame memory allocation overhead
2. **Write-combined memory** - Optimized for CPU→GPU data transfers
3. **Asynchronous operations** - Overlapping computation with data transfers
4. **Tile-based rendering** - Efficient GPU thread utilization

## Build & Run
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./renderer ../obj/diablo3_pose/diablo3_pose.obj
```

## Profiling
```bash
# Generate profile
nsys profile --output=profile_output ./renderer model.obj

# View in GUI
nsys-ui profile_output.nsys-rep

# Generate statistics
nsys stats profile_output.nsys-rep
```

## Output
- Renders a 1024×1024 image to `framebuffer.tga`
- Generates multiple shader variants: toon, grayscale, tangent-texture

## Rasterization Architecture

### Tile-Based Rendering Pipeline
1. **Triangle Binning (CPU)**: Sort triangles into 32×32 pixel tiles based on bounding boxes
2. **GPU Kernel Launch**: One CUDA block per tile (1024×1024 image = 32×32 = 1024 blocks)
3. **Per-Tile Processing**: Each block processes its assigned triangles using shared memory
4. **Edge Function Testing**: Half-space edge functions for triangle containment
5. **Shared Z-Buffer**: Fast per-tile depth testing in shared memory
6. **Fragment Shading**: Perspective-correct interpolation and shader execution

### Key Differences from CPU Implementation
- **CPU**: Sequential bounding box iteration with global z-buffer
- **GPU**: Parallel tile processing with local shared memory z-buffers
- **CPU**: Barycentric coordinate testing
- **GPU**: Half-space edge function testing (more GPU-friendly)
- **CPU**: Immediate rendering per triangle
- **GPU**: Deferred rendering with triangle binning

## Optimizations Implemented

### Memory Management
- **Pre-allocated Pinned Buffers**: 128MB staging buffer allocated once at startup
- **Write-Combined Memory**: Optimized for CPU→GPU texture transfers
- **Unified Memory Strategy**: Proper host/device buffer separation

### Performance Issues Solved
1. **cudaHostAlloc Bottleneck**: Moved from per-model allocation (213ms) to startup allocation
2. **Memory Transfer Latency**: Used async streams and pinned memory
3. **Thread Divergence**: Tile-based approach reduces warp divergence
4. **Shared Memory Utilization**: Per-tile z-buffers maximize shared memory bandwidth
5. **Atomic Operation Conflicts**: Careful design to minimize z-buffer conflicts

### Build System Fixes
- Corrected CMakeLists.txt source file paths
- Fixed relative import paths for cross-platform compatibility
- Standardized CUDA include usage

## Profiling with NVIDIA Nsight

### Nsight Systems (System-Level Profiling)
```bash
# Navigate to build directory
cd cuda_gpu/build

# Generate system-level profile
nsys profile --output=gpu_system_profile ./renderer ../obj/diablo3_pose/diablo3_pose.obj

# View in GUI (requires display)
nsys-ui gpu_system_profile.nsys-rep

# Generate statistics report
nsys stats gpu_system_profile.nsys-rep
```

### Nsight Compute (Kernel-Level Profiling)
```bash
# Navigate to build directory
cd cuda_gpu/build

# Generate comprehensive kernel profile with all sections enabled
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

# View detailed analysis in GUI (requires display)
ncu-ui gpu_kernel_profile.ncu-rep

# Print rule details and performance suggestions to terminal
ncu --import gpu_kernel_profile.ncu-rep --print-rule-details --page details

# Quick summary of key metrics
ncu --import gpu_kernel_profile.ncu-rep --print-summary
```

### Key Profiling Metrics to Analyze
- **Compute Throughput**: Should be >60% of peak for good utilization
- **Memory Throughput**: Should be >60% of peak for efficient bandwidth usage
- **Achieved Occupancy**: Register-limited (typically 66.67% theoretical max)
- **Issue Slots Busy**: Indicates pipeline utilization
- **Memory Workload Analysis**: Check for coalescing and bank conflicts

### Common Optimization Findings
- **Latency-bound workload**: Small problem size limits GPU utilization
- **Register pressure**: Complex shaders reduce theoretical occupancy
- **Memory efficiency**: Well-coalesced access patterns with no spilling