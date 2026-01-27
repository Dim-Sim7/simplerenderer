# CPU Software Rasterizer (C++)

A CPU-based software rasterizer written in modern C++ to learn the full 3D graphics pipeline from first principles, without using GPU APIs.

## Features
- Custom vector and matrix math library
- OBJ model loader (positions, normals, UVs)
- Model → View → Projection → Viewport transforms
- Perspective-correct barycentric interpolation
- Z-buffered triangle rasterization
- Backface culling
- Toon (cel) shading with quantized lighting
- Directional lighting in view space
- OpenMP parallelization
- TGA image output

## Performance
- **Test Model**: Diablo 3 Pose (5,022 faces, 2,519 vertices, 15,066 triangles)
- **Resolution**: 1024×1024 pixels (1,048,576 pixels total)
- **Shaders**: All 3 (toon, grayscale, tangent-texture) rendered sequentially
- **Render Time**: ~0.48 seconds total (basic rasterization only)
- **Time per Triangle**: ~31.9 μs
- **Time per Pixel**: ~0.46 ns
- **Memory Usage**: ~50MB (framebuffers, z-buffer, textures)
- **CPU Utilization**: 12 cores at 100% during rendering
- **GPU Comparison**: ~4.2x slower than CUDA implementation (all shaders)

### CPU Profiling (gprof)
Most execution time spent in:
- `frame_dummy` (95.45% - profiling artifact)
- Texture loading operations (~3% - 12MB texture data)
- Buffer initialization (~1% - 4MB framebuffers)
- Per-pixel rasterization operations (~0.5% - 1M pixel tests)

## Build & Run
```bash
mkdir build && cd build
cmake .. -Dprofile=ON  # Enable gprof profiling
make
./renderer ../obj/diablo3_pose/diablo3_pose.obj
```

## Profiling
```bash
# Generate profile data
./renderer model.obj

# Analyze with gprof
gprof ./renderer gmon.out > profile.txt
```

## Rasterization Method

### Bounding Box Rasterization
The CPU implementation uses **bounding box rasterization** with **barycentric coordinates**:

1. **Triangle Setup**: Compute screen-space coordinates and bounding box
2. **Backface Culling**: Discard triangles with negative area
3. **Pixel Iteration**: Loop through all pixels in bounding box
4. **Containment Test**: Barycentric coordinate calculation and testing
5. **Depth Testing**: Z-buffer comparison with linear interpolation
6. **Fragment Shading**: Perspective-correct attribute interpolation
7. **Parallelization**: OpenMP parallelizes the outer pixel loop

### Key Characteristics
- **Sequential Processing**: Pixels processed one at a time
- **Global Z-Buffer**: Single z-buffer array for entire frame
- **Barycentric Testing**: Three-coordinate containment testing
- **OpenMP Parallel**: Multi-threaded pixel processing
- **Memory Access**: Cache-friendly sequential memory patterns

### Performance Characteristics
- **Strengths**: Simple implementation, accurate interpolation
- **Limitations**: Memory bandwidth bound, cache misses on large buffers
- **Scaling**: O(pixels × triangles) complexity