# Software Rasterizer (C++)

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
- Screen-space ambient occlusion (learning implementation)
- Sobel edge detection using depth buffer
- OpenMP parallelization
- TGA image output

## Output
- Renders a 1000×1000 image to `framebuffer.tga`

## Build & Run
```bash
g++ -O2 -fopenmp main.cpp -o renderer
./renderer model.obj

OR

g++ tgaimage.cpp graphics_library.cpp model.cpp main.cpp -O0 
-fopenmp -Wno-narrowing && time ./renderer obj/diablo3_pose/*.obj && display framebuffer.tga


Multiple OBJ files can be passed and are rendered sequentially.

Notes

The ambient occlusion pass is intentionally incomplete:
No normal buffer or TBN rotation
View-space hemisphere sampling
Kept as a learning reference, not a final SSAO solution

Purpose

This project exists to understand rasterization, projection, depth testing, and shading by implementing them directly. It is a learning renderer, not a production engine.