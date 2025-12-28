# SimpleRenderer

A small software renderer built in C++ as a learning project focused on the fundamentals of 3D graphics pipelines.

This project is inspired by *Tiny Renderer*–style approaches and is intended to deepen understanding of how modern GPUs work by re-implementing core concepts on the CPU.

---

## Features

- Custom math library (vectors, matrices)
- Model loading (OBJ)
- Perspective projection
- Camera transforms
- Triangle rasterization
- Depth buffering (Z-buffer)
- Basic shading
- Framebuffer output

---

## Motivation

Modern graphics APIs abstract away a large amount of complexity.  
This project intentionally avoids high-level graphics libraries to focus on:

- Coordinate spaces and transformations
- Rasterization logic
- Depth testing
- Shading fundamentals
- Data-oriented rendering design

The goal is not performance, but **correctness and understanding**.

---
## Build & Run

### Requirements
- C++17 or newer
- CMake
- Linux (tested on WSL)

### Build
```bash
mkdir build
cd build
cmake ..
make
```
---


By Dimitrije Simic

This project is part of a longer-term focus on graphics programming and rendering systems.
