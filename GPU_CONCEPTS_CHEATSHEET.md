# GPU Programming & Architecture - Quick Reference Cheatsheet

## Core Concepts

### What is GPU?
**Graphics Processing Unit** - Specialized parallel processor
- **Purpose**: Execute thousands of operations simultaneously
- **Your GPU**: NVIDIA RTX 3050 (2560 CUDA cores, 8GB VRAM)

### What is CUDA?
**Compute Unified Device Architecture** - NVIDIA's parallel computing platform
- **Language**: Extension of C/C++
- **Purpose**: Program GPUs for general-purpose computing

### What is NVCC?
**NVIDIA CUDA Compiler** - Compiles CUDA code
- **Input**: `.cu` files (CUDA source)
- **Output**: Executable with GPU code
- **Process**: Separates CPU code (host) and GPU code (device)

---

## CUDA Terminology Quick Reference

| Term | Definition | Example |
|------|------------|---------|
| **Host** | CPU and its memory | Your computer's processor |
| **Device** | GPU and its memory | RTX 3050 graphics card |
| **Kernel** | Function that runs on GPU | `__global__ void myKernel()` |
| **Thread** | Single execution unit | Processes one pixel |
| **Block** | Group of threads (max 1024) | 16×16 = 256 threads |
| **Grid** | Collection of blocks | All blocks together |
| **Warp** | 32 threads executed together | Basic scheduling unit |
| **SM** | Streaming Multiprocessor | Physical execution unit (20 on RTX 3050) |

---

## Thread Hierarchy

```
Grid (Entire GPU)
 │
 ├─ Block (0,0) ────── Block (0,1) ────── Block (0,2)
 │   │                   │                   │
 │   ├─ Thread(0,0)      ├─ Thread(0,0)     ├─ Thread(0,0)
 │   ├─ Thread(0,1)      ├─ Thread(0,1)     ├─ Thread(0,1)
 │   ├─ Thread(1,0)      ├─ Thread(1,0)     ├─ Thread(1,0)
 │   └─ ...              └─ ...             └─ ...
 │
 ├─ Block (1,0) ────── Block (1,1) ────── Block (1,2)
 └─ ...
```

**Key Point**: Each thread has unique ID within block, each block has unique ID within grid

---

## Memory Hierarchy (Fast → Slow)

| Memory Type | Size | Speed | Scope | Usage |
|-------------|------|-------|-------|-------|
| **Registers** | 64K per SM | 1 cycle | Per thread | Local variables |
| **Shared Memory** | 100KB per SM | ~20 cycles | Per block | Tile caching |
| **L2 Cache** | 2 MB | ~100 cycles | Device | Automatic |
| **Global Memory** | 8 GB | 200-400 cycles | Device | Main data |
| **Host Memory** | 16+ GB | 1000+ cycles | CPU | System RAM |

**Optimization Goal**: Use faster memory whenever possible!

---

## CUDA Function Qualifiers

```cuda
__global__ void kernel() { }    // Called by CPU, runs on GPU
__device__ void helper() { }    // Called by GPU, runs on GPU
__host__ void cpuFunc() { }     // Called by CPU, runs on CPU (default)
__host__ __device__ void both() { }  // Can run on both
```

---

## Built-in Variables

### Thread Identification
```cuda
// Inside a kernel:
int x = blockIdx.x * blockDim.x + threadIdx.x;  // Global X coordinate
int y = blockIdx.y * blockDim.y + threadIdx.y;  // Global Y coordinate
```

### Available Variables
- `threadIdx.x/y/z`: Thread index within block (0 to blockDim-1)
- `blockIdx.x/y/z`: Block index within grid (0 to gridDim-1)
- `blockDim.x/y/z`: Number of threads per block dimension
- `gridDim.x/y/z`: Number of blocks per grid dimension

---

## Kernel Launch Syntax

```cuda
// 1D Grid of 1D Blocks
kernel<<<numBlocks, threadsPerBlock>>>(params);

// 2D Grid of 2D Blocks (our project uses this)
dim3 blocks(bx, by);
dim3 threads(tx, ty);
kernel<<<blocks, threads>>>(params);

// With shared memory
kernel<<<blocks, threads, sharedMemBytes>>>(params);
```

**Example from project:**
```cuda
dim3 blockSize(16, 16);  // 256 threads per block
int bx = (width + 15) / 16;   // Ceiling division
int by = (height + 15) / 16;
dim3 gridSize(bx, by);
myKernel<<<gridSize, blockSize>>>(device_img, width, height);
```

---

## Memory Management Functions

```cuda
// Allocate
cudaMalloc(&d_ptr, size);              // Device only
cudaMallocManaged(&d_ptr, size);       // Unified (auto-migrates)

// Transfer
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);    // CPU → GPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);    // GPU → CPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);  // GPU → GPU

// Free
cudaFree(d_ptr);

// Synchronize
cudaDeviceSynchronize();  // Wait for GPU to finish
```

---

## Synchronization

```cuda
// Within a block (threads wait for each other)
__syncthreads();

// Between kernels (CPU waits for GPU)
cudaDeviceSynchronize();
```

**Critical**: `__syncthreads()` only works within a block, not across blocks!

---

## Shared Memory Usage Pattern

```cuda
__global__ void kernel(float *input, float *output) {
    // 1. Declare shared memory
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    // 2. Load data cooperatively (all threads help)
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = input[globalIdx];
    
    // 3. Wait for all threads to finish loading
    __syncthreads();
    
    // 4. Process from shared memory (FAST!)
    float result = computeFromTile(tile, tx, ty);
    
    // 5. Wait before next phase
    __syncthreads();
    
    // 6. Write result
    output[globalIdx] = result;
}
```

---

## Performance Optimization Checklist

### ✅ Do's
1. **Coalesce memory accesses**: Consecutive threads access consecutive memory
2. **Use shared memory**: 100x faster than global memory
3. **Maximize occupancy**: More active warps = hide latency
4. **Minimize divergence**: Keep threads in a warp on same path
5. **Check bounds**: `if (x >= width || y >= height) return;`

### ❌ Don'ts
1. **Don't use too many registers**: Causes spilling (slow)
2. **Don't have too much shared memory**: Reduces occupancy
3. **Don't forget synchronization**: Race conditions
4. **Don't access memory randomly**: Non-coalesced = slow
5. **Don't use recursion**: Limited stack on GPU

---

## Common Patterns in This Project

### Pattern 1: Simple Image Processing
```cuda
__global__ void processImage(Rgb* output, Rgb* input, int width, int height) {
    // 1. Compute global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 2. Boundary check
    if (x >= width || y >= height) return;
    
    // 3. Compute linear index
    int idx = y * width + x;
    
    // 4. Process pixel
    output[idx] = processPixel(input[idx]);
}
```

### Pattern 2: Neighborhood Processing (Convolution)
```cuda
__global__ void convolve(Rgb* output, Rgb* input, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    Rgb sum = {0, 0, 0};
    int count = 0;
    
    // Iterate over neighborhood
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Check bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum.r += input[ny * width + nx].r;
                sum.g += input[ny * width + nx].g;
                sum.b += input[ny * width + nx].b;
                count++;
            }
        }
    }
    
    // Average
    output[y * width + x].r = sum.r / count;
    output[y * width + x].g = sum.g / count;
    output[y * width + x].b = sum.b / count;
}
```

### Pattern 3: Shared Memory Tiling
```cuda
__global__ void tiledConvolve(Rgb* output, Rgb* input, int width, int height) {
    extern __shared__ Rgb tile[];
    
    // Thread and block indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    // Global position
    int x = bx * TILE_SIZE + tx;
    int y = by * TILE_SIZE + ty;
    
    // Load tile (including halo)
    if (x < width && y < height)
        tile[ty * TILE_SIZE + tx] = input[y * width + x];
    else
        tile[ty * TILE_SIZE + tx] = {0, 0, 0};
    
    __syncthreads();  // Wait for all loads
    
    // Compute from shared memory
    if (x < width && y < height) {
        Rgb result = computeFromTile(tile, tx, ty);
        output[y * width + x] = result;
    }
}
```

---

## Project-Specific Formulas

### 1. Sobel Edge Detection
```
Gx = [-1  0  1]    Gy = [-1 -2 -1]
     [-2  0  2]         [ 0  0  0]
     [-1  0  1]         [ 1  2  1]

Magnitude = √(Gx² + Gy²)
Direction = atan2(Gy, Gx)
```

### 2. Gaussian Weight (KNN/NLM)
```
w(x,y,i,j) = exp(-||I(x,y) - I(i,j)||² / h²) × exp(-||(x,y) - (i,j)||² / σ²)
             ↑                                  ↑
          Intensity similarity              Spatial proximity
```

### 3. Image Indexing
```
2D[y][x] → 1D[y * width + x]
```

---

## RTX 3050 Specifications

| Specification | Value |
|---------------|-------|
| Compute Capability | 8.6 (sm_86) |
| CUDA Cores | 2560 |
| Streaming Multiprocessors | 20 |
| Memory | 8 GB GDDR6 |
| Memory Bandwidth | 224 GB/s |
| Max Threads per Block | 1024 |
| Max Threads per SM | 1536 |
| Shared Memory per SM | 100 KB |
| Registers per SM | 65536 |
| Warp Size | 32 threads |

---

## Speedup Calculations

### Amdahl's Law
```
Speedup = 1 / (SerialPortion + ParallelPortion/Processors)

Example:
- 95% parallel code
- 2560 cores
- Speedup = 1 / (0.05 + 0.95/2560) ≈ 19.6x
```

### Efficiency
```
Efficiency = Speedup / NumberOfProcessors

Goal: High efficiency (close to 1.0 = linear scaling)
```

---

## Debugging Tips

1. **Compile with debug info**: `nvcc -g -G kernel.cu`
2. **Use cuda-memcheck**: `cuda-memcheck ./program`
3. **Check errors**: 
   ```cuda
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       printf("Error: %s\n", cudaGetErrorString(err));
   }
   ```
4. **Print from kernel** (debug only):
   ```cuda
   if (threadIdx.x == 0 && blockIdx.x == 0)
       printf("Debug: value = %f\n", value);
   ```

---

## Compilation Commands

```bash
# Basic compilation
nvcc main.cu kernel.cu kernelcall.cu -o main

# With optimization
nvcc -O3 main.cu kernel.cu kernelcall.cu -o main

# Show PTX/resource usage
nvcc --ptxas-options=-v kernel.cu

# Specify architecture
nvcc -arch=sm_86 kernel.cu  # For RTX 3050

# With OpenCV (Linux)
nvcc main.cu kernel.cu kernelcall.cu -o main `pkg-config --cflags --libs opencv4`

# With CMake (recommended - already configured)
mkdir build && cd build
cmake ..
make
```

---

## Project Execution Examples

```bash
# Edge Detection
./main image.jpg edge_detect

# Blur (convolution size = 5)
./main image.jpg conv 5

# Optimized blur with shared memory
./main image.jpg shared_conv 5

# K-NN Denoising (kernel=7, h=0.5)
./main image.jpg knn 7 0.5

# Non-Local Means (kernel=7, block=2, h=0.1)
./main image.jpg nlm 7 2 0.1

# Pixelize (pixel size = 10)
./main image.jpg pixelize 10
```

---

## Key Metrics for Presentations

### Performance Comparison (512×512 image)
- **Convolution Blur**: CPU ~100ms → GPU ~5ms = **20x speedup**
- **KNN Denoising**: CPU ~500ms → GPU ~20ms = **25x speedup**
- **NLM Denoising**: CPU ~30s → GPU ~500ms = **60x speedup**

### Why Such Speedup?
1. **Parallelism**: 2 million operations simultaneously
2. **Memory Bandwidth**: 224 GB/s vs CPU's 50 GB/s
3. **Specialized Hardware**: Optimized for floating-point math
4. **Shared Memory**: 100x faster than main memory

---

## Important Concepts for Exams

1. **Thread Hierarchy**: Grid → Block → Thread
2. **Memory Hierarchy**: Registers → Shared → L2 → Global → Host
3. **Synchronization**: `__syncthreads()` within block, `cudaDeviceSynchronize()` for host
4. **Coalescing**: Consecutive threads → consecutive memory
5. **Occupancy**: Active warps / Max possible warps
6. **Warp**: 32 threads execute together (SIMT)
7. **Divergence**: Performance penalty when threads take different paths
8. **Shared Memory**: Fast, limited, requires synchronization
9. **CUDA Compilation**: NVCC separates host/device code
10. **Memory Transfer**: PCIe bottleneck, minimize transfers

---

## Quick Reference: File Extensions

- `.cu`: CUDA source file (contains kernels)
- `.cuh`: CUDA header file
- `.cpp`: C++ source (host only)
- `.h`: C++ header
- `.ptx`: PTX assembly (intermediate)
- `.cubin`: CUDA binary

---

**End of Cheatsheet**

This cheatsheet covers the essential GPU programming concepts needed for your project understanding and exams. Refer to `GPU_PROJECT_DOCUMENTATION.md` for detailed explanations!
