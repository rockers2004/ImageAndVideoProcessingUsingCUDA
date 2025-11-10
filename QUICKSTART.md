# Quick Start Guide - Windows Setup

## Current Status
✅ NVIDIA RTX 3050 GPU detected  
❌ CUDA Toolkit - **NEED TO INSTALL**  
❌ CMake - **NEED TO INSTALL**  
❌ Visual Studio - **NEED TO CHECK/INSTALL**  
❌ OpenCV - **NEED TO INSTALL**

## Installation Order

### 1. Install Visual Studio 2022 (30-60 minutes)
**Required for CUDA compilation**

1. Download: https://visualstudio.microsoft.com/downloads/
2. Get "Community" edition (free)
3. During installation, select: **"Desktop development with C++"**
4. This includes MSVC compiler needed by CUDA

### 2. Install CUDA Toolkit (10-20 minutes)
**Development tools for your NVIDIA GPU**

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select: Windows → x86_64 → 10 or 11 → exe (local)
3. Download (≈3GB) and run installer
4. Choose "Express Installation"
5. Restart PowerShell after installation
6. Verify: `nvcc --version`

### 3. Install CMake (5 minutes)
**Build system for the project**

1. Go to: https://cmake.org/download/
2. Download: "Windows x64 Installer"
3. During installation: **CHECK "Add CMake to system PATH"**
4. Restart PowerShell
5. Verify: `cmake --version`

### 4. Install OpenCV via vcpkg (30-60 minutes)
**Image processing library**

```powershell
# Install vcpkg (package manager)
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install OpenCV (will take time to compile)
.\vcpkg install opencv4:x64-windows

# Integrate with Visual Studio
.\vcpkg integrate install
```

## Build the Project

After installing all prerequisites:

```powershell
# Navigate to project
cd C:\Users\Omsai\projects\ImageAndVideoProcessingUsingCUDA

# Check prerequisites
.\check_prerequisites.ps1

# Build
cd src\gpu
mkdir build
cd build

# Configure (if using vcpkg)
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build . --config Release

# The executable will be at: .\Release\main.exe
```

## Test the Application

```powershell
# From build directory
cd C:\Users\Omsai\projects\ImageAndVideoProcessingUsingCUDA\src\gpu\build

# Test with images from the images folder
.\Release\main.exe ..\..\..\images\<image_name>.jpg edge_detect
```

## Available Algorithms

```powershell
# Edge Detection
.\Release\main.exe <image> edge_detect

# Non Local-Means Denoising  
.\Release\main.exe <image> nlm <kernel_size> <h_parameter>
# Example: .\Release\main.exe image.jpg nlm 7 0.1

# K-Nearest Neighbors Denoising
.\Release\main.exe <image> knn <kernel_size> <block_radius> <weight_decay>
# Example: .\Release\main.exe image.jpg knn 7 2 0.5

# Convolution Blur
.\Release\main.exe <image> conv <kernel_size>
# Example: .\Release\main.exe image.jpg conv 5

# Pixelize
.\Release\main.exe <image> pixelize <pixel_size>
# Example: .\Release\main.exe image.jpg pixelize 10
```

## Troubleshooting

### "nvcc not found" after CUDA installation
- Restart PowerShell/Terminal
- Check: `$env:PATH` should include CUDA bin directory
- Default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

### CMake can't find CUDA
- Make sure CUDA Toolkit is installed (not just drivers)
- Set environment variable: `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

### CMake can't find OpenCV
- Make sure vcpkg integrate was run
- Or manually set: `cmake .. -DOpenCV_DIR=C:/vcpkg/installed/x64-windows/share/opencv4`

### Build errors about sm_35
- This is already fixed in the updated CMakeLists.txt
- Your RTX 3050 uses compute capability 8.6 (sm_86)

## Estimated Total Time
- **With fast internet**: 1.5 - 2.5 hours
- **Including build time**: 2 - 3 hours total

## Need Help?
1. Run `.\check_prerequisites.ps1` to see what's missing
2. Check `WINDOWS_SETUP.md` for detailed instructions
3. Check individual component websites for installation issues
