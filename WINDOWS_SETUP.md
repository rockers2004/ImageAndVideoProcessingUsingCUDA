# Windows Setup Guide for CUDA Image Processing Project

## Prerequisites

Your system:
- GPU: NVIDIA GeForce RTX 3050
- CUDA Version: 13.0 (from nvidia-smi)
- OS: Windows

## Required Software

### 1. CUDA Toolkit
You already have CUDA drivers (version 13.0), but you need the CUDA Toolkit for development.

**Download and Install:**
- Go to: https://developer.nvidia.com/cuda-downloads
- Select: Windows -> x86_64 -> Version (10/11) -> exe (network or local)
- Install CUDA Toolkit 12.x or 13.x
- Default installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

**Verify Installation:**
```powershell
nvcc --version
```

### 2. Visual Studio (C++ Compiler)
CUDA requires MSVC compiler from Visual Studio.

**Download and Install:**
- Visual Studio 2019 or 2022 Community Edition
- Download from: https://visualstudio.microsoft.com/downloads/
- During installation, select:
  - "Desktop development with C++"
  - Windows 10/11 SDK
  - MSVC v142 or v143 build tools

### 3. CMake
Build system generator.

**Download and Install:**
- Go to: https://cmake.org/download/
- Download Windows x64 Installer
- During installation: **Check "Add CMake to system PATH"**

**Verify Installation:**
```powershell
cmake --version
```

### 4. OpenCV
Computer vision library for image I/O and processing.

**Option A: Using vcpkg (Recommended)**
```powershell
# Install vcpkg
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install OpenCV (this will take time)
.\vcpkg install opencv4:x64-windows

# Integrate with Visual Studio
.\vcpkg integrate install
```

**Option B: Pre-built binaries**
- Download from: https://opencv.org/releases/
- Extract to `C:\opencv`
- Add to system PATH: `C:\opencv\build\x64\vc15\bin`
- Set environment variable: `OpenCV_DIR=C:\opencv\build`

## Project Setup

### Step 1: Update CMakeLists.txt
The project's CMakeLists.txt is configured for Linux. I'll create a Windows-compatible version.

### Step 2: Build the Project
```powershell
# Navigate to GPU source directory
cd C:\Users\Omsai\projects\ImageAndVideoProcessingUsingCUDA\src\gpu

# Create build directory
mkdir build
cd build

# Configure with CMake (if using vcpkg)
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# Or without vcpkg (if OpenCV is installed manually)
cmake .. -DOpenCV_DIR="C:/opencv/build"

# Build the project
cmake --build . --config Release
```

### Step 3: Run the Application
```powershell
# The executable will be in build/Release/main.exe
.\Release\main.exe <image_path> <algorithm> <parameters>
```

## Usage Examples

```powershell
# Edge Detection
.\Release\main.exe ..\..\..images\test.jpg edge_detect

# Non Local-Means Denoising
.\Release\main.exe ..\..\..images\test.jpg nlm 7 0.1

# K-Nearest Neighbors Denoising
.\Release\main.exe ..\..\..images\test.jpg knn 7 2 0.5

# Convolution Blurring
.\Release\main.exe ..\..\..images\test.jpg conv 5

# Pixelize
.\Release\main.exe ..\..\..images\test.jpg pixelize 10
```

## Troubleshooting

### CUDA Architecture Mismatch
If you get architecture errors, the CMakeLists.txt specifies `sm_35` which is old. Your RTX 3050 uses `sm_86`.

### OpenCV Not Found
Make sure `OpenCV_DIR` environment variable points to OpenCV's build directory or use vcpkg.

### NVCC Compiler Errors
Ensure Visual Studio C++ tools are installed and CUDA Toolkit is properly installed.

### Path Issues
Windows uses backslashes in paths. In PowerShell, you can use forward slashes or escape backslashes.

## Next Steps
1. Install all prerequisites in order
2. Run the build commands
3. Test with sample images from the `images` directory
