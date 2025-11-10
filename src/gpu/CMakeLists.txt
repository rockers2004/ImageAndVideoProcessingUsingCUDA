cmake_minimum_required(VERSION 3.18)
project(gpu_img_proc LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA architecture for RTX 3050 (Ampere architecture)
# sm_86 for RTX 3050, also supporting sm_75 and sm_70 for compatibility
set(CMAKE_CUDA_ARCHITECTURES 86 75 70)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DVERBOSE")

if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Add executable
add_executable(main 
    main.cu 
    kernel.cu 
    kernelcall.cu
)

# Set CUDA properties
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(main 
    ${OpenCV_LIBS}
    CUDA::cudart
)

# Print configuration info
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
