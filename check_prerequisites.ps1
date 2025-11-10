# CUDA Image Processing - Prerequisites Check Script
Write-Host ""
Write-Host "=== CUDA Image Processing Prerequisites Check ===" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: NVIDIA GPU
Write-Host "Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaInfo = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] NVIDIA GPU detected" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] NVIDIA GPU not detected" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "[FAIL] nvidia-smi not found" -ForegroundColor Red
    $allGood = $false
}

# Check 2: CUDA Toolkit
Write-Host ""
Write-Host "Checking CUDA Toolkit..." -ForegroundColor Yellow
try {
    $nvccCheck = Get-Command nvcc -ErrorAction Stop
    $nvccVersion = & nvcc --version 2>&1
    Write-Host "[OK] CUDA Toolkit installed" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] CUDA Toolkit not installed" -ForegroundColor Red
    Write-Host "       Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    $allGood = $false
}

# Check 3: CMake
Write-Host ""
Write-Host "Checking CMake..." -ForegroundColor Yellow
try {
    $cmakeCheck = Get-Command cmake -ErrorAction Stop
    $cmakeVersion = & cmake --version 2>&1
    Write-Host "[OK] CMake installed" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] CMake not installed" -ForegroundColor Red
    Write-Host "       Download from: https://cmake.org/download/" -ForegroundColor Yellow
    $allGood = $false
}

# Check 4: Visual Studio
Write-Host ""
Write-Host "Checking Visual Studio..." -ForegroundColor Yellow
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -property installationPath 2>&1
    if ($vsPath) {
        Write-Host "[OK] Visual Studio installed" -ForegroundColor Green
        $vcToolsPath = Join-Path $vsPath "VC\Tools\MSVC"
        if (Test-Path $vcToolsPath) {
            Write-Host "[OK] C++ build tools found" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] C++ build tools not found" -ForegroundColor Red
            Write-Host "       Run Visual Studio Installer and add Desktop C++ workload" -ForegroundColor Yellow
            $allGood = $false
        }
    }
} else {
    Write-Host "[FAIL] Visual Studio not installed" -ForegroundColor Red
    Write-Host "       Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
    $allGood = $false
}

# Check 5: OpenCV
Write-Host ""
Write-Host "Checking OpenCV..." -ForegroundColor Yellow
$opencvFound = $false

if (Test-Path "C:\vcpkg\vcpkg.exe") {
    Write-Host "[OK] vcpkg found" -ForegroundColor Green
    $vcpkgList = & C:\vcpkg\vcpkg.exe list opencv 2>&1
    if ($vcpkgList -match "opencv") {
        Write-Host "[OK] OpenCV installed via vcpkg" -ForegroundColor Green
        $opencvFound = $true
    }
}

$opencvPaths = @("C:\opencv", "C:\Program Files\opencv", "$env:OPENCV_DIR")
foreach ($path in $opencvPaths) {
    if ($path -and (Test-Path $path)) {
        Write-Host "[OK] OpenCV found at: $path" -ForegroundColor Green
        $opencvFound = $true
        break
    }
}

if (-not $opencvFound) {
    Write-Host "[FAIL] OpenCV not found" -ForegroundColor Red
    Write-Host "       Install via vcpkg (recommended) or download from opencv.org" -ForegroundColor Yellow
    $allGood = $false
}

# Summary
Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "[SUCCESS] All prerequisites installed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  cd src\gpu" -ForegroundColor Gray
    Write-Host "  mkdir build" -ForegroundColor Gray
    Write-Host "  cd build" -ForegroundColor Gray
    Write-Host "  cmake .." -ForegroundColor Gray
    Write-Host "  cmake --build . --config Release" -ForegroundColor Gray
} else {
    Write-Host "[ACTION REQUIRED] Missing prerequisites detected" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install missing components (see WINDOWS_SETUP.md)" -ForegroundColor Yellow
}
Write-Host ""
