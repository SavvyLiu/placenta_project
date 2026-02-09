#!/usr/bin/env python
"""
Advanced CUDA Detection Debugging Script
Helps diagnose why PyTorch can't find CUDA libraries
"""

import os
import subprocess
import sys
import ctypes
from pathlib import Path

print("=" * 70)
print("ADVANCED CUDA DEBUGGING")
print("=" * 70)

# 1. Check environment variables
print("\n1. ENVIRONMENT VARIABLES:")
cuda_home = os.environ.get('CUDA_HOME', 'NOT SET')
ld_lib_path = os.environ.get('LD_LIBRARY_PATH', 'NOT SET')
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')

print(f"   CUDA_HOME: {cuda_home}")
print(f"   LD_LIBRARY_PATH: {ld_lib_path[:100]}...")
print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")

# 2. Check if library exists and is accessible
print("\n2. CHECKING LIBCUDART.SO ACCESSIBILITY:")
cuda_lib_paths = [
    "/usr/local/lib/ollama/cuda_v12/lib64/libcudart.so.12",
    "/usr/local/lib/ollama/cuda_v12/lib/libcudart.so.12",
    "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so.13",
]

for lib_path in cuda_lib_paths:
    if os.path.exists(lib_path):
        print(f"   ✓ Found: {lib_path}")
        # Try to load with ctypes
        try:
            ctypes.CDLL(lib_path)
            print(f"     ✓ Successfully loaded with ctypes")
        except Exception as e:
            print(f"     ✗ Failed to load: {e}")
    else:
        # Check the pattern without version number
        pattern = lib_path.split('.12')[0] if '.12' in lib_path else lib_path.split('.13')[0]
        matches = subprocess.run(['find', pattern, '-name', 'libcudart.so*', '-type', 'f'],
                               capture_output=True, timeout=5)
        if matches.stdout:
            print(f"   Found variants of {lib_path}:")
            for match in matches.stdout.decode().strip().split('\n'):
                print(f"     - {match}")

# 3. Check PyTorch's view of things
print("\n3. PYTORCH INTERNALS:")
import torch

print(f"   PyTorch version: {torch.__version__}")
print(f"   PyTorch file: {torch.__file__}")

# Check if CUDA libraries are in PyTorch's lib directory
torch_cuda_libs = Path(torch.__file__).parent / "lib"
if torch_cuda_libs.exists():
    print(f"   PyTorch lib directory: {torch_cuda_libs}")
    cuda_libs = list(torch_cuda_libs.glob("*cuda*.so*"))
    if cuda_libs:
        print(f"   ✓ Found {len(cuda_libs)} CUDA libraries in PyTorch:")
        for lib in cuda_libs[:5]:  # Show first 5
            print(f"     - {lib.name}")
    else:
        print(f"   ✗ NO CUDA libraries found in PyTorch lib directory!")

# 4. Try to import torch-specific CUDA checks
print("\n4. TORCH CUDA MODULE CHECKS:")
try:
    from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home
    print(f"   torch.utils.cpp_extension.CUDA_HOME: {torch_cuda_home}")
except Exception as e:
    print(f"   ✗ Could not get CUDA_HOME from torch: {e}")

# 5. Check if NVCC can be found
print("\n5. NVCC COMPILER CHECK:")
try:
    result = subprocess.run(['which', 'nvcc'], capture_output=True, timeout=5)
    if result.returncode == 0:
        nvcc_path = result.stdout.decode().strip()
        print(f"   ✓ Found nvcc: {nvcc_path}")
        
        # Get version
        version = subprocess.run(['nvcc', '--version'], capture_output=True, timeout=5)
        print(f"   Version: {version.stdout.decode().split('release')[1].split(',')[0].strip() if 'release' in version.stdout.decode() else 'unknown'}")
    else:
        print(f"   ✗ nvcc not found in PATH")
except Exception as e:
    print(f"   ✗ Error checking nvcc: {e}")

# 6. Check nvidia-smi
print("\n6. NVIDIA-SMI CHECK:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
    if result.returncode == 0:
        # Parse first few lines
        lines = result.stdout.decode().split('\n')[:3]
        for line in lines:
            if line.strip():
                print(f"   {line}")
    else:
        print(f"   ✗ nvidia-smi failed")
except Exception as e:
    print(f"   ✗ Error running nvidia-smi: {e}")

# 7. Check for PyTorch pre-built CUDA stubs
print("\n7. PYTORCH CUDA STUBS:")
torch_dir = Path(torch.__file__).parent
stubs_dir = torch_dir / "lib" / "stubs"
if stubs_dir.exists():
    print(f"   ✓ Found stubs directory: {stubs_dir}")
    stubs = list(stubs_dir.glob("*.so*"))
    for stub in stubs[:5]:
        print(f"     - {stub.name}")
else:
    print(f"   ✗ No stubs directory found")

# 8. FINAL CHECK - Is PyTorch built for CUDA?
print("\n8. PYTORCH CUDA BUILD INFO:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.version.cuda: {torch.version.cuda}")
print(f"   torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
print(f"   torch.backends.cudnn.is_available(): {torch.backends.cudnn.is_available()}")

# 9. Recommendations
print("\n9. RECOMMENDATIONS:")
if not torch.cuda.is_available():
    print("""
   ✗ CUDA is NOT AVAILABLE to PyTorch. This usually means:
   
   A) PyTorch was installed without CUDA support
      → Solution: Reinstall PyTorch with explicit CUDA support:
      
      conda install pytorch::pytorch pytorch::pytorch-cuda=12.6 -c pytorch -c nvidia
      
   B) LD_LIBRARY_PATH is not pointing to correct CUDA libraries
      → Current setting: {}
      → Should include: /usr/local/lib/ollama/cuda_v12/lib64
      → Try: export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12/lib64:$LD_LIBRARY_PATH
      
   C) CUDA version mismatch
      → PyTorch expects: CUDA 12.6
      → System has: CUDA 12.8
      → Try installing PyTorch for CUDA 12.8 (may not be available)
      
   D) PyTorch conda package is CPU-only
      → Check: conda list | grep pytorch
      → Look for torch WITHOUT 'cuda' or 'cu126' in the name
      → If CPU-only, reinstall with CUDA variant
      
   NEXT STEPS:
   1. Run: conda list | grep pytorch  (check current packages)
   2. If needed, reinstall: conda remove pytorch -y && \\
      conda install pytorch::pytorch pytorch::pytorch-cuda=12.6 -c pytorch -c nvidia
   3. Re-run this script
   """.format(ld_lib_path[:100]))
else:
    print("   ✓ CUDA IS AVAILABLE! You're ready to train on GPU.")

print("\n" + "=" * 70)
