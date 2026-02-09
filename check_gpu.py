#!/usr/bin/env python3
"""
GPU Detection and Environment Validation Script
Run this to verify CUDA, cuDNN, and PyTorch are correctly configured.
"""

import sys
import os

def check_gpu():
    """Comprehensive GPU and CUDA detection."""
    print("=" * 70)
    print("GPU & CUDA CONFIGURATION CHECK")
    print("=" * 70)
    
    # 0. Check Python and PyTorch Environment
    print("\n0. Python & PyTorch Environment:")
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   Python Executable: {sys.executable}")
    
    try:
        import torch
        print(f"   ✓ PyTorch Version: {torch.__version__}")
        print(f"   ✓ PyTorch Location: {torch.__file__}")
    except ImportError as e:
        print(f"   ✗ PyTorch import failed: {e}")
        return False
    
    # 1. Check CUDA availability
    print("\n1. CUDA Runtime Check:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"   ✓ CUDA Version: {torch.version.cuda}")
        print(f"   ✓ cuDNN Version: {torch.backends.cudnn.version()}")
        device_count = torch.cuda.device_count()
        print(f"   ✓ GPU Count: {device_count}")
        
        for i in range(device_count):
            print(f"\n   GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Test GPU access
        print("\n   GPU Memory Test:")
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"   ✓ Successfully allocated tensor on GPU")
        print(f"   ✓ GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        del test_tensor
        torch.cuda.empty_cache()
        
    else:
        print(f"   ✗ CUDA NOT AVAILABLE")
        print("\n   Debugging CUDA Detection Issue:")
        
        # Check if CUDA libraries are findable
        print("\n   2. CUDA Library Search Paths:")
        cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', 'NOT SET'))
        print(f"      CUDA_HOME: {cuda_home}")
        
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'NOT SET')
        if 'cuda' in ld_library_path.lower():
            print(f"      LD_LIBRARY_PATH: (contains cuda paths)")
        else:
            print(f"      LD_LIBRARY_PATH: (NO cuda paths - THIS IS THE ISSUE)")
        
        print("\n   3. Checking for CUDA Runtime Libraries on System:")
        import subprocess
        try:
            # Try to find libcudart
            result = subprocess.run(['find', '/usr', '-name', 'libcudart.so*', '-type', 'f'],
                                  capture_output=True, text=True, timeout=5)
            if result.stdout:
                cuda_libs = result.stdout.strip().split('\n')
                print(f"      Found CUDA libraries:")
                for lib in cuda_libs[:3]:  # Show first 3
                    print(f"        - {lib}")
                if len(cuda_libs) > 3:
                    print(f"        ... and {len(cuda_libs) - 3} more")
            else:
                print(f"      No CUDA libraries found in /usr")
        except Exception as e:
            print(f"      Could not search for libraries: {e}")
        
        print("\n   4. Checking for CUDA in Common Locations:")
        common_cuda_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-12.6',
            '/usr/local/cuda-12.4',
            '/opt/cuda',
            '/software/cuda',
            '/apps/cuda',
        ]
        found_cuda = []
        for path in common_cuda_paths:
            if os.path.exists(path):
                found_cuda.append(path)
                print(f"      ✓ Found: {path}")
        
        if not found_cuda:
            print(f"      ✗ No CUDA installation found in common locations")
            print(f"         Try: find / -name 'cuda' -type d 2>/dev/null | head -5")
        
        return False
    
    # 2. Check CUDA Environment Variables
    print("\n2. CUDA Environment Variables:")
    cuda_env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
    for var in cuda_env_vars:
        value = os.environ.get(var, "NOT SET")
        if var == 'LD_LIBRARY_PATH' or var == 'PATH':
            if 'cuda' in value.lower():
                print(f"   ✓ {var}: (contains 'cuda')")
            else:
                print(f"   ⚠ {var}: (no 'cuda' detected)")
        else:
            print(f"   {'✓' if value != 'NOT SET' else '✗'} {var}: {value if value != 'NOT SET' else 'NOT SET'}")
    
    # 3. Check Critical Dependencies
    print("\n3. Critical Dependencies:")
    dependencies = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'opencv': 'cv2',
        'segmentation_models_pytorch': 'segmentation_models_pytorch',
        'numpy': 'numpy',
        'scipy': 'scipy',
    }
    
    all_installed = True
    for name, import_name in dependencies.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✓ {name}: {version}")
        except ImportError:
            print(f"   ✗ {name}: NOT INSTALLED")
            all_installed = False
    
    # 4. Check nvidia-smi (if available)
    print("\n4. GPU Detection via nvidia-smi:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                print(f"   ✓ GPU {i}: {line}")
        else:
            print(f"   ⚠ nvidia-smi available but returned error")
    except FileNotFoundError:
        print(f"   ⚠ nvidia-smi not found in PATH")
    except Exception as e:
        print(f"   ⚠ nvidia-smi check failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    if cuda_available and all_installed:
        print("✓ GPU SETUP LOOKS GOOD!")
        print("\nYou can proceed with GPU training:")
        print("  python -m models.efficicentnet_train_smp --epochs 50 --batch-size 16")
        print("=" * 70)
        return True
    else:
        print("✗ GPU SETUP ISSUES DETECTED")
        print("\nIMPORTANT: The cluster may need CUDA environment setup")
        print("\nTry these commands on the cluster:")
        print("  1. module load cuda  # if available")
        print("  2. which nvcc        # check if CUDA compiler is available")
        print("  3. export CUDA_HOME=/path/to/cuda")
        print("  4. export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")
        print("\nThen re-run this script to verify.")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)

