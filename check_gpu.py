#!/usr/bin/env python3
"""
GPU Detection and Environment Validation Script
Run this to verify CUDA, cuDNN, and PyTorch are correctly configured.
"""

import sys

def check_gpu():
    """Comprehensive GPU and CUDA detection."""
    print("=" * 70)
    print("GPU & CUDA CONFIGURATION CHECK")
    print("=" * 70)
    
    # 1. Check CUDA availability
    print("\n1. CUDA Runtime Check:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   ✓ PyTorch Version: {torch.__version__}")
        print(f"   ✓ CUDA Available: {cuda_available}")
        
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
            print("   ✗ CUDA NOT AVAILABLE - GPU training will not work!")
            return False
            
    except ImportError as e:
        print(f"   ✗ PyTorch import failed: {e}")
        return False
    
    # 2. Check CUDA Environment Variables
    print("\n2. CUDA Environment Variables:")
    import os
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
        print("\nTroubleshooting:")
        if not cuda_available:
            print("  1. CUDA not detected - check CUDA_HOME and PATH environment variables")
            print("  2. Run: export CUDA_HOME=/path/to/cuda (if on cluster)")
            print("  3. Add to LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")
        if not all_installed:
            print("  4. Install missing packages: pip install -r requirements.txt")
        print("\nThen re-run this script to verify.")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
