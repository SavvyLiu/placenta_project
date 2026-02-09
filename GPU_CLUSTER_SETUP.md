# GPU Cluster Setup Guide - CUDA Detection Fix

## Problem Summary
Your cluster has:
- ✓ PyTorch 2.9.1+cu126 installed
- ✓ CUDA & cuDNN packages installed
- ✗ **CUDA environment NOT initialized** → GPU not detected by PyTorch

## Root Cause
PyTorch can't find the CUDA libraries because:
1. `CUDA_HOME` environment variable not set
2. `LD_LIBRARY_PATH` doesn't include CUDA library paths
3. CUDA runtime not initialized in shell environment

## Solution: Setup CUDA Environment

### Step 1: Find Your CUDA Installation
On the cluster, run:
```bash
find /usr -name "cuda" -type d 2>/dev/null | head -5
find /opt -name "cuda" -type d 2>/dev/null | head -5
which nvidia-smi
nvcc --version
```

Expected output:
```
/usr/local/cuda
/usr/local/cuda-12.6
```

### Step 2: Add CUDA to Your Shell Environment

**Option A: One-Time (for current session only)**
```bash
# Find your CUDA path first (from Step 1)
export CUDA_HOME=/usr/local/cuda-12.6  # <-- Adjust path if needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify setup
python check_gpu.py
```

**Option B: Permanent (add to ~/.bashrc)**
```bash
# Edit ~/.bashrc
nano ~/.bashrc

# Add these lines at the end:
export CUDA_HOME=/usr/local/cuda-12.6  # <-- Adjust path!
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Save and exit (Ctrl+X, Y, Enter)

# Apply changes
source ~/.bashrc
python check_gpu.py
```

**Option C: Using Module System (if available)**
```bash
# Check what CUDA modules are available
module avail cuda

# Load CUDA module
module load cuda/12.6  # or whatever version is available

# Verify
python check_gpu.py
```

### Step 3: Install Project Dependencies

Now that CUDA is set up, install the project requirements:

```bash
cd ~/Histology_AI/placenta_project
git pull origin main

# Install dependencies (without specifying PyTorch version)
pip install -r requirements.txt --no-cache-dir
```

### Step 4: Verify GPU Detection

```bash
python check_gpu.py
```

**Expected Output:**
```
======================================================================
GPU & CUDA CONFIGURATION CHECK
======================================================================

1. CUDA Runtime Check:
   ✓ PyTorch Version: 2.9.1+cu126
   ✓ CUDA Available: True
   ✓ CUDA Version: 12.6
   ✓ cuDNN Version: 90501
   ✓ GPU Count: 1

   GPU 0:
      Name: NVIDIA A40
      Memory: 48.00 GB

   GPU Memory Test:
   ✓ Successfully allocated tensor on GPU
   ✓ GPU Memory Usage: 3.81 GB

...

✓ GPU SETUP LOOKS GOOD!
```

## Step 5: Start Training

Once GPU is detected, you can train:

```bash
# Quick test
python -m models.efficicentnet_train_smp \
    --epochs 5 \
    --batch-size 16 \
    --subset-size 20

# Full training
python -m models.efficicentnet_train_smp \
    --epochs 50 \
    --batch-size 16 \
    --augment
```

---

## Cluster Job Script Example

**File: `train.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=placenta-efficientnet
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu

# IMPORTANT: Set CUDA environment on every job
export CUDA_HOME=/usr/local/cuda-12.6  # <-- Adjust path if needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate Python environment (if using venv or conda)
source /path/to/venv/bin/activate  # or: conda activate placenta

# Go to project directory
cd ~/Histology_AI/placenta_project

# Verify CUDA before training
echo "Checking GPU availability..."
python check_gpu.py

if [ $? -eq 0 ]; then
    echo "GPU detected! Starting training..."
    python -m models.efficicentnet_train_smp \
        --epochs 50 \
        --batch-size 16 \
        --augment \
        --early-stopping-patience 10
else
    echo "GPU not detected. Check error messages above."
    exit 1
fi
```

**Submit job:**
```bash
sbatch train.sh
```

---

## What Changed in requirements.txt

The requirements.txt has been updated to:
1. **Not pin PyTorch version** - lets cluster's existing PyTorch 2.9.1+cu126 be used
2. **Not pin CUDA package versions** - lets pip resolve compatible versions
3. **Keep all other dependencies** - numpy, opencv, segmentation models, etc.

This avoids version conflicts while ensuring GPU support.

---

## Troubleshooting

### "CUDA NOT AVAILABLE" when running check_gpu.py
1. ✓ Did you set `CUDA_HOME`? → `echo $CUDA_HOME`
2. ✓ Did you set `LD_LIBRARY_PATH`? → `echo $LD_LIBRARY_PATH | grep cuda`
3. ✓ Is CUDA actually installed? → `which nvcc`
4. ✓ Are GPU utilities available? → `nvidia-smi`

### "Cannot find libcudart.so.12"
```bash
# Add to your shell environment setup
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

### "No module named 'torch'"
```bash
pip install --upgrade -r requirements.txt
```

### "RuntimeError: CUDA out of memory"
Reduce batch size:
```bash
python -m models.efficicentnet_train_smp \
    --epochs 50 \
    --batch-size 8  # <-- Reduced from 16
```

---

## Key Points

✓ **CUDA 12.6** is required (cluster has this)
✓ **PyTorch 2.9.1+cu126** is already installed (use as-is)
✓ **CUDA environment variables** are NOT automatically set (you must set them)
✓ **Check GPU** with `python check_gpu.py` before training

---

## Quick Copy-Paste Setup (Minimal)

```bash
# Step 1: Set CUDA (adjust path if different!)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Step 2: Navigate and update
cd ~/Histology_AI/placenta_project
git pull origin main
pip install -r requirements.txt --no-cache-dir --no-deps

# Step 3: Verify
python check_gpu.py

# Step 4: Train!
python -m models.efficicentnet_train_smp --epochs 50 --batch-size 16
```

If this doesn't work, share the output of `python check_gpu.py` and we can debug further!

