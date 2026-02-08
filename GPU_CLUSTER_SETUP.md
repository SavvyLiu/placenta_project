# GPU Cluster Setup Guide

## Problem
The cluster cannot detect the GPU when running training scripts.

## Solution

### 1. Install Dependencies with Correct CUDA Version

The `requirements.txt` has been updated with CUDA 12.6 packages that match your working cluster environment.

**On the Cluster:**

```bash
cd ~/Histology_AI/placenta_project

# Option A: Fresh environment (recommended)
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Option B: If you have conda
conda create -n placenta-gpu python=3.10
conda activate placenta-gpu
pip install -r requirements.txt --no-cache-dir
```

### 2. Verify GPU Detection

After installation, run the GPU validation script:

```bash
python check_gpu.py
```

**Expected Output:**
```
======================================================================
GPU & CUDA CONFIGURATION CHECK
======================================================================

1. CUDA Runtime Check:
   ✓ PyTorch Version: 2.6.0
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

You can proceed with GPU training:
  python -m models.efficicentnet_train_smp --epochs 50 --batch-size 16
```

### 3. If GPU Still Not Detected

#### Check CUDA Environment Variables
```bash
# On cluster, check if CUDA is properly set up:
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# If CUDA_HOME is not set, find CUDA:
find /usr -name "cuda" -type d 2>/dev/null | head -5
```

#### Set CUDA Paths (if needed)
**Add to your `~/.bashrc` or cluster job script:**

```bash
# Adjust paths based on your cluster's CUDA installation
export CUDA_HOME=/usr/local/cuda-12.6  # or wherever CUDA is installed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# For conda environments:
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available
```

#### Verify Paths
```bash
nvcc --version       # Should show CUDA 12.6
nvidia-smi          # Should list GPUs
```

### 4. Cluster Job Script Example

**File: `train.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=placenta-gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH --partition=gpu

# Set up CUDA environment
module load cuda/12.6  # or whatever module your cluster uses
export CUDA_VISIBLE_DEVICES=0

# Activate environment
source /path/to/venv/bin/activate  # if using venv
# OR
# conda activate placenta-gpu      # if using conda

# Navigate to project
cd ~/Histology_AI/placenta_project

# Verify GPU
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
    echo "GPU not detected. Check above error messages."
    exit 1
fi
```

**Run the job:**
```bash
sbatch train.sh
```

### 5. Critical Packages for GPU

These **must** be installed correctly for GPU detection:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.6.0 | Core PyTorch |
| `nvidia-cuda-runtime-cu12` | 12.6.77 | CUDA runtime |
| `nvidia-cudnn-cu12` | 9.5.1.17 | cuDNN library |
| `nvidia-nccl-cu12` | 2.21.5 | GPU communication |
| `nvidia-cuda-cupti-cu12` | 12.6.80 | CUDA profiling |

**Verify these are installed:**
```bash
pip list | grep nvidia
```

Should show all nvidia-* packages with cu12 in the name.

### 6. Troubleshooting Checklist

- [ ] GPU is physically present: `nvidia-smi`
- [ ] CUDA is available: `nvcc --version`
- [ ] PyTorch recognizes CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] All nvidia packages installed: `pip list | grep nvidia`
- [ ] No version conflicts: `pip check`
- [ ] CUDA_HOME is set: `echo $CUDA_HOME`
- [ ] LD_LIBRARY_PATH includes CUDA: `echo $LD_LIBRARY_PATH | grep cuda`

### 7. Common Issues & Fixes

**"RuntimeError: CUDA out of memory"**
→ Reduce batch size: `--batch-size 8` instead of 16

**"Could not load dynamic library 'libcudart.so.12'"**
→ Add to environment:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

**"NVIDIA driver" version mismatch**
→ Check driver version: `nvidia-smi | grep Driver`
→ Ensure PyTorch version is compatible with driver

**"ImportError: cannot import name 'cuda'"**
→ Reinstall PyTorch:
```bash
pip install --force-reinstall torch==2.6.0
```

### 8. Quick Start on Cluster

After setup, to start training:

```bash
# Verify GPU works
python check_gpu.py

# Start training with optimal settings for A40
python -m models.efficicentnet_train_smp \
    --epochs 50 \
    --batch-size 16 \
    --augment \
    --early-stopping-patience 10 \
    --lr-patience 5

# Monitor in another terminal
watch -n 2 nvidia-smi
```

### 9. Cluster-Specific Notes

**If using `conda` environment on cluster:**
```bash
# Create new environment with specific Python version
conda create -n placenta python=3.10 -y
conda activate placenta

# Install PyTorch with CUDA 12.6 support
conda install pytorch::pytorch pytorch::torchvision torchaudio pytorch::pytorch-cuda=12.6 -c pytorch -c nvidia

# Install remaining requirements
pip install -r requirements.txt --no-deps  # --no-deps to avoid conflicts
```

**If cluster uses module system (LMOD):**
```bash
# Check available modules
module avail cuda
module avail pytorch

# Load modules
module load cuda/12.6
module load pytorch/2.6.0  # if available
# OR install via pip as shown above
```

---

## Key Difference: CUDA 12.4 vs 12.6

Your current setup had **CUDA 12.4** packages, but the working cluster setup uses **CUDA 12.6**.

**This mismatch was preventing GPU detection.**

The updated `requirements.txt` now includes:
- ✓ All CUDA 12.6 packages
- ✓ PyTorch 2.6.0 (compatible with CUDA 12.6)
- ✓ Correct numpy version (1.26.4)
- ✓ All supporting libraries

---

## After Installation: Start Training

```bash
# Test on subset first
python -m models.efficicentnet_train_smp --epochs 5 --subset-size 20 --batch-size 16

# Full training if subset works
python -m models.efficicentnet_train_smp --epochs 50 --batch-size 16 --augment
```

**Expected speed with GPU:**
- ~5-10 seconds per epoch (vs 120 seconds before GPU setup)
- Total 50 epochs: ~5-10 minutes (vs ~100 minutes without GPU)

---

If issue persists after following this guide, please share output of:
```bash
python check_gpu.py
nvidia-smi
pip list | grep nvidia
python -c "import torch; print(torch.cuda.is_available())"
```
