# ST-VAD Framework v2.0 - Update Summary

## What Changed

Based on your file structure, I've updated the ST-VAD framework to use the correct paths for your installation.

## Your File Structure

```
/home/yizhou/Mprojects/VAD/TubeletGraph/
├── stvad_framework.py          # ← Place the new framework here
├── quick_run.py                # ✓ Your existing file
├── configs/
│   └── default.yaml
├── annotate/
│   └── vlm_mask_grounded.py    # ✓ Your existing file
└── TubeletGraph/
    ├── run.py
    └── vlm/
        └── prompt_vlm.py       # ✓ Your existing file
```

## Key Updates in Framework v2.0

### 1. Automatic Path Resolution
The framework now automatically detects its location and constructs correct paths to:
- `annotate/vlm_mask_grounded.py` (Stage 1)
- `quick_run.py` (Stage 2)
- `TubeletGraph/vlm/prompt_vlm.py` (Stage 3)

### 2. Script Verification
On startup, the framework verifies all required scripts exist and shows helpful error messages if anything is missing.

### 3. Correct Script Calls

**Stage 1 - Object Grounding:**
```python
# OLD (incorrect)
cmd = ["python", "generate_mask_grounded.py", ...]

# NEW (correct for your structure)
cmd = ["python", "/path/to/TubeletGraph/annotate/vlm_mask_grounded.py", ...]
```

**Stage 2 - Object Tracking:**
```python
# Uses your existing quick_run.py
cmd = ["python", "/path/to/TubeletGraph/quick_run.py", ...]
```

**Stage 3 - State Analysis:**
```python
# OLD (incorrect)
cmd = ["python", "prompt_vlm_compatible.py", ...]

# NEW (correct for your structure)
cmd = ["python", "/path/to/TubeletGraph/TubeletGraph/vlm/prompt_vlm.py", ...]
```

### 4. Enhanced Error Handling

The framework now:
- Tries multiple possible mask file naming conventions
- Provides detailed error messages with file paths
- Shows exactly what's missing and where it should be

### 5. Better Debugging

With verbose mode (`-v`), you can see:
- Exact commands being executed
- Full paths to all scripts
- Detailed error messages
- Progress through each stage

## Installation

### Simple Installation

```bash
# 1. Copy framework to your TubeletGraph directory
cp stvad_framework_v2.py /home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py

# 2. Make it executable
chmod +x /home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py

# 3. Test it
cd /home/yizhou/Mprojects/VAD/TubeletGraph/
python stvad_framework.py analyze --help
```

## Usage Examples

### Basic Usage

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

python stvad_framework.py analyze \
    assets/example/PhysAD/0000.mp4 \
    -c configs/default.yaml \
    -v
```

### What You'll See

With verbose mode, you'll see the actual paths being used:

```
✅ ST-VAD Framework: Analyzing assets/example/PhysAD/0000.mp4
🔄 STAGE 1: Object Grounding
  Command: python /home/yizhou/Mprojects/VAD/TubeletGraph/annotate/vlm_mask_grounded.py -i assets/example/PhysAD/0000.mp4 --vlm gpt4v --target_fps 10 --output_dir output --auto
✅ Completed: Object grounding with VLM
✅ Frames saved to: output/JPEGImages/0000
✅ Mask saved to: output/Annotations/0000/0000000.png

🔄 STAGE 2: Object-Centric Tracking
  Command: python /home/yizhou/Mprojects/VAD/TubeletGraph/quick_run.py -c configs/default.yaml --input_dir output/JPEGImages/0000 --input_mask output/Annotations/0000/0000000.png --fps 10 --method Ours
✅ Completed: TubeletGraph tracking pipeline

🔄 STAGE 3 & 4: State Analysis and Anomaly Detection
  Command: python /home/yizhou/Mprojects/VAD/TubeletGraph/TubeletGraph/vlm/prompt_vlm.py -c configs/default.yaml -p custom-0000-Ours --sample_interval 10 --video_path assets/example/PhysAD/0000.mp4 --detect_anomalies
✅ Completed: VLM-based state analysis and anomaly detection
```

## Files Provided

1. **stvad_framework_v2.py** - Main framework with correct paths
2. **README_STVAD_v2.md** - Complete documentation
3. **INSTALL.md** - Installation guide specific to your setup
4. **MIGRATION_GUIDE.md** - Guide for upgrading from old version
5. **QUICKSTART.md** - Quick start guide

## Comparison: Old vs New

### Old Version Issues

```python
# Stage 1 called wrong script
"python generate_mask_grounded.py ..."  # ❌ Doesn't exist

# Stage 3 called wrong script  
"python prompt_vlm_compatible.py ..."   # ❌ Doesn't exist

# Hardcoded paths wouldn't work in your structure
```

### New Version Fixes

```python
# Stage 1 uses correct path
"python /home/yizhou/.../annotate/vlm_mask_grounded.py ..."  # ✅

# Stage 3 uses correct path
"python /home/yizhou/.../TubeletGraph/vlm/prompt_vlm.py ..."  # ✅

# Automatic path detection based on framework location
self.base_dir = osp.dirname(osp.abspath(__file__))
self.vlm_mask_script = osp.join(self.base_dir, "annotate", "vlm_mask_grounded.py")
```

## Testing Your Installation

### Step 1: Verify Files
```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Check all required files exist
ls stvad_framework.py                      # Should exist after installation
ls annotate/vlm_mask_grounded.py          # Your existing file
ls quick_run.py                            # Your existing file
ls TubeletGraph/vlm/prompt_vlm.py         # Your existing file
```

### Step 2: Test Framework
```bash
# Should show help without errors
python stvad_framework.py --help
```

### Step 3: Run Full Pipeline
```bash
# Test with verbose mode
python stvad_framework.py analyze \
    YOUR_VIDEO.mp4 \
    -c configs/default.yaml \
    -v
```

## Expected Behavior

### On Success

Each stage should complete with ✅:
1. Stage 1: Creates frames and masks
2. Stage 2: Generates tracking predictions  
3. Stage 3: Creates anomaly reports

### On Failure

If any script is missing, you'll see:
```
❌ Error: Missing required scripts:
   - vlm_mask_grounded.py (expected at: /home/yizhou/Mprojects/VAD/TubeletGraph/annotate/vlm_mask_grounded.py)

Current base directory: /home/yizhou/Mprojects/VAD/TubeletGraph

Expected structure:
  /home/yizhou/Mprojects/VAD/TubeletGraph/
  ├── annotate/vlm_mask_grounded.py
  ├── quick_run.py
  └── TubeletGraph/vlm/prompt_vlm.py
```

## Why This Failed Before

Your original error:
```
$ python VADagent/stvad_framework.py analyze ...
[Stage 1] Object Grounding...
  Running: python generate_mask_grounded.py ...  # ❌ Wrong filename
[Stage 2] Object Tracking...
  Running: python quick_run.py ...               # ✅ Correct
[Stage 3] State Change Analysis...
  Running: python prompt_vlm_compatible.py ...   # ❌ Wrong filename
```

The framework was calling:
- `generate_mask_grounded.py` instead of `annotate/vlm_mask_grounded.py`
- `prompt_vlm_compatible.py` instead of `TubeletGraph/vlm/prompt_vlm.py`

## What to Do Next

1. **Install:** Copy `stvad_framework_v2.py` to your TubeletGraph directory
2. **Test:** Run with verbose mode on a test video
3. **Use:** Process your videos with correct paths
4. **Customize:** Adjust parameters as needed

## Quick Commands

```bash
# Installation
cp stvad_framework_v2.py /home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py

# Testing
cd /home/yizhou/Mprojects/VAD/TubeletGraph/
python stvad_framework.py analyze video.mp4 -c configs/default.yaml -v

# Batch Processing
python stvad_framework.py batch ./videos/ -c configs/default.yaml -v
```

## Support

If you encounter issues:
1. Run with `-v` flag for detailed logging
2. Check INSTALL.md for troubleshooting
3. Verify all file paths match your structure
4. Check API keys are set correctly

---

**Updated:** February 2026  
**Version:** 2.0  
**Target Path:** `/home/yizhou/Mprojects/VAD/TubeletGraph/`
