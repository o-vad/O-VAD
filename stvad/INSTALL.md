# ST-VAD Framework Installation Guide

## Your Current File Structure

Based on your paths, your TubeletGraph project is located at:
```
/home/yizhou/Mprojects/VAD/TubeletGraph/
```

With the following key files:
- `quick_run.py`
- `annotate/vlm_mask_grounded.py`
- `TubeletGraph/vlm/prompt_vlm.py`

## Installation Steps

### Step 1: Place the Framework File

Place `stvad_framework_v2.py` in your TubeletGraph root directory and rename it:

```bash
# Copy the framework to your TubeletGraph directory
cp stvad_framework_v2.py /home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py

# Make it executable
chmod +x /home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py
```

Your directory structure should now look like:
```
/home/yizhou/Mprojects/VAD/TubeletGraph/
├── stvad_framework.py          # ← NEW FILE (the framework)
├── quick_run.py                # ← Existing
├── annotate/
│   └── vlm_mask_grounded.py    # ← Existing
└── TubeletGraph/
    └── vlm/
        └── prompt_vlm.py       # ← Existing
```

### Step 2: Verify Installation

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Test that the framework can find all required scripts
python stvad_framework.py analyze --help
```

**Expected output:** You should see the help message. If you see an error about missing scripts, verify your file paths.

**If you see errors like:**
```
❌ Error: Missing required scripts:
   - vlm_mask_grounded.py (expected at: /path/to/annotate/vlm_mask_grounded.py)
```

Then verify the file exists:
```bash
ls /home/yizhou/Mprojects/VAD/TubeletGraph/annotate/vlm_mask_grounded.py
```

### Step 3: Set Up API Keys

```bash
# For GPT-4V (OpenAI)
export OPENAI_API_KEY="your-openai-api-key"

# OR for Claude (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# To make permanent, add to ~/.bashrc:
echo 'export OPENAI_API_KEY="***REMOVED-CREDENTIAL***"' >> ~/.zshrc
source ~/.zshrc
```

### Step 4: Test with Example Video

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Run on example video (adjust path if your example is elsewhere)
python stvad_framework.py analyze \
    assets/example/PhysAD/0000.mp4 \
    -c configs/default.yaml \
    -v
```

## Quick Reference

### File Paths (Verified for Your System)

The framework automatically uses these paths:

| Component | File Path |
|-----------|-----------|
| Framework | `/home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py` |
| Stage 1 | `/home/yizhou/Mprojects/VAD/TubeletGraph/annotate/vlm_mask_grounded.py` |
| Stage 2 | `/home/yizhou/Mprojects/VAD/TubeletGraph/quick_run.py` |
| Stage 3 | `/home/yizhou/Mprojects/VAD/TubeletGraph/TubeletGraph/vlm/prompt_vlm.py` |

### Basic Commands

All commands should be run from the TubeletGraph directory:

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Analyze single video
python stvad_framework.py analyze video.mp4 -c configs/default.yaml

# Batch process
python stvad_framework.py batch ./videos/ -c configs/default.yaml

# With verbose output (recommended for debugging)
python stvad_framework.py analyze video.mp4 -c configs/default.yaml -v
```

## Common Issues

### Issue 1: "Missing required scripts"

**Cause:** Framework can't find one of the component scripts.

**Solution:**
```bash
# Verify all files exist
ls /home/yizhou/Mprojects/VAD/TubeletGraph/annotate/vlm_mask_grounded.py
ls /home/yizhou/Mprojects/VAD/TubeletGraph/quick_run.py
ls /home/yizhou/Mprojects/VAD/TubeletGraph/TubeletGraph/vlm/prompt_vlm.py
```

If any are missing, check if they're in different locations. The framework uses relative paths from where `stvad_framework.py` is located.

### Issue 2: "ImportError" or "ModuleNotFoundError"

**Cause:** Missing Python dependencies.

**Solution:**
```bash
# Install required packages
pip install opencv-python pillow numpy torch
pip install openai anthropic  # For VLM APIs
```

### Issue 3: VLM API errors

**Cause:** API key not set or invalid.

**Solution:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="your-key-here"

# Test the API
python -c "import openai; print('API key is set')"
```

## Testing Your Installation

### Quick Test

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# This should show the help message
python stvad_framework.py --help

# This should show command-specific help
python stvad_framework.py analyze --help
```

### Full Pipeline Test

If you have an example video:

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Run full pipeline with verbose output
python stvad_framework.py analyze \
    /path/to/test/video.mp4 \
    -c configs/default.yaml \
    -v
```

**Expected behavior:**
1. ✅ Stage 1 completes: Creates frames and masks
2. ✅ Stage 2 completes: Generates tracking predictions
3. ✅ Stage 3 completes: Creates anomaly report
4. 📄 Output files created in `output/` directory

## Next Steps

After successful installation:

1. **Read the full documentation:** See `README_STVAD_v2.md`
2. **Try the examples:** Process a few test videos
3. **Adjust parameters:** Experiment with `--fps`, `--sample_interval`, etc.
4. **Integrate into your workflow:** Use batch processing or JSON outputs

## Getting Help

If you encounter issues:

1. Run with `-v` flag to see detailed logs
2. Check that all file paths are correct
3. Verify API keys are set
4. Check the troubleshooting section in README

## Quick Start Command

Copy and paste this to run your first analysis:

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/ && \
python stvad_framework.py analyze \
    YOUR_VIDEO.mp4 \
    -c configs/default.yaml \
    -v
```

Replace `YOUR_VIDEO.mp4` with the path to your video file.

---

**Installation Path:** `/home/yizhou/Mprojects/VAD/TubeletGraph/`  
**Framework File:** `stvad_framework.py`  
**Version:** 2.0
