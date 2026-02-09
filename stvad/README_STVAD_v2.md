# ST-VAD Framework: Spatio-Temporal Video Anomaly Detection

A comprehensive pipeline for detecting anomalies in videos using Vision Language Models (VLMs), object tracking, and step-by-step reasoning.

## File Structure

This framework expects the following directory structure:

```
/home/yizhou/Mprojects/VAD/TubeletGraph/
├── stvad_demo.py          # Main framework (this file - place here)
├── quick_run.py                # Stage 2: Object tracking
├── configs/
│   └── default.yaml            # TubeletGraph configuration
├── annotate/
│   └── vlm_mask_grounded.py    # Stage 1: Object grounding
└── TubeletGraph/
    ├── run.py                  # TubeletGraph pipeline
    └── vlm/
        └── prompt_vlm.py       # Stage 2: State tracking
        └── prompt_vad.py       # Stage 3&4: State analysis & anomaly detection
```

## Installation

### 1. Place the Framework Script

Copy `stvad_framework.py` to your TubeletGraph root directory:

```bash
# Assuming you're in the directory where you downloaded the framework
cp stvad_framework.py /home/yizhou/Mprojects/VAD/TubeletGraph/

# Make it executable
chmod +x /home/yizhou/Mprojects/VAD/TubeletGraph/stvad_framework.py
```

### 2. Set up VLM API Keys

```bash
# For GPT-4V (OpenAI)
export OPENAI_API_KEY="sk-..."

# OR for Claude (Anthropic)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Verify Installation

The framework will automatically verify all required scripts exist when you run it:

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/
python stvad_framework.py analyze --help
```

If any scripts are missing, you'll see a helpful error message showing what's missing and where it should be.

## Architecture

```
Video Input
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Object Grounding                                   │
│ File: annotate/vlm_mask_grounded.py                         │
│ - VLM identifies objects in video                           │
│ - Generates segmentation masks                              │
│ - Extracts frames at target FPS                             │
└─────────────────────────────────────────────────────────────┘
    ↓
    ├─→ Frames: output/JPEGImages/{video_name}/
    └─→ Masks: output/Annotations/{video_name}/
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Object-Centric Tracking                            │
│ File: quick_run.py                                          │
│ - Creates custom dataset from frames and masks              │
│ - Runs TubeletGraph pipeline                                │
│ - Generates tracking predictions                            │
└─────────────────────────────────────────────────────────────┘
    ↓
    └─→ Predictions: {outdir}/custom-{video}-{method}/
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3 & 4: Analysis & Anomaly Detection                   │
│ File: TubeletGraph/vlm/prompt_vlm.py                        │
│ - Analyzes object state changes                             │
│ - Detects temporal patterns                                 │
│ - Step-by-step anomaly reasoning                            │
│ - Generates detailed reports                                │
└─────────────────────────────────────────────────────────────┘
    ↓
    └─→ Reports: output/anomaly_reports/
        ├─ {video}_report.json
        └─ {video}_summary.txt
```

## Usage

### Quick Start

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Analyze a single video with verbose output
python stvad_framework.py analyze \
    assets/example/PhysAD/0000.mp4 \
    -c configs/default.yaml \
    -v
```

### Basic Commands

```bash
# Single video analysis
python stvad_framework.py analyze <video_path> -c <config_path>

# Batch processing
python stvad_framework.py batch <video_directory> -c <config_path>

# With custom settings
python stvad_framework.py analyze <video_path> \
    -c <config_path> \
    --vlm openai \
    --fps 10 \
    --sample_interval 10 \
    --method Ours \
    -o ./results \
    -v
```

### Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `-c, --config` | (required) | Path to TubeletGraph config file |
| `-o, --output` | `output` | Output directory |
| `--vlm` | `gpt4v` | VLM model (gpt4v, claude) |
| `--fps` | `10` | Target FPS for frame extraction |
| `--sample_interval` | `10` | Sampling interval for analysis |
| `--method` | `Ours` | Tracking method name |
| `--no-auto` | `False` | Disable automatic object detection |
| `-v, --verbose` | `False` | Enable verbose output |

## Examples

### Example 1: Analyze Physical Anomaly Video

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

python stvad_framework.py analyze \
    assets/example/PhysAD/0000.mp4 \
    -c configs/default.yaml \
    --vlm gpt4v \
    --fps 10 \
    -v
```

**Expected Output:**
```
ℹ️  ============================================================
ℹ️  ST-VAD Framework: Analyzing assets/example/PhysAD/0000.mp4
ℹ️  ============================================================
🔄 STAGE 1: Object Grounding
  Command: python /home/yizhou/Mprojects/VAD/TubeletGraph/annotate/vlm_mask_grounded.py -i assets/example/PhysAD/0000.mp4 --vlm gpt4v --target_fps 10 --output_dir output --auto
✅ Completed: Object grounding with VLM
✅ Frames saved to: output/JPEGImages/0000
✅ Mask saved to: output/Annotations/0000/0000000.png

🔄 STAGE 2: Object-Centric Tracking
  Command: python /home/yizhou/Mprojects/VAD/TubeletGraph/quick_run.py -c configs/default.yaml --input_dir output/JPEGImages/0000 --input_mask output/Annotations/0000/0000000.png --fps 10 --method Ours
✅ Completed: TubeletGraph tracking pipeline
✅ Tracking results: custom-0000-Ours

🔄 STAGE 3 & 4: State Analysis and Anomaly Detection
  Command: python /home/yizhou/Mprojects/VAD/TubeletGraph/TubeletGraph/vlm/prompt_vlm.py -c configs/default.yaml -p custom-0000-Ours --sample_interval 10 --video_path assets/example/PhysAD/0000.mp4 --detect_anomalies
✅ Completed: VLM-based state analysis and anomaly detection
✅ Anomaly report saved to: output/anomaly_reports/0000_report.json

================================================================================
ST-VAD ANOMALY DETECTION REPORT
================================================================================
Video: 0000
Path: assets/example/PhysAD/0000.mp4
================================================================================

DETECTION RESULT:
  Anomaly Detected: YES ❌
  Number of Anomalies: 1
...
```

### Example 2: Batch Process Multiple Videos

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/

# Create a directory with videos
mkdir test_videos
cp video1.mp4 video2.mp4 video3.mp4 test_videos/

# Process all videos
python stvad_framework.py batch test_videos/ \
    -c configs/default.yaml \
    --fps 10 \
    -v
```

### Example 3: High Quality Analysis

```bash
python stvad_framework.py analyze video.mp4 \
    -c configs/default.yaml \
    --fps 20 \
    --sample_interval 5 \
    --vlm gpt4v \
    -o high_quality_results \
    -v
```

## Output Structure

After running the framework, outputs are organized as:

```
output/
├── JPEGImages/              # Extracted video frames
│   └── {video_name}/
│       ├── 0000000.jpg
│       ├── 0000001.jpg
│       └── ...
│
├── Annotations/             # Object segmentation masks
│   └── {video_name}/
│       └── 0000000.png
│
├── tracking_results/        # TubeletGraph tracking outputs
│
├── anomaly_reports/         # Anomaly detection results
│   ├── {video}_report.json      # Machine-readable JSON
│   ├── {video}_summary.txt      # Human-readable summary
│   └── batch_summary.txt        # (for batch processing)
│
└── visualizations/          # Visual outputs (if enabled)
```

## Report Formats

### JSON Report (`*_report.json`)

```json
{
  "prediction_name": "custom-0000-Ours",
  "anomaly_detected": true,
  "num_anomalies": 1,
  "reasoning_trace": "Step 1: Identified events...\nStep 2: Found anomalies...",
  "identified_events": [
    "Object movement at frames 10-45",
    "State change at frame 67"
  ],
  "anomalous_transitions": []
}
```

### Text Summary (`*_summary.txt`)

```
================================================================================
ST-VAD ANOMALY DETECTION REPORT
================================================================================
Video: 0000
Path: /path/to/0000.mp4
================================================================================

DETECTION RESULT:
  Anomaly Detected: YES ❌
  Number of Anomalies: 1

================================================================================
REASONING TRACE:
================================================================================
Step 1: Identified events in video:
  - Object moved from position A to B
  - State changed at critical moment

Step 2: Found 1 anomalous transition
  - Unexpected object behavior detected

================================================================================
IDENTIFIED EVENTS:
================================================================================
1. Normal object movement (frames 0-45)
2. Anomalous state transition (frame 67)
```

## Troubleshooting

### Error: "Missing required scripts"

**Problem:** The framework can't find one or more required scripts.

**Solution:** Verify your directory structure matches the expected layout:

```bash
cd /home/yizhou/Mprojects/VAD/TubeletGraph/
ls annotate/vlm_mask_grounded.py          # Should exist
ls quick_run.py                           # Should exist
ls TubeletGraph/vlm/prompt_vlm.py         # Should exist
```

If any are missing, check your installation or file paths.

### Error: "VLM API key not found"

**Problem:** API key not set in environment.

**Solution:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, export it
export OPENAI_API_KEY="your-key-here"

# To make it permanent, add to ~/.bashrc:
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Error: Stage 1 fails

**Common causes:**
1. VLM API key not set or invalid
2. Video file not accessible
3. Output directory permissions

**Debug:**
```bash
# Run with verbose flag to see detailed error
python stvad_framework.py analyze video.mp4 -c config.yaml -v

# Check video file
file video.mp4

# Test VLM API manually
python annotate/vlm_mask_grounded.py -i video.mp4 --vlm gpt4v --auto
```

### Error: Stage 2 fails

**Common causes:**
1. Mask file not generated in Stage 1
2. Config file path incorrect
3. TubeletGraph not properly installed

**Debug:**
```bash
# Check if mask was created
ls output/Annotations/*/

# Verify config exists
cat configs/default.yaml

# Test tracking manually
python quick_run.py -c configs/default.yaml \
    --input_dir output/JPEGImages/video \
    --input_mask output/Annotations/video/0000000.png \
    --fps 10
```

### Error: Stage 3 fails

**Common causes:**
1. Tracking predictions not generated
2. Video path incorrect
3. VLM API issues

**Debug:**
```bash
# Check tracking results
ls {outdir}/custom-*/

# Test manually
python TubeletGraph/vlm/prompt_vlm.py \
    -c configs/default.yaml \
    -p custom-video-Ours \
    --video_path video.mp4 \
    --detect_anomalies
```

### No anomalies detected when expected

**Possible issues:**
1. Objects not properly grounded in Stage 1
2. Sample interval too large (missing key frames)
3. VLM model limitations

**Try:**
```bash
# Reduce sample interval for finer analysis
python stvad_framework.py analyze video.mp4 \
    -c config.yaml \
    --sample_interval 5

# Increase FPS for more frames
python stvad_framework.py analyze video.mp4 \
    -c config.yaml \
    --fps 20

# Try different VLM model
python stvad_framework.py analyze video.mp4 \
    -c config.yaml \
    --vlm claude
```

## Advanced Usage

### Custom Output Directory

```bash
# Specify custom output location
python stvad_framework.py analyze video.mp4 \
    -c configs/default.yaml \
    -o /path/to/custom/output
```

### Processing with Different VLM Models

```bash
# Use Claude instead of GPT-4V
python stvad_framework.py analyze video.mp4 \
    -c configs/default.yaml \
    --vlm claude

# Ensure you have the Claude API key set
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Adjusting Processing Speed vs Quality

```bash
# Fast processing (lower quality)
python stvad_framework.py analyze video.mp4 \
    -c configs/default.yaml \
    --fps 5 \
    --sample_interval 20

# High quality (slower)
python stvad_framework.py analyze video.mp4 \
    -c configs/default.yaml \
    --fps 20 \
    --sample_interval 5
```

## Integration with Other Tools

### Using JSON Output in Python

```python
import json

# Load anomaly report
with open('output/anomaly_reports/video_report.json') as f:
    report = json.load(f)

# Check for anomalies
if report['anomaly_detected']:
    print(f"Found {report['num_anomalies']} anomalies")
    print(f"Events: {report['identified_events']}")
```

### Batch Processing with Custom Logic

```python
from pathlib import Path
import subprocess
import json

video_dir = Path("./videos")
for video in video_dir.glob("*.mp4"):
    # Run analysis
    subprocess.run([
        "python", "stvad_framework.py", "analyze",
        str(video), "-c", "configs/default.yaml"
    ])
    
    # Process results
    report_path = f"output/anomaly_reports/{video.stem}_report.json"
    with open(report_path) as f:
        report = json.load(f)
        
    # Custom logic here
    if report['anomaly_detected']:
        print(f"⚠️  Anomaly in {video.name}")
```

## Performance Considerations

### Memory Usage
- Higher FPS = more frames = more memory
- Batch processing processes videos sequentially to manage memory

### Processing Time
Approximate times (depends on video length, hardware, VLM API speed):
- Stage 1: 30-60 seconds per video
- Stage 2: 1-3 minutes per video
- Stage 3: 30-90 seconds per video

### Optimization Tips
1. Use appropriate FPS for your use case (surveillance: 5-10, action: 15-30)
2. Adjust sample_interval based on video complexity
3. For batch processing, use multiple machines if possible
4. Cache intermediate results (frames, masks) if reprocessing

## Contributing

To extend or modify the framework:

1. **Add new VLM models:** Modify `annotate/vlm_mask_grounded.py`
2. **Custom tracking methods:** Update `quick_run.py`
3. **Enhanced anomaly detection:** Modify `TubeletGraph/vlm/prompt_vlm.py`
4. **New output formats:** Update `generate_summary_report()` in framework

## License

[Your License Here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{stvad2024,
  title={ST-VAD: Spatio-Temporal Video Anomaly Detection},
  author={Your Name},
  year={2024}
}
```

## Contact

For questions or issues, please contact the maintainers.

---

**Installation Directory:** `/home/yizhou/Mprojects/VAD/TubeletGraph/`  
**Framework Version:** 2.0  
**Last Updated:** February 2026
