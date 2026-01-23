# Using Claude (you're already using Claude, so you have access)
# export ANTHROPIC_API_KEY="your-key"
# python generate_mask_grounded.py --image frame.jpg --vlm claude --auto

# Or interactively select which objects to segment
export OPENAI_API_KEY="***REMOVED-CREDENTIAL***"
# python annotate/vlm_mask_grounded.py --image /home/yizhou/Mprojects/VAD/TubeletGraph/assets/example/01/000.jpg --vlm openai

# Using free local LLaVA (no API key needed)
# ollama serve  # Start Ollama
# ollama pull llava  # Download LLaVA model
# python generate_mask_grounded.py --image frame.jpg --vlm ollama --auto




# From MP4 video - auto detect and segment all objects
# python annotate/vlm_mask_grounded.py -i video.mp4 --vlm openai --auto

# From MP4 video - interactively select which objects to segment  
# python annotate/vlm_mask_grounded.py -i video.mp4 --vlm openai

# From MP4 video - save to specific output directory (TubeletGraph format)

# Automatically scan 5 frames to find best segmentation
# python generate_mask_grounded.py -i video.mp4 --vlm claude --auto --scan_frames
# Or scan more frames
# python generate_mask_grounded.py -i video.mp4 --vlm claude --auto --scan_frames --num_scan_frames 10
python annotate/vlm_mask_grounded.py -i assets/example/PhysAD/0000.mp4 --vlm openai --auto --output_dir assets/example/PhysAD/ --scan_frames --threshold 0.1




# From single image
# python annotate/vlm_mask_grounded.py -i frame.jpg --vlm openai --auto

# From frame folder
# python annotate/vlm_mask_grounded.py -i ./frames/ --vlm openai --auto

# Use free local LLaVA (no API key needed)
# python generate_mask_grounded.py -i video.mp4 --vlm ollama --auto




# 1. **Video Processing**: Automatically extracts frames from `.mp4`, `.avi`, `.mov`, etc.
# 2. **TubeletGraph Structure**: Creates proper directory structure:
# ```
#    output_dir/
#    ├── JPEGImages/<video_name>/
#    │   ├── 0000000.jpg
#    │   └── ...
#    ├── Annotations/<video_name>/
#    │   └── 0000000.png
#    └── splits/
#        └── val.txt