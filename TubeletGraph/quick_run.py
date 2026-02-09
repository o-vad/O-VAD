#!/usr/bin/env python3
"""
Quick demo script for TubeletGraph: single-command inference from input frames to predictions.

This script combines the following steps:
1. Build custom dataset from input frames and mask
2. Run TubeletGraph pipeline (entity segmentation, tubelets, semantic similarity, predictions)
3. Visualize all predictions

Usage:
    python quick_demo.py --input_dir <VIDEO_FRAME_DIR_PATH> --input_mask <ANNO_PNG_PATH> [--fps 30 --method Ours]
"""

import os.path as osp
import sys
import argparse
import subprocess
from PIL import Image
import numpy as np

from eval.build_custom_dataset import create_custom_dataset
from utils import load_anno, load_yaml_file


def run_command(cmd, description, check=True):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if check and result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully\n")
    return result.returncode == 0


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quick demo for TubeletGraph: end-to-end processing from input frames to predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a custom video with default settings
    python quick_demo.py --input_dir ./my_video_frames --input_mask ./my_mask.png   

    # Process with custom FPS and method
    python quick_demo.py -c configs/default.yaml --input_dir ./frames --input_mask ./mask.png --fps 30 --method Ours
    # Use multiple GPUs for faster processing
    python quick_demo.py --input_dir ./frames --input_mask ./mask.png --gpus 0 1 2 3
        """
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/default.yaml",
        help="Path to base config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to directory containing video frames (required)"
    )
    parser.add_argument(
        "--input_mask",
        required=True,
        help="Path to PNG file containing the prompt objects annotation (required)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for the video (default: 30)"
    )
    parser.add_argument(
        "--method",
        default="Ours",
        help="Method name for predictions (default: Ours)"
    )
    return parser


def main():
    args = get_parser().parse_args()
    
    # Validate inputs
    if not osp.isdir(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not osp.isfile(args.input_mask):
        print(f"❌ Error: Input mask file does not exist: {args.input_mask}")
        sys.exit(1)
    
    # Extract object IDs from mask
    obj_ids = list(load_anno(args.input_mask).keys())
    print(f"Found {len(obj_ids)} objects in mask: {obj_ids}")
    
    print("="*80)
    print("TubeletGraph Quick Demo")
    print("="*80)
    print(f"Base config:      {args.config}")
    print(f"Input directory:  {args.input_dir}")
    print(f"Input mask:       {args.input_mask}")
    print(f"Object IDs:       {obj_ids}")
    print(f"Method:           {args.method}")
    print(f"FPS:              {args.fps}")
    print("="*80)
    
    # ========================================================================
    # Step 1: Build custom dataset (generates custom config automatically)
    # ========================================================================
    custom_config_path, video_name = create_custom_dataset(
        args.config,
        args.input_dir,
        args.input_mask,
        fps=args.fps
    )
    
    # ========================================================================
    # Step 2: Run TubeletGraph pipeline
    # ========================================================================
    cmd = [
        "python3", "TubeletGraph/run.py",
        "-c", custom_config_path,
        "-d", "custom",
        "-s", video_name,
        "-m", args.method
    ]
    if not run_command(cmd, "Running TubeletGraph pipeline"):
        sys.exit(1)
    
    cfg = load_yaml_file(custom_config_path)
    pred_name = f"custom-{video_name}-{args.method}_{cfg.vlm.model_name}"
    
    # ========================================================================
    # Step 3: Visualize predictions for all objects
    # ========================================================================

    for obj_id in obj_ids:
        instance_id = f"{video_name}_{obj_id}"
        
        cmd = [
            "python3", "eval/vis.py",
            "-c", custom_config_path,
            "-p", pred_name,
            "-i", instance_id
        ]
        success = run_command(cmd, f"Visualizing object {obj_id}", check=False)
        if not success:
            print(f"⚠ Warning: Visualization failed for object {obj_id}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    print("\n" + "="*80)
    print("✓ Quick demo completed successfully!")
    print("="*80)
    print(f"\nResults:")
    print(f"  - Predictions: ", osp.join(cfg.paths.outdir, pred_name))
    print(f"  - Visualizations: ", osp.join(cfg.paths.outdir, "predictions", pred_name))
    print("\n" + "="*80 + "\n")

## Testing: python quick_run.py  -c configs/default.yaml --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png
if __name__ == "__main__":
    main()