#!/usr/bin/env python3
"""
Quick run script for TubeletGraph: single-command inference from input frames to predictions.

Steps:
1. Build custom dataset from input frames and mask
2. Run TubeletGraph pipeline (entity segmentation, tubelets, semantic similarity, predictions)

Visualization (eval/vis.py) is intentionally omitted: its output is not consumed
by any downstream stage and adds significant per-object subprocess overhead.

Usage:
    python quick_run.py --input_dir <VIDEO_FRAME_DIR_PATH> --input_mask <ANNO_PNG_PATH> [--fps 30 --method Ours --mask_frame_id 0]
"""

import os
import os.path as osp
import sys
import argparse
import subprocess
import shutil
import tempfile
import glob

from eval.build_custom_dataset import create_custom_dataset
from utils import load_yaml_file


def run_command(cmd, description, check=True):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, check=False)

    if check and result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n{description} completed successfully\n")
    return result.returncode == 0


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quick run for TubeletGraph: end-to-end processing from input frames to predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--input_mask", required=True)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--vlm_model", default="openai")

    # NEW ARGUMENT
    parser.add_argument("--mask_frame_id", type=int, default=0, help="Frame ID for the mask input.")
    parser.add_argument("--hint_str", type=str, default="", help="Optional hint string to provide to the VLM for better predictions.")
    
    return parser


def main():
    args = get_parser().parse_args()

    if not osp.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not osp.isfile(args.input_mask):
        print(f"Error: Input mask file does not exist: {args.input_mask}")
        sys.exit(1)

    print("=" * 80)
    print("TubeletGraph Quick Run")
    print("=" * 80)
    
    video_stem = osp.basename(osp.normpath(args.input_dir))

    with tempfile.TemporaryDirectory(prefix="stvad_fps_stage_") as staging_dir:
        working_input_dir = args.input_dir

        custom_config_path, video_name = create_custom_dataset(
            args.config,
            working_input_dir,
            args.input_mask,
            fps=args.fps
        )

        # Removed --skip_vlm so TubeletGraph generates the state semantics!
        cmd = [
            "python3", "TubeletGraph/run.py",
            "-c", custom_config_path,
            "-d", "custom",
            "-s", video_name,
            "-m", args.method,
            "--mask_frame_id", str(args.mask_frame_id),  # Added here to pass it to run.py
            "--hint_str", args.hint_str,  # Added here to pass the hint string to run.py
        ]
        
        if not run_command(cmd, "Running TubeletGraph pipeline"):
            sys.exit(1)

        cfg = load_yaml_file(custom_config_path)
        pred_outdir = cfg.paths.outdir
        
        # Rename the resulting folder to exactly the original video name
        generated_folders = glob.glob(osp.join(pred_outdir, f"custom-{video_name}-*"))
        final_pred_path = osp.join(pred_outdir, video_name)
        
        if generated_folders:
            latest_folder = max(generated_folders, key=os.path.getmtime)
            if osp.exists(final_pred_path):
                shutil.rmtree(final_pred_path)
            os.rename(latest_folder, final_pred_path)
        else:
            print(f"[Warning] Could not find generated folder to rename.")
            final_pred_path = osp.join(pred_outdir, f"custom-{video_name}-{args.method}")

        print("\n" + "=" * 80)
        print("Quick run completed.")
        print(f"  Predictions: {final_pred_path}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()