#!/usr/bin/env python3
"""
Build a custom (single-instance) dataset from input frames and mask annotation.
This script creates the necessary directory structure and split file for custom video processing.
"""

import os
import os.path as osp
import sys
import argparse
import shutil
import glob
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from utils import load_yaml_file


def inspect_image_format(input_dir):
    """
    Inspect the image directory to determine the image format/pattern.
    Returns a format string like '*.png'
    """
    # Check for common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in extensions:
        files = glob.glob(osp.join(input_dir, f'*{ext}'))
        if files:
            assert np.all(
                [f.endswith(ext) for f in os.listdir(input_dir)]
            ), f"*{ext} did not match all files in input directory."
            return '*' + ext
    raise ValueError(f"No image files with ext: {extensions} found in {input_dir}")


def inspect_annotation_format(input_mask):
    """
    Inspect the annotation file to determine its format.
    Returns the file extension.
    """
    ext = osp.splitext(input_mask)[1]
    if ext not in ['.png', '.PNG']:
        raise ValueError(f"Annotation file must be a PNG file, got {ext}")

    return '*' + ext


def create_custom_config(config, output_config_path, image_format, anno_format, fps=None):
    """
    Create a custom config file for the specific video with proper formats and FPS.
    
    Args:
        config: OmegaConf object loaded from the base config file
        output_config_path: Path to output custom config file
        image_format: Detected image format pattern
        anno_format: Detected annotation format pattern
        fps: Frames per second
    """
    config.datasets.custom.image_format = image_format
    config.datasets.custom.anno_format = anno_format
    if fps is not None:
        config.datasets.custom.fps = fps

    os.makedirs(osp.dirname(output_config_path), exist_ok=True)
    OmegaConf.save(config, output_config_path)
    return output_config_path


def create_custom_dataset(config_path, input_dir, input_mask, fps=None):
    """
    Create a custom (single-instance) dataset structure for processing.
    
    Args:
        config_path: Path to the config YAML file
        input_dir: Directory containing video frames
        input_mask: Path to the annotation PNG file (first frame mask)
        fps: Frames per second for the video
    """
    # Load config
    config = load_yaml_file(config_path)
    
    # Get the basename of input directory (will be used as video name)
    video_name = osp.basename(osp.normpath(input_dir))
    
    # Get custom dataset paths from config and create if not exist
    custom_config = config.datasets.custom

    image_base_dir = custom_config.image_dir
    anno_base_dir = custom_config.anno_dir
    split_dir = custom_config.split_dir
    os.makedirs(image_base_dir, exist_ok=True)
    os.makedirs(anno_base_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    
    # Create symlinks for input frames
    target_image_dir = osp.join(image_base_dir, video_name)
    if osp.exists(target_image_dir) or osp.islink(target_image_dir):
        print(f"Removing existing link/directory: {target_image_dir}")
        if osp.islink(target_image_dir):
            os.unlink(target_image_dir)
        else:
            shutil.rmtree(target_image_dir)
    input_dir_abs = osp.abspath(input_dir)
    os.symlink(input_dir_abs, target_image_dir)

    # Create symlinks for prompt annotation
    target_anno_dir = osp.join(anno_base_dir, video_name)
    os.makedirs(target_anno_dir, exist_ok=True)
    
    image_format = inspect_image_format(input_dir)
    image_files = sorted(glob.glob(osp.join(input_dir, image_format)))
    first_frame_name = osp.splitext(osp.basename(image_files[0]))[0]
    
    target_anno_file = osp.join(target_anno_dir, f"{first_frame_name}.png")
    anno_format = inspect_annotation_format(input_mask)

    mask = Image.open(input_mask)
    if mask.mode == 'P':
        input_mask_abs = osp.abspath(input_mask)
        if osp.exists(target_anno_file):
            os.remove(target_anno_file)
        os.symlink(input_mask_abs, target_anno_file)
    else:
        print(f"Converting annotation to indexed PNG: {target_anno_file}")
        mask.convert('P').save(target_anno_file)
    
    # Create split file
    split_file = osp.join(split_dir, f"{video_name}.txt")
    with open(split_file, 'w') as f:
        f.write(video_name)
    
    # Create custom config file automatically
    output_config_path = f"configs/custom_{video_name}.yaml"
    create_custom_config(
        config,
        output_config_path,
        image_format,
        anno_format,
        fps
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Single-Video Custom Dataset Created Successfully!")
    print(f"{'='*60}")
    print(f"  Video name:            {video_name}")
    print(f"  Input frame directory: {target_image_dir}")
    print(f"  Prompt object mask:    {target_anno_file}")
    print(f"  Split file:            {split_file}")
    print(f"  New Config file:       {output_config_path}")
    print(f"  Image format:          {image_format}")
    print(f"  Annotation format:     {anno_format}")
    print(f"  FPS:                   {fps if fps is not None else config.datasets.custom.fps}")
    print(f"{'='*60}\n")
    
    return output_config_path, video_name


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build custom dataset from input frames and annotation mask"
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to directory containing video frames"
    )
    parser.add_argument(
        "--input_mask",
        required=True,
        help="Path to PNG file containing the prompt objects (first frame annotation)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for the video (default: 10)"
    )
    return parser


def main():
    args = get_parser().parse_args()
    assert osp.isdir(args.input_dir) and osp.isfile(args.input_mask), "Invalid input directory or mask file"
    create_custom_dataset(
        args.config,
        args.input_dir,
        args.input_mask,
        args.fps
    )

## Test: python eval/build_custom_dataset.py -c configs/default.yaml --input_dir assets/example/0334_cut_fruit_1 --input_mask assets/example/0334_cut_fruit_1_0000000.png
if __name__ == "__main__":
    main()
