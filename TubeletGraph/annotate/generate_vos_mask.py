#!/usr/bin/env python3
"""
SAM3 First-Frame Mask Generator for Video Object Segmentation
==============================================================

Generate VOS-compatible first-frame masks using SAM3.

This script is designed to create FIRST_FRAME_MASK.png files compatible with:
- TubeletGraph (arxiv:2511.04678)
- DAVIS benchmark format
- YouTube-VOS format
- VOST dataset format

Output format:
- 8-bit indexed PNG
- Pixel value 0 = background
- Pixel values 1, 2, 3, ... = object IDs
- Pixel value 255 = ignore/void region

Requirements:
    conda create -n sam3 python=3.12
    pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3 && pip install -e .
    huggingface-cli login  # Required for model access

Usage Examples:
    # Text prompt (recommended for most use cases)
    python generate_vos_mask.py --image frame.jpg --text "apple" "knife"
    
    # Multiple objects with same text
    python generate_vos_mask.py --image frame.jpg --text "person"
    
    # Box prompts
    python generate_vos_mask.py --image frame.jpg --box 100,150,300,400 500,200,700,450
    
    # Point prompts (positive points)
    python generate_vos_mask.py --image frame.jpg --points 150,200 600,300
    
    # Specify output path
    python generate_vos_mask.py --image frame.jpg --text "cup" --output Annotations/video/0000000.png
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional


# VOS Color Palette (DAVIS-style)
def get_vos_palette() -> List[int]:
    """
    Generate VOS-compatible color palette.
    
    Returns:
        List of RGB values for 256 indices
    """
    palette = []
    
    # Index 0: Background (black)
    palette.extend([0, 0, 0])
    
    # Indices 1-254: Object colors
    # Using distinct, visually separable colors
    object_colors = [
        [128, 0, 0],      # 1: Dark red
        [0, 128, 0],      # 2: Dark green  
        [128, 128, 0],    # 3: Olive
        [0, 0, 128],      # 4: Dark blue
        [128, 0, 128],    # 5: Purple
        [0, 128, 128],    # 6: Teal
        [128, 128, 128],  # 7: Gray
        [64, 0, 0],       # 8: Maroon
        [192, 0, 0],      # 9: Red
        [64, 128, 0],     # 10: ...
    ]
    
    for i in range(1, 255):
        if i <= len(object_colors):
            palette.extend(object_colors[i-1])
        else:
            # Generate colors for remaining indices
            r = (i * 67 + 100) % 256
            g = (i * 137 + 80) % 256
            b = (i * 37 + 120) % 256
            palette.extend([r, g, b])
    
    # Index 255: Void/ignore (white)
    palette.extend([255, 255, 255])
    
    return palette


def save_indexed_mask(
    mask: np.ndarray,
    output_path: str,
    save_visualization: bool = True
) -> None:
    """
    Save mask as indexed PNG in VOS format.
    
    Args:
        mask: (H, W) array with object IDs as values
        output_path: Path to save PNG
        save_visualization: Also save a colored PNG for visualization
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create indexed image
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    mask_img.putpalette(get_vos_palette())
    mask_img.save(str(output_path))
    
    print(f"✓ Saved mask: {output_path}")
    
    # Print object statistics
    unique_ids = np.unique(mask)
    obj_ids = unique_ids[unique_ids > 0]
    if 255 in obj_ids:
        obj_ids = obj_ids[obj_ids != 255]
    
    print(f"  Objects: {len(obj_ids)}")
    for obj_id in obj_ids:
        pixel_count = np.sum(mask == obj_id)
        percentage = 100 * pixel_count / mask.size
        print(f"    ID {obj_id}: {pixel_count} pixels ({percentage:.1f}%)")
    
    # Save visualization
    if save_visualization:
        vis_path = output_path.parent / f"{output_path.stem}_colored.png"
        mask_img.convert('RGB').save(str(vis_path))
        print(f"✓ Saved visualization: {vis_path}")


class SAM3Segmenter:
    """SAM3-based segmentation for VOS mask generation."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None
    
    def _ensure_loaded(self):
        """Lazy load SAM3 model."""
        if self._model is not None:
            return
        
        print("Loading SAM3 model (this may take a moment)...")
        
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self._model = build_sam3_image_model()
            self._processor = Sam3Processor(self._model)
            print("✓ SAM3 loaded successfully\n")
            
        except ImportError:
            raise RuntimeError(
                "SAM3 not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e .\n"
                "  huggingface-cli login"
            )
    
    def segment_text(
        self,
        image: Image.Image,
        prompts: List[str],
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Segment using text prompts.
        
        SAM3's key feature is finding ALL instances of a concept,
        making it ideal for multi-object segmentation.
        
        Args:
            image: PIL Image
            prompts: List of text descriptions
            threshold: Confidence threshold
        
        Returns:
            Combined mask with object IDs
        """
        self._ensure_loaded()
        
        state = self._processor.set_image(image)
        H, W = np.array(image).shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        obj_id = 1
        
        for prompt in prompts:
            print(f"  Segmenting '{prompt}'...")
            
            output = self._processor.set_text_prompt(
                state=state,
                prompt=prompt
            )
            
            masks = output["masks"]
            scores = output["scores"]
            
            # Convert to numpy
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            
            for i, score in enumerate(scores):
                if score >= threshold:
                    mask = masks[i]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    combined[mask.astype(bool)] = obj_id
                    print(f"    Found instance (score: {score:.3f}) -> Object ID: {obj_id}")
                    obj_id += 1
        
        return combined
    
    def segment_boxes(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Segment using bounding box prompts.
        
        Args:
            image: PIL Image
            boxes: List of (x1, y1, x2, y2) boxes
        
        Returns:
            Combined mask with object IDs
        """
        import torch
        self._ensure_loaded()
        
        state = self._processor.set_image(image)
        H, W = np.array(image).shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        
        for obj_id, box in enumerate(boxes, start=1):
            print(f"  Segmenting box {box}...")
            
            box_tensor = torch.tensor([list(box)], dtype=torch.float32)
            output = self._processor.set_box_prompt(
                state=state,
                box=box_tensor
            )
            
            mask = output["masks"][0]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            combined[mask.astype(bool)] = obj_id
            print(f"    -> Object ID: {obj_id}")
        
        return combined
    
    def segment_points(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Segment using point prompts.
        
        Each point creates a separate object (for positive points).
        
        Args:
            image: PIL Image
            points: List of (x, y) coordinates
            labels: Point labels (1=positive, 0=negative). Default: all positive
        
        Returns:
            Combined mask with object IDs
        """
        import torch
        self._ensure_loaded()
        
        if labels is None:
            labels = [1] * len(points)
        
        state = self._processor.set_image(image)
        H, W = np.array(image).shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        
        # Group positive points - each becomes separate object
        positive_points = [p for p, l in zip(points, labels) if l == 1]
        
        for obj_id, point in enumerate(positive_points, start=1):
            print(f"  Segmenting point {point}...")
            
            points_tensor = torch.tensor([[list(point)]], dtype=torch.float32)
            labels_tensor = torch.tensor([[1]], dtype=torch.int32)
            
            output = self._processor.set_point_prompt(
                state=state,
                points=points_tensor,
                labels=labels_tensor
            )
            
            mask = output["masks"][0]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            combined[mask.astype(bool)] = obj_id
            print(f"    -> Object ID: {obj_id}")
        
        return combined


def parse_box(box_str: str) -> Tuple[int, int, int, int]:
    """Parse box string 'x1,y1,x2,y2' to tuple."""
    coords = [int(x.strip()) for x in box_str.split(',')]
    if len(coords) != 4:
        raise ValueError(f"Invalid box format: {box_str}. Expected x1,y1,x2,y2")
    return tuple(coords)


def parse_point(point_str: str) -> Tuple[int, int]:
    """Parse point string 'x,y' to tuple."""
    coords = [int(x.strip()) for x in point_str.split(',')]
    if len(coords) != 2:
        raise ValueError(f"Invalid point format: {point_str}. Expected x,y")
    return tuple(coords)


def main():
    parser = argparse.ArgumentParser(
        description="Generate VOS-compatible masks using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--image", "-i", required=True,
                       help="Input image path")
    
    # Prompt type (at least one required)
    prompt_group = parser.add_argument_group("Prompt Options (choose at least one)")
    prompt_group.add_argument("--text", "-t", nargs="+",
                             help="Text prompts (e.g., 'apple' 'knife')")
    prompt_group.add_argument("--box", "-b", nargs="+",
                             help="Box prompts as 'x1,y1,x2,y2'")
    prompt_group.add_argument("--points", "-p", nargs="+",
                             help="Point prompts as 'x,y'")
    
    # Output
    parser.add_argument("--output", "-o", default=None,
                       help="Output mask path (default: <image>_mask.png)")
    parser.add_argument("--no-vis", action="store_true",
                       help="Skip saving colored visualization")
    
    # Model options
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Confidence threshold for text prompts")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        parser.error(f"Image not found: {args.image}")
    
    if not any([args.text, args.box, args.points]):
        parser.error("At least one prompt type required: --text, --box, or --points")
    
    # Determine output path
    if args.output is None:
        image_path = Path(args.image)
        args.output = str(image_path.parent / f"{image_path.stem}_mask.png")
    
    print("=" * 50)
    print("SAM3 VOS Mask Generator")
    print("=" * 50)
    print(f"Input: {args.image}")
    print(f"Output: {args.output}")
    
    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size[0]}x{image.size[1]}")
    
    # Initialize segmenter
    segmenter = SAM3Segmenter(device=args.device)
    
    # Generate mask based on prompt type
    print("\nGenerating segmentation...")
    
    if args.text:
        mask = segmenter.segment_text(image, args.text, args.threshold)
    elif args.box:
        boxes = [parse_box(b) for b in args.box]
        mask = segmenter.segment_boxes(image, boxes)
    else:  # points
        points = [parse_point(p) for p in args.points]
        mask = segmenter.segment_points(image, points)
    
    # Save mask
    print("\nSaving mask...")
    save_indexed_mask(mask, args.output, save_visualization=not args.no_vis)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
