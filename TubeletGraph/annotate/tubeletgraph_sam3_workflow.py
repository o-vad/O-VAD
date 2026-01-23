#!/usr/bin/env python3
"""
TubeletGraph + SAM3 Workflow Script
====================================

Complete pipeline for:
1. Extracting frames from a video
2. Generating first-frame mask using SAM3 (text or visual prompts)
3. Saving masks in VOS-compatible format for TubeletGraph
4. Running TubeletGraph inference

Requirements:
- Python 3.12+
- PyTorch 2.7+ with CUDA 12.6+
- SAM3: pip install -e . (from https://github.com/facebookresearch/sam3)
- OpenCV: pip install opencv-python
- HuggingFace authentication for SAM3 checkpoints

Usage:
    # Text prompt mode
    python tubeletgraph_sam3_workflow.py \
        --video input_video.mp4 \
        --output_dir ./my_dataset \
        --text_prompts "apple" "knife" \
        --run_tubeletgraph

    # Interactive mode (click to select objects)
    python tubeletgraph_sam3_workflow.py \
        --video input_video.mp4 \
        --output_dir ./my_dataset \
        --interactive

    # Box prompt mode
    python tubeletgraph_sam3_workflow.py \
        --video input_video.mp4 \
        --output_dir ./my_dataset \
        --boxes "100,150,300,400" "500,200,700,450"

Author: Generated for TubeletGraph (arxiv:2511.04678) integration with SAM3
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

# ==============================================================================
# STEP 1: Frame Extraction from Video
# ==============================================================================

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None
) -> Tuple[str, int, float]:
    """
    Extract frames from video to JPEG files.
    
    Args:
        video_path: Path to input video file (mp4, avi, etc.)
        output_dir: Directory to save extracted frames
        fps: Target frames per second (None = original fps)
        max_frames: Maximum number of frames to extract (None = all)
    
    Returns:
        Tuple of (frames_dir, num_frames, actual_fps)
    """
    import cv2
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Create output structure: <output_dir>/JPEGImages/<video_name>/
    video_name = video_path.stem
    frames_dir = output_dir / "JPEGImages" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        fps = original_fps
        frame_interval = 1
    else:
        frame_interval = max(1, int(original_fps / fps))
    
    print(f"Video: {video_path}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Target FPS: {fps:.2f} (interval: {frame_interval})")
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            if max_frames and saved_count >= max_frames:
                break
            
            # Save as 7-digit zero-padded filename (TubeletGraph format)
            frame_path = frames_dir / f"{saved_count:07d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    print(f"  Extracted {saved_count} frames to: {frames_dir}")
    
    return str(frames_dir), saved_count, fps


# ==============================================================================
# STEP 2: SAM3 Mask Generation
# ==============================================================================

class SAM3MaskGenerator:
    """
    Generate segmentation masks using SAM3 (Segment Anything Model 3).
    
    SAM3 supports:
    - Text prompts: Find all instances of a concept (e.g., "apple", "person")
    - Box prompts: Segment objects within bounding boxes
    - Point prompts: Segment objects by clicking positive/negative points
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize SAM3 model.
        
        Note: Requires HuggingFace authentication for checkpoint access.
        Run `huggingface-cli login` before first use.
        """
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM3 model and processor."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print("Loading SAM3 model...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            print("SAM3 model loaded successfully!")
            
        except ImportError as e:
            print(f"Error: SAM3 not installed. Please install from:")
            print("  git clone https://github.com/facebookresearch/sam3.git")
            print("  cd sam3 && pip install -e .")
            raise e
        except Exception as e:
            print(f"Error loading SAM3: {e}")
            print("Make sure you have authenticated with HuggingFace:")
            print("  huggingface-cli login")
            raise e
    
    def segment_with_text(
        self,
        image_path: str,
        text_prompts: List[str],
        score_threshold: float = 0.5
    ) -> Tuple[np.ndarray, dict]:
        """
        Segment objects using text prompts.
        
        Args:
            image_path: Path to input image
            text_prompts: List of text descriptions (e.g., ["apple", "knife"])
            score_threshold: Minimum confidence score to keep masks
        
        Returns:
            Tuple of (combined_mask, metadata_dict)
            - combined_mask: np.ndarray (H, W) with object IDs as pixel values
            - metadata_dict: Contains masks, boxes, scores per object
        """
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        inference_state = self.processor.set_image(image)
        
        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for prompt in text_prompts:
            print(f"  Segmenting: '{prompt}'")
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )
            
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Filter by score threshold
            if hasattr(scores, 'cpu'):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.array(scores)
            
            for i, score in enumerate(scores_np):
                if score >= score_threshold:
                    if hasattr(masks[i], 'cpu'):
                        mask = masks[i].cpu().numpy()
                    else:
                        mask = np.array(masks[i])
                    
                    all_masks.append(mask)
                    all_boxes.append(boxes[i] if boxes is not None else None)
                    all_scores.append(float(score))
                    all_labels.append(prompt)
        
        # Combine masks into single indexed array
        if len(all_masks) == 0:
            H, W = np.array(image).shape[:2]
            combined = np.zeros((H, W), dtype=np.uint8)
        else:
            H, W = all_masks[0].shape[-2:]
            combined = np.zeros((H, W), dtype=np.uint8)
            
            for obj_id, mask in enumerate(all_masks, start=1):
                # Handle different mask shapes
                if mask.ndim == 3:
                    mask = mask.squeeze()
                mask_bool = mask.astype(bool)
                combined[mask_bool] = obj_id
        
        metadata = {
            "num_objects": len(all_masks),
            "labels": all_labels,
            "scores": all_scores,
            "boxes": all_boxes
        }
        
        return combined, metadata
    
    def segment_with_boxes(
        self,
        image_path: str,
        boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, dict]:
        """
        Segment objects within specified bounding boxes.
        
        Args:
            image_path: Path to input image
            boxes: List of (x1, y1, x2, y2) bounding boxes
        
        Returns:
            Tuple of (combined_mask, metadata_dict)
        """
        from PIL import Image
        import torch
        
        image = Image.open(image_path).convert("RGB")
        inference_state = self.processor.set_image(image)
        
        all_masks = []
        
        for i, box in enumerate(boxes):
            print(f"  Segmenting box {i+1}: {box}")
            
            # SAM3 box prompt format
            box_tensor = torch.tensor([box], dtype=torch.float32)
            
            output = self.processor.set_box_prompt(
                state=inference_state,
                box=box_tensor
            )
            
            masks = output["masks"]
            if hasattr(masks, 'cpu'):
                mask = masks[0].cpu().numpy()
            else:
                mask = np.array(masks[0])
            
            all_masks.append(mask)
        
        # Combine masks
        if len(all_masks) == 0:
            H, W = np.array(image).shape[:2]
            combined = np.zeros((H, W), dtype=np.uint8)
        else:
            H, W = all_masks[0].shape[-2:]
            combined = np.zeros((H, W), dtype=np.uint8)
            
            for obj_id, mask in enumerate(all_masks, start=1):
                if mask.ndim == 3:
                    mask = mask.squeeze()
                mask_bool = mask.astype(bool)
                combined[mask_bool] = obj_id
        
        metadata = {
            "num_objects": len(all_masks),
            "boxes": boxes
        }
        
        return combined, metadata
    
    def segment_interactive(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, dict]:
        """
        Interactive segmentation - click to select objects.
        
        Opens a window for clicking positive (left-click) and 
        negative (right-click) points. Press 'n' for next object,
        'q' to finish.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Tuple of (combined_mask, metadata_dict)
        """
        import cv2
        from PIL import Image
        import torch
        
        image_pil = Image.open(image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        inference_state = self.processor.set_image(image_pil)
        
        H, W = image_cv.shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        
        current_obj_id = 1
        positive_points = []
        negative_points = []
        
        window_name = "SAM3 Interactive Segmentation"
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal positive_points, negative_points
            
            if event == cv2.EVENT_LBUTTONDOWN:
                positive_points.append([x, y])
                print(f"    Positive point: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                negative_points.append([x, y])
                print(f"    Negative point: ({x}, {y})")
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("\nInteractive Segmentation Controls:")
        print("  Left-click: Add positive point (include in object)")
        print("  Right-click: Add negative point (exclude from object)")
        print("  'n': Confirm object and start next")
        print("  'r': Reset current object points")
        print("  'q': Finish and save")
        
        display_img = image_cv.copy()
        
        while True:
            # Draw points on display image
            temp_display = display_img.copy()
            for pt in positive_points:
                cv2.circle(temp_display, tuple(pt), 5, (0, 255, 0), -1)
            for pt in negative_points:
                cv2.circle(temp_display, tuple(pt), 5, (0, 0, 255), -1)
            
            cv2.imshow(window_name, temp_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n') and (positive_points or negative_points):
                # Segment current object
                print(f"  Segmenting object {current_obj_id}...")
                
                points = positive_points + negative_points
                labels = [1] * len(positive_points) + [0] * len(negative_points)
                
                points_tensor = torch.tensor([points], dtype=torch.float32)
                labels_tensor = torch.tensor([labels], dtype=torch.int32)
                
                output = self.processor.set_point_prompt(
                    state=inference_state,
                    points=points_tensor,
                    labels=labels_tensor
                )
                
                mask = output["masks"][0]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                combined[mask.astype(bool)] = current_obj_id
                
                # Update display with mask overlay
                mask_color = np.zeros_like(image_cv)
                color = [(current_obj_id * 67) % 256, 
                        (current_obj_id * 137) % 256, 
                        (current_obj_id * 37) % 256]
                mask_color[mask.astype(bool)] = color
                display_img = cv2.addWeighted(image_cv, 0.7, mask_color, 0.3, 0)
                
                current_obj_id += 1
                positive_points = []
                negative_points = []
                print(f"  Object saved. Click for object {current_obj_id} or press 'q' to finish.")
                
            elif key == ord('r'):
                positive_points = []
                negative_points = []
                print("  Points reset.")
                
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        metadata = {
            "num_objects": current_obj_id - 1,
            "mode": "interactive"
        }
        
        return combined, metadata


# ==============================================================================
# STEP 3: Save Mask in VOS Format
# ==============================================================================

def save_vos_mask(
    mask: np.ndarray,
    output_path: str,
    create_visualization: bool = True
) -> str:
    """
    Save mask in VOS-compatible format (8-bit indexed PNG).
    
    Args:
        mask: np.ndarray (H, W) with object IDs as pixel values
              - 0 = background
              - 1, 2, 3, ... = object IDs
              - 255 = ignore/void region
        output_path: Path to save the mask PNG
        create_visualization: Also save a colored visualization
    
    Returns:
        Path to saved mask file
    """
    from PIL import Image
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    # Create indexed PNG with palette
    mask_img = Image.fromarray(mask, mode='P')
    
    # Create DAVIS-style color palette
    # Background = black, then distinct colors for each object
    palette = [0, 0, 0]  # Index 0: background (black)
    
    # Generate distinct colors for objects 1-254
    for i in range(1, 255):
        # Use prime multipliers for good color distribution
        r = (i * 67 + 100) % 256
        g = (i * 137 + 80) % 256
        b = (i * 37 + 120) % 256
        palette.extend([r, g, b])
    
    # Index 255: void/ignore (typically white or special color)
    palette.extend([255, 255, 255])
    
    mask_img.putpalette(palette)
    mask_img.save(str(output_path))
    
    print(f"Saved mask: {output_path}")
    
    # Optionally save visualization
    if create_visualization:
        vis_path = output_path.parent / f"{output_path.stem}_vis.png"
        vis_img = mask_img.convert('RGB')
        vis_img.save(str(vis_path))
        print(f"Saved visualization: {vis_path}")
    
    return str(output_path)


def create_dataset_structure(
    output_dir: str,
    video_name: str,
    mask: np.ndarray,
    fps: float = 30.0
) -> dict:
    """
    Create complete TubeletGraph-compatible dataset structure.
    
    Expected structure:
        <output_dir>/
        ├── JPEGImages/<video_name>/
        │   ├── 0000000.jpg
        │   ├── 0000001.jpg
        │   └── ...
        ├── Annotations/<video_name>/
        │   └── 0000000.png  # First-frame mask
        └── splits/
            └── val.txt
    
    Args:
        output_dir: Base output directory
        video_name: Name of the video
        mask: First-frame mask array
        fps: Video FPS for config
    
    Returns:
        Dict with paths to created files/directories
    """
    output_dir = Path(output_dir)
    
    # Create directories
    anno_dir = output_dir / "Annotations" / video_name
    anno_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Save first-frame mask
    mask_path = anno_dir / "0000000.png"
    save_vos_mask(mask, str(mask_path))
    
    # Create split file
    split_file = splits_dir / "val.txt"
    with open(split_file, 'w') as f:
        f.write(f"{video_name}\n")
    
    print(f"\nDataset structure created:")
    print(f"  Annotations: {anno_dir}")
    print(f"  Split file: {split_file}")
    
    return {
        "anno_dir": str(anno_dir),
        "mask_path": str(mask_path),
        "split_file": str(split_file),
        "fps": fps
    }


# ==============================================================================
# STEP 4: Run TubeletGraph
# ==============================================================================

def run_tubeletgraph(
    frames_dir: str,
    mask_path: str,
    fps: float = 30.0,
    tubeletgraph_path: Optional[str] = None
) -> str:
    """
    Run TubeletGraph inference.
    
    Args:
        frames_dir: Directory containing JPEG frames
        mask_path: Path to first-frame mask PNG
        fps: Video FPS
        tubeletgraph_path: Path to TubeletGraph repository (optional)
    
    Returns:
        Path to output predictions directory
    """
    # Construct command
    cmd = [
        "python", "quick_run.py",
        "--input_dir", frames_dir,
        "--input_mask", mask_path,
        "--fps", str(int(fps))
    ]
    
    print("\nRunning TubeletGraph...")
    print(f"  Command: {' '.join(cmd)}")
    
    # Change to TubeletGraph directory if specified
    cwd = tubeletgraph_path if tubeletgraph_path else None
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        # Output is typically in ./_pred_out/predictions/
        output_dir = "_pred_out/predictions"
        if cwd:
            output_dir = os.path.join(cwd, output_dir)
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error running TubeletGraph: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise


# ==============================================================================
# Main Workflow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TubeletGraph + SAM3 Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames and generate mask with text prompts
  python tubeletgraph_sam3_workflow.py \\
      --video cooking.mp4 \\
      --output_dir ./dataset \\
      --text_prompts "apple" "knife" "cutting board"
  
  # Use bounding boxes instead
  python tubeletgraph_sam3_workflow.py \\
      --video cooking.mp4 \\
      --output_dir ./dataset \\
      --boxes "100,150,300,400" "500,200,700,450"
  
  # Interactive clicking mode
  python tubeletgraph_sam3_workflow.py \\
      --video cooking.mp4 \\
      --output_dir ./dataset \\
      --interactive
  
  # Full pipeline including TubeletGraph
  python tubeletgraph_sam3_workflow.py \\
      --video cooking.mp4 \\
      --output_dir ./dataset \\
      --text_prompts "apple" \\
      --run_tubeletgraph \\
      --tubeletgraph_path /path/to/TubeletGraph
        """
    )
    
    # Input options
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="./vos_dataset",
                       help="Output directory for dataset")
    
    # Frame extraction options
    parser.add_argument("--fps", type=float, default=None,
                       help="Target FPS for frame extraction (default: original)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to extract")
    
    # SAM3 prompt options (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--text_prompts", nargs="+",
                             help="Text prompts for segmentation (e.g., 'apple' 'knife')")
    prompt_group.add_argument("--boxes", nargs="+",
                             help="Bounding boxes as 'x1,y1,x2,y2' strings")
    prompt_group.add_argument("--interactive", action="store_true",
                             help="Interactive point-click mode")
    
    # SAM3 options
    parser.add_argument("--score_threshold", type=float, default=0.5,
                       help="Minimum confidence score for masks")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for SAM3 (cuda/cpu)")
    
    # TubeletGraph options
    parser.add_argument("--run_tubeletgraph", action="store_true",
                       help="Run TubeletGraph after mask generation")
    parser.add_argument("--tubeletgraph_path", type=str, default=None,
                       help="Path to TubeletGraph repository")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TubeletGraph + SAM3 Workflow")
    print("=" * 60)
    
    # Step 1: Extract frames
    print("\n[Step 1] Extracting frames from video...")
    frames_dir, num_frames, actual_fps = extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames
    )
    
    if num_frames == 0:
        print("Error: No frames extracted!")
        sys.exit(1)
    
    # Step 2: Generate mask using SAM3
    print("\n[Step 2] Generating first-frame mask with SAM3...")
    
    first_frame = os.path.join(frames_dir, "0000000.jpg")
    if not os.path.exists(first_frame):
        print(f"Error: First frame not found: {first_frame}")
        sys.exit(1)
    
    generator = SAM3MaskGenerator(device=args.device)
    
    if args.text_prompts:
        print(f"  Using text prompts: {args.text_prompts}")
        mask, metadata = generator.segment_with_text(
            image_path=first_frame,
            text_prompts=args.text_prompts,
            score_threshold=args.score_threshold
        )
    elif args.boxes:
        # Parse box strings
        boxes = []
        for box_str in args.boxes:
            coords = [int(x) for x in box_str.split(",")]
            if len(coords) != 4:
                print(f"Error: Invalid box format: {box_str}")
                sys.exit(1)
            boxes.append(tuple(coords))
        
        print(f"  Using bounding boxes: {boxes}")
        mask, metadata = generator.segment_with_boxes(
            image_path=first_frame,
            boxes=boxes
        )
    else:  # interactive
        print("  Starting interactive mode...")
        mask, metadata = generator.segment_interactive(
            image_path=first_frame
        )
    
    print(f"  Found {metadata['num_objects']} object(s)")
    
    # Step 3: Save mask and create dataset structure
    print("\n[Step 3] Creating dataset structure...")
    video_name = Path(args.video).stem
    
    dataset_info = create_dataset_structure(
        output_dir=args.output_dir,
        video_name=video_name,
        mask=mask,
        fps=actual_fps
    )
    
    # Step 4: Run TubeletGraph (optional)
    if args.run_tubeletgraph:
        print("\n[Step 4] Running TubeletGraph inference...")
        output_path = run_tubeletgraph(
            frames_dir=frames_dir,
            mask_path=dataset_info["mask_path"],
            fps=actual_fps,
            tubeletgraph_path=args.tubeletgraph_path
        )
        print(f"\nTubeletGraph output: {output_path}")
    
    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  Frames: {frames_dir}")
    print(f"  Mask: {dataset_info['mask_path']}")
    print(f"\nTo run TubeletGraph manually:")
    print(f"  python quick_run.py \\")
    print(f"      --input_dir {frames_dir} \\")
    print(f"      --input_mask {dataset_info['mask_path']} \\")
    print(f"      --fps {int(actual_fps)}")


if __name__ == "__main__":
    main()
