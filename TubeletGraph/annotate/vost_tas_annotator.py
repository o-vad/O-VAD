#!/usr/bin/env python3
"""
VOST-TAS Dataset Annotation Generator
=====================================

This script helps create annotations in the VOST-TAS format for the TubeletGraph paper
(arxiv:2511.04678). VOST-TAS extends standard VOS with transformation annotations.

VOST-TAS Annotation Requirements:
---------------------------------
1. First-Frame Object Masks (standard VOS)
   - Objects you want to track initially
   
2. Transformation Boundary Masks  
   - Masks at frames where transformations START and END
   - Including ALL resulting objects (e.g., apple slices after cutting)

3. Transformation Metadata (JSON)
   - Temporal boundaries (start/end frames)
   - Action verb (cut, peel, tear, fold, etc.)
   - Source object ID and resulting object IDs

Directory Structure:
--------------------
<dataset_root>/
├── JPEGImages/<video_name>/
│   ├── 0000000.jpg
│   ├── 0000001.jpg
│   └── ...
├── Annotations/<video_name>/
│   ├── 0000000.png          # First-frame mask (objects 1, 2, ...)
│   ├── 0000045.png          # Transformation start (optional)
│   ├── 0000078.png          # Transformation end with new objects
│   └── ...
├── Transformations/<video_name>.json   # VOST-TAS specific
└── splits/
    └── val.txt

Usage:
------
    # Interactive annotation with SAM3
    python vost_tas_annotator.py --video cooking.mp4 --output_dir ./dataset
    
    # With pre-defined transformation times
    python vost_tas_annotator.py --video cooking.mp4 --output_dir ./dataset \
        --transform_frames "45-78" "120-150"
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image


# ==============================================================================
# Data Classes for VOST-TAS Annotations
# ==============================================================================

@dataclass
class ResultingObject:
    """An object that emerges from a transformation."""
    object_id: int                    # Unique ID in the mask
    description: str                  # e.g., "apple slice", "torn piece"
    first_appearance_frame: int       # Frame where it first appears


@dataclass 
class Transformation:
    """A single transformation event in the video."""
    start_frame: int                  # Frame where transformation begins
    end_frame: int                    # Frame where transformation ends
    action_verb: str                  # e.g., "cut", "peel", "tear", "fold"
    source_object_id: int             # ID of object being transformed
    source_description: str           # e.g., "apple", "banana"
    resulting_objects: List[ResultingObject] = field(default_factory=list)
    
    # Optional: For object-object interactions
    tool_object_id: Optional[int] = None      # e.g., knife ID
    tool_description: Optional[str] = None    # e.g., "knife"


@dataclass
class VideoAnnotation:
    """Complete VOST-TAS annotation for a video."""
    video_name: str
    num_frames: int
    fps: float
    width: int
    height: int
    
    # Initial objects (from first frame mask)
    initial_objects: Dict[int, str] = field(default_factory=dict)  # {obj_id: description}
    
    # All transformations in the video
    transformations: List[Transformation] = field(default_factory=list)
    
    # Frame indices where masks are annotated
    annotated_frames: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "video_name": self.video_name,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "initial_objects": self.initial_objects,
            "transformations": [
                {
                    "start_frame": t.start_frame,
                    "end_frame": t.end_frame,
                    "action_verb": t.action_verb,
                    "source_object_id": t.source_object_id,
                    "source_description": t.source_description,
                    "resulting_objects": [
                        {
                            "object_id": r.object_id,
                            "description": r.description,
                            "first_appearance_frame": r.first_appearance_frame
                        }
                        for r in t.resulting_objects
                    ],
                    "tool_object_id": t.tool_object_id,
                    "tool_description": t.tool_description
                }
                for t in self.transformations
            ],
            "annotated_frames": self.annotated_frames
        }
    
    def save(self, output_path: str):
        """Save annotation to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Saved annotation: {output_path}")


# ==============================================================================
# VOS Mask Utilities
# ==============================================================================

def get_vos_palette() -> List[int]:
    """Generate DAVIS-style color palette for visualization."""
    palette = [0, 0, 0]  # Background: black
    colors = [
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    ]
    for i in range(1, 255):
        if i <= len(colors):
            palette.extend(colors[i-1])
        else:
            palette.extend([(i*67+100)%256, (i*137+80)%256, (i*37+120)%256])
    palette.extend([255, 255, 255])  # 255: void
    return palette


def save_vos_mask(mask: np.ndarray, output_path: str) -> None:
    """Save mask in VOS format (8-bit indexed PNG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    mask_img.putpalette(get_vos_palette())
    mask_img.save(str(output_path))


def load_vos_mask(mask_path: str) -> np.ndarray:
    """Load VOS mask and return as numpy array."""
    mask = Image.open(mask_path).convert('P')
    return np.array(mask)


def merge_masks(masks: List[np.ndarray], object_ids: List[int]) -> np.ndarray:
    """Merge multiple binary masks into single indexed mask."""
    H, W = masks[0].shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)
    
    for mask, obj_id in zip(masks, object_ids):
        if mask.ndim == 3:
            mask = mask.squeeze()
        combined[mask.astype(bool)] = obj_id
    
    return combined


# ==============================================================================
# SAM3 Integration for Multi-Object Annotation
# ==============================================================================

class SAM3Annotator:
    """
    SAM3-based annotator for VOST-TAS dataset creation.
    
    Supports:
    - Multi-object annotation in single frame
    - Text prompts for finding all instances
    - Interactive refinement
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None
    
    def _ensure_loaded(self):
        if self._model is not None:
            return
            
        print("Loading SAM3...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self._model = build_sam3_image_model()
            self._processor = Sam3Processor(self._model)
            print("✓ SAM3 loaded")
        except ImportError:
            raise RuntimeError(
                "SAM3 not installed. Install from:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e ."
            )
    
    def annotate_frame(
        self,
        image_path: str,
        objects: Dict[str, int],  # {description: object_id}
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Annotate multiple objects in a single frame.
        
        Args:
            image_path: Path to frame image
            objects: Dict mapping object descriptions to desired IDs
                     e.g., {"apple": 1, "knife": 2}
            threshold: Confidence threshold
        
        Returns:
            Tuple of (combined_mask, {obj_id: description})
        """
        self._ensure_loaded()
        
        image = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(image)
        
        H, W = np.array(image).shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        id_to_desc = {}
        
        for description, obj_id in objects.items():
            print(f"  Segmenting '{description}' -> ID {obj_id}")
            
            output = self._processor.set_text_prompt(
                state=state,
                prompt=description
            )
            
            masks = output["masks"]
            scores = output["scores"]
            
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            
            # Take best mask above threshold
            best_idx = None
            best_score = threshold
            for i, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx is not None:
                mask = masks[best_idx]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                combined[mask.astype(bool)] = obj_id
                id_to_desc[obj_id] = description
                print(f"    ✓ Found (score: {best_score:.3f})")
            else:
                print(f"    ✗ Not found above threshold")
        
        return combined, id_to_desc
    
    def annotate_transformation_result(
        self,
        image_path: str,
        source_description: str,
        expected_count: int = None,
        start_id: int = 2
    ) -> Tuple[np.ndarray, List[ResultingObject]]:
        """
        Annotate resulting objects after a transformation.
        
        For example, after cutting an apple, find all apple slices.
        
        Args:
            image_path: Path to frame at transformation end
            source_description: What to look for (e.g., "apple slice")
            expected_count: Expected number of resulting objects (optional)
            start_id: Starting object ID for new objects
        
        Returns:
            Tuple of (mask, list of ResultingObject)
        """
        self._ensure_loaded()
        
        image = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(image)
        
        print(f"  Finding all instances of '{source_description}'...")
        
        output = self._processor.set_text_prompt(
            state=state,
            prompt=source_description
        )
        
        masks = output["masks"]
        scores = output["scores"]
        
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()
        
        H, W = np.array(image).shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        resulting_objects = []
        
        # Sort by score and take top instances
        sorted_indices = np.argsort(scores)[::-1]
        
        if expected_count:
            sorted_indices = sorted_indices[:expected_count]
        
        for i, idx in enumerate(sorted_indices):
            if scores[idx] < 0.3:  # Minimum threshold
                break
                
            obj_id = start_id + i
            mask = masks[idx]
            
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            combined[mask.astype(bool)] = obj_id
            
            resulting_objects.append(ResultingObject(
                object_id=obj_id,
                description=f"{source_description}_{i+1}",
                first_appearance_frame=-1  # To be filled later
            ))
            
            print(f"    ✓ Object {obj_id}: score {scores[idx]:.3f}")
        
        return combined, resulting_objects


# ==============================================================================
# Video Frame Extraction
# ==============================================================================

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = None
) -> Tuple[str, int, float, int, int]:
    """
    Extract frames from video.
    
    Returns: (frames_dir, num_frames, fps, width, height)
    """
    import cv2
    
    video_path = Path(video_path)
    video_name = video_path.stem
    
    frames_dir = Path(output_dir) / "JPEGImages" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = original_fps
        interval = 1
    else:
        interval = max(1, int(original_fps / fps))
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval == 0:
            frame_path = frames_dir / f"{saved_count:07d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"✓ Extracted {saved_count} frames to {frames_dir}")
    
    return str(frames_dir), saved_count, fps, width, height


# ==============================================================================
# VOST-TAS Annotation Workflow
# ==============================================================================

class VOSTTASAnnotator:
    """
    Complete workflow for creating VOST-TAS annotations.
    
    Workflow:
    1. Extract video frames
    2. Annotate first frame with initial objects
    3. For each transformation:
       a. Mark start frame (when action begins)
       b. Mark end frame (when transformation completes)
       c. Annotate resulting objects at end frame
       d. Record action verb and object descriptions
    4. Save all annotations
    """
    
    def __init__(
        self,
        output_dir: str,
        device: str = "cuda"
    ):
        self.output_dir = Path(output_dir)
        self.sam3 = SAM3Annotator(device)
        
        # Create directories
        (self.output_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "Annotations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "Transformations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "splits").mkdir(parents=True, exist_ok=True)
    
    def create_annotation(
        self,
        video_path: str,
        initial_objects: Dict[str, int],
        transformations: List[Dict],
        fps: float = None
    ) -> VideoAnnotation:
        """
        Create complete VOST-TAS annotation for a video.
        
        Args:
            video_path: Path to input video
            initial_objects: Objects to track from first frame
                            {"apple": 1, "knife": 2}
            transformations: List of transformation specs:
                [
                    {
                        "start_frame": 45,
                        "end_frame": 78,
                        "action_verb": "cut",
                        "source_id": 1,
                        "source_desc": "apple",
                        "result_desc": "apple slice",
                        "tool_id": 2,  # optional
                        "tool_desc": "knife"  # optional
                    },
                    ...
                ]
            fps: Target FPS for extraction
        
        Returns:
            VideoAnnotation object
        """
        video_name = Path(video_path).stem
        
        # Step 1: Extract frames
        print("\n[1/4] Extracting frames...")
        frames_dir, num_frames, actual_fps, width, height = extract_frames(
            video_path, str(self.output_dir), fps
        )
        
        # Step 2: Annotate first frame
        print("\n[2/4] Annotating first frame objects...")
        first_frame = Path(frames_dir) / "0000000.jpg"
        
        first_mask, id_to_desc = self.sam3.annotate_frame(
            str(first_frame),
            initial_objects
        )
        
        anno_dir = self.output_dir / "Annotations" / video_name
        anno_dir.mkdir(parents=True, exist_ok=True)
        save_vos_mask(first_mask, str(anno_dir / "0000000.png"))
        
        # Initialize annotation object
        annotation = VideoAnnotation(
            video_name=video_name,
            num_frames=num_frames,
            fps=actual_fps,
            width=width,
            height=height,
            initial_objects={v: k for k, v in initial_objects.items()},
            annotated_frames=[0]
        )
        
        # Step 3: Process each transformation
        print("\n[3/4] Processing transformations...")
        next_obj_id = max(initial_objects.values()) + 1
        
        for i, trans_spec in enumerate(transformations):
            print(f"\n  Transformation {i+1}: {trans_spec['action_verb']}")
            
            # Get end frame
            end_frame = trans_spec["end_frame"]
            end_frame_path = Path(frames_dir) / f"{end_frame:07d}.jpg"
            
            if not end_frame_path.exists():
                print(f"    ⚠ Frame {end_frame} not found, skipping")
                continue
            
            # Annotate resulting objects at end frame
            result_mask, resulting_objects = self.sam3.annotate_transformation_result(
                str(end_frame_path),
                trans_spec["result_desc"],
                start_id=next_obj_id
            )
            
            # Update appearance frames
            for obj in resulting_objects:
                obj.first_appearance_frame = end_frame
            
            # Merge with existing objects (keep source object and tool if present)
            # Load previous mask to get continuing objects
            if annotation.annotated_frames:
                last_anno_frame = max(annotation.annotated_frames)
                prev_mask = load_vos_mask(
                    str(anno_dir / f"{last_anno_frame:07d}.png")
                )
                # Keep objects that aren't the transformed source
                for obj_id in np.unique(prev_mask):
                    if obj_id == 0 or obj_id == 255:
                        continue
                    if obj_id != trans_spec["source_id"]:
                        result_mask[prev_mask == obj_id] = obj_id
            
            # Save end frame mask
            save_vos_mask(result_mask, str(anno_dir / f"{end_frame:07d}.png"))
            annotation.annotated_frames.append(end_frame)
            
            # Create Transformation object
            trans = Transformation(
                start_frame=trans_spec["start_frame"],
                end_frame=end_frame,
                action_verb=trans_spec["action_verb"],
                source_object_id=trans_spec["source_id"],
                source_description=trans_spec["source_desc"],
                resulting_objects=resulting_objects,
                tool_object_id=trans_spec.get("tool_id"),
                tool_description=trans_spec.get("tool_desc")
            )
            annotation.transformations.append(trans)
            
            # Update next available object ID
            if resulting_objects:
                next_obj_id = max(obj.object_id for obj in resulting_objects) + 1
        
        # Step 4: Save annotation JSON
        print("\n[4/4] Saving annotations...")
        anno_json_path = self.output_dir / "Transformations" / f"{video_name}.json"
        annotation.save(str(anno_json_path))
        
        # Update split file
        split_file = self.output_dir / "splits" / "val.txt"
        with open(split_file, 'a') as f:
            f.write(f"{video_name}\n")
        
        return annotation


# ==============================================================================
# Example Usage and CLI
# ==============================================================================

def example_annotation():
    """
    Example: Annotating a fruit cutting video.
    
    Scenario: A video shows someone cutting an apple with a knife.
    - Frame 0: Whole apple + knife visible
    - Frame 45-78: Knife cuts through apple
    - Frame 78+: Apple slices visible
    """
    
    annotator = VOSTTASAnnotator(output_dir="./vost_tas_dataset")
    
    # Define initial objects to track
    initial_objects = {
        "apple": 1,    # The apple we want to track through transformation
        "knife": 2     # The tool causing the transformation
    }
    
    # Define transformations
    # This captures the interaction: knife cuts apple -> apple slices
    transformations = [
        {
            "start_frame": 45,       # When cutting action starts
            "end_frame": 78,         # When cutting completes
            "action_verb": "cut",    # The transformation action
            "source_id": 1,          # Apple (being transformed)
            "source_desc": "apple",
            "result_desc": "apple slice",  # What to look for after
            "tool_id": 2,            # Knife (causing transformation)
            "tool_desc": "knife"
        }
    ]
    
    annotation = annotator.create_annotation(
        video_path="cooking.mp4",
        initial_objects=initial_objects,
        transformations=transformations
    )
    
    print("\n" + "="*50)
    print("Annotation complete!")
    print("="*50)
    print(f"\nState Graph for apple (ID 1):")
    print(f"  apple --[cut by knife, frames 45-78]--> apple slices")
    
    return annotation


def main():
    parser = argparse.ArgumentParser(
        description="VOST-TAS Annotation Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (programmatic):
  from vost_tas_annotator import VOSTTASAnnotator
  
  annotator = VOSTTASAnnotator("./dataset")
  annotation = annotator.create_annotation(
      video_path="cooking.mp4",
      initial_objects={"apple": 1, "knife": 2},
      transformations=[{
          "start_frame": 45,
          "end_frame": 78,
          "action_verb": "cut",
          "source_id": 1,
          "source_desc": "apple",
          "result_desc": "apple slice",
          "tool_id": 2,
          "tool_desc": "knife"
      }]
  )
        """
    )
    
    parser.add_argument("--video", "-v", required=True,
                       help="Input video path")
    parser.add_argument("--output_dir", "-o", default="./vost_tas_dataset",
                       help="Output directory")
    parser.add_argument("--config", "-c",
                       help="JSON config file with objects and transformations")
    parser.add_argument("--fps", type=float, default=None,
                       help="Target FPS for extraction")
    parser.add_argument("--device", default="cuda",
                       help="Device for SAM3")
    
    args = parser.parse_args()
    
    if args.config:
        # Load from config file
        with open(args.config) as f:
            config = json.load(f)
        
        annotator = VOSTTASAnnotator(args.output_dir, args.device)
        annotator.create_annotation(
            video_path=args.video,
            initial_objects=config["initial_objects"],
            transformations=config["transformations"],
            fps=args.fps
        )
    else:
        # Run example
        print("No config provided. Running example annotation...")
        print("Create a JSON config file with 'initial_objects' and 'transformations'")
        print("\nExample config.json:")
        example_config = {
            "initial_objects": {"apple": 1, "knife": 2},
            "transformations": [{
                "start_frame": 45,
                "end_frame": 78,
                "action_verb": "cut",
                "source_id": 1,
                "source_desc": "apple", 
                "result_desc": "apple slice",
                "tool_id": 2,
                "tool_desc": "knife"
            }]
        }
        print(json.dumps(example_config, indent=2))


if __name__ == "__main__":
    main()