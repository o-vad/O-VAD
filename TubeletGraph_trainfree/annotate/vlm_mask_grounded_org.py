#!/usr/bin/env python3
"""
SAM3 Mask Generator with VLM Grounding
======================================

This script uses a Vision-Language Model (VLM) to automatically identify
objects in an image or video, then uses SAM3 to segment them.

Supports:
- Single image files (jpg, png, etc.)
- Video files (mp4, avi, mov, etc.)
- Folder of video frames (jpg sequence)

Workflow:
1. Extract first frame (if video/folder input)
2. VLM analyzes the frame and lists all visible objects
3. User selects which objects to segment (or segment all)
4. SAM3 segments the selected objects using text prompts
5. Output is saved as VOS-compatible mask

Supported VLMs:
- Claude (via Anthropic API) - RECOMMENDED, best quality
- GPT-4V (via OpenAI API)
- LLaVA (local, via Ollama)

Requirements:
    pip install anthropic openai pillow numpy opencv-python
    
    # For SAM3
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3 && pip install -e .
    
    # Set API keys
    export ANTHROPIC_API_KEY="your-key"
    # or
    export OPENAI_API_KEY="your-key"

Usage:
    # From a video file (mp4)
    python generate_mask_grounded.py --input video.mp4 --vlm claude --auto
    
    # From a single image
    python generate_mask_grounded.py --input frame.jpg --vlm claude --auto
    
    # From a folder of frames
    python generate_mask_grounded.py --input ./frames_folder/ --vlm claude
    
    # Specify output directory for TubeletGraph format
    python generate_mask_grounded.py --input video.mp4 --vlm claude --auto --output_dir ./dataset
    
    # Use local LLaVA via Ollama (free, no API key)
    python generate_mask_grounded.py --input video.mp4 --vlm ollama --auto
    
    # Use a specific frame number (useful when objects appear later in video)
    python generate_mask_grounded.py --input video.mp4 --vlm claude --auto --frame 60
    
    # Scan multiple frames to find best frame for each object
    python generate_mask_grounded.py --input video.mp4 --vlm claude --auto --scan_frames
    
    # Lower threshold for difficult objects (metallic, reflective, etc.)
    python generate_mask_grounded.py --input video.mp4 --vlm claude --auto --threshold 0.1
"""

import os
import sys
import argparse
import base64
import json
import re
import shutil
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict, Union
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# Input Processing (Video, Image, Folder)
# ==============================================================================

class InputProcessor:
    """Handle different input types: video files, images, or frame folders."""
    
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, input_path: str, output_dir: Optional[str] = None, frame_index: int = 0):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.frame_index = frame_index  # Which frame to use for segmentation
        self.input_type = self._detect_input_type()
        self.temp_dir = None
        self.frames_dir = None
        self.first_frame_path = None
        self.selected_frame_path = None  # The frame used for segmentation
        self.video_name = None
        self.fps = 30.0
        self.num_frames = 0
        
    def _detect_input_type(self) -> str:
        """Detect whether input is video, image, or folder."""
        if self.input_path.is_dir():
            return "folder"
        elif self.input_path.suffix.lower() in self.VIDEO_EXTENSIONS:
            return "video"
        elif self.input_path.suffix.lower() in self.IMAGE_EXTENSIONS:
            return "image"
        else:
            raise ValueError(f"Unknown input type: {self.input_path}")
    
    def process(self) -> Tuple[str, str]:
        """
        Process input and return (selected_frame_path, frames_directory).
        
        For videos: extracts frames to output_dir or temp directory
        For folders: uses existing frames
        For images: uses the image directly
        
        Returns:
            Tuple of (selected_frame_path, frames_dir)
        """
        print(f"\nInput type: {self.input_type}")
        print(f"Input path: {self.input_path}")
        if self.frame_index > 0:
            print(f"Using frame index: {self.frame_index}")
        
        if self.input_type == "video":
            return self._process_video()
        elif self.input_type == "folder":
            return self._process_folder()
        else:  # image
            return self._process_image()
    
    def _process_video(self) -> Tuple[str, str]:
        """Extract frames from video file."""
        import cv2
        
        self.video_name = self.input_path.stem
        
        # Determine output directory for frames
        if self.output_dir:
            self.frames_dir = self.output_dir / "JPEGImages" / self.video_name
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
            self.frames_dir = Path(self.temp_dir) / "JPEGImages" / self.video_name
        
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.input_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Extracting frames to: {self.frames_dir}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame with 7-digit zero-padded filename (TubeletGraph format)
            frame_path = self.frames_dir / f"{frame_idx:07d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"    Extracted {frame_idx}/{total_frames} frames...")
        
        cap.release()
        self.num_frames = frame_idx
        print(f"  ✓ Extracted {frame_idx} frames")
        
        # First frame (always frame 0)
        self.first_frame_path = self.frames_dir / "0000000.jpg"
        
        # Selected frame for segmentation (may be different from frame 0)
        selected_idx = min(self.frame_index, frame_idx - 1) if frame_idx > 0 else 0
        self.selected_frame_path = self.frames_dir / f"{selected_idx:07d}.jpg"
        
        if selected_idx != 0:
            print(f"  Using frame {selected_idx} for object detection/segmentation")
        
        return str(self.selected_frame_path), str(self.frames_dir)
    
    def _process_folder(self) -> Tuple[str, str]:
        """Process existing folder of frames."""
        self.video_name = self.input_path.name
        self.frames_dir = self.input_path
        
        # Find image files in folder
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(self.input_path.glob(f"*{ext}"))
            image_files.extend(self.input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.input_path}")
        
        # Sort by filename
        image_files = sorted(image_files, key=lambda x: x.name)
        self.num_frames = len(image_files)
        
        print(f"Folder: {self.input_path}")
        print(f"  Found {self.num_frames} frames")
        
        # First frame (always index 0)
        self.first_frame_path = image_files[0]
        
        # Selected frame for segmentation
        selected_idx = min(self.frame_index, len(image_files) - 1)
        self.selected_frame_path = image_files[selected_idx]
        
        if selected_idx != 0:
            print(f"  Using frame {selected_idx} for object detection/segmentation")
        
        # If output_dir specified, copy frames to proper structure
        if self.output_dir:
            new_frames_dir = self.output_dir / "JPEGImages" / self.video_name
            if new_frames_dir != self.frames_dir:
                new_frames_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Copying frames to: {new_frames_dir}")
                
                for i, frame_path in enumerate(image_files):
                    new_path = new_frames_dir / f"{i:07d}.jpg"
                    if frame_path.suffix.lower() in {'.jpg', '.jpeg'}:
                        shutil.copy2(frame_path, new_path)
                    else:
                        # Convert to JPEG
                        img = Image.open(frame_path).convert('RGB')
                        img.save(new_path, 'JPEG', quality=95)
                
                self.frames_dir = new_frames_dir
                self.first_frame_path = new_frames_dir / "0000000.jpg"
                self.selected_frame_path = new_frames_dir / f"{selected_idx:07d}.jpg"
        
        return str(self.selected_frame_path), str(self.frames_dir)
    
    def _process_image(self) -> Tuple[str, str]:
        """Process single image file."""
        self.video_name = self.input_path.stem
        self.first_frame_path = self.input_path
        self.selected_frame_path = self.input_path
        self.num_frames = 1
        
        print(f"Image: {self.input_path}")
        
        # If output_dir specified, copy to proper structure
        if self.output_dir:
            self.frames_dir = self.output_dir / "JPEGImages" / self.video_name
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            
            new_path = self.frames_dir / "0000000.jpg"
            if self.input_path.suffix.lower() in {'.jpg', '.jpeg'}:
                shutil.copy2(self.input_path, new_path)
            else:
                img = Image.open(self.input_path).convert('RGB')
                img.save(new_path, 'JPEG', quality=95)
            
            self.first_frame_path = new_path
            self.selected_frame_path = new_path
        else:
            self.frames_dir = self.input_path.parent
        
        return str(self.selected_frame_path), str(self.frames_dir)
    
    def get_frame_path(self, frame_idx: int) -> Optional[str]:
        """Get path to a specific frame by index."""
        if self.frames_dir is None:
            return None
        frame_path = Path(self.frames_dir) / f"{frame_idx:07d}.jpg"
        if frame_path.exists():
            return str(frame_path)
        return None
    
    def cleanup(self):
        """Clean up temporary directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"\nCleaned up temporary directory: {self.temp_dir}")
    
    def get_annotation_dir(self) -> Path:
        """Get the annotation directory path."""
        if self.output_dir:
            return self.output_dir / "Annotations" / self.video_name
        elif self.frames_dir:
            return Path(self.frames_dir).parent.parent / "Annotations" / self.video_name
        else:
            return Path(self.first_frame_path).parent


# ==============================================================================
# VOS Mask Utilities
# ==============================================================================

def get_vos_palette() -> List[int]:
    """Generate VOS-compatible color palette."""
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
    palette.extend([255, 255, 255])
    return palette


def save_vos_mask(mask: np.ndarray, output_path: str) -> None:
    """Save mask in VOS format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    mask_img.putpalette(get_vos_palette())
    mask_img.save(str(output_path))
    
    vis_path = output_path.parent / f"{output_path.stem}_vis.png"
    mask_img.convert('RGB').save(str(vis_path))
    
    unique = np.unique(mask)
    obj_ids = unique[(unique > 0) & (unique < 255)]
    print(f"\n{'='*50}")
    print(f"✓ Saved mask: {output_path}")
    print(f"✓ Saved visualization: {vis_path}")
    print(f"{'='*50}")
    print(f"Objects segmented: {len(obj_ids)}")
    for oid in obj_ids:
        pct = 100 * np.sum(mask == oid) / mask.size
        print(f"  Object {oid}: {pct:.1f}% of image")


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get media type from image path."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ==============================================================================
# VLM Object Detection
# ==============================================================================

class VLMObjectDetector:
    """Use Vision-Language Models to identify objects in images."""
    
    DETECTION_PROMPT = """Analyze this image and list ALL distinct objects/items visible.

For each object, provide a SHORT, SPECIFIC description that could be used to identify it.
Focus on:
- Main objects (not background elements)
- Objects that could be tracked in a video
- Use specific descriptions (e.g., "red apple" not just "fruit")

Return ONLY a JSON array of object descriptions, nothing else.
Example format: ["red apple being cut", "kitchen knife with black handle", "green cutting board", "whole red apple"]

Be specific and descriptive. List each distinct object separately."""

    def __init__(self, vlm_type: str = "claude"):
        self.vlm_type = vlm_type.lower()
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate API keys and dependencies."""
        if self.vlm_type == "claude":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Run:\n"
                    "  export ANTHROPIC_API_KEY='your-key'"
                )
        elif self.vlm_type == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY not set. Run:\n"
                    "  export OPENAI_API_KEY='your-key'"
                )
        elif self.vlm_type == "ollama":
            # Check if ollama is running
            try:
                import requests
                requests.get("http://localhost:11434/api/tags", timeout=2)
            except:
                raise ValueError(
                    "Ollama not running. Start with:\n"
                    "  ollama serve\n"
                    "  ollama pull llava"
                )
    
    def detect_objects(self, image_path: str) -> List[str]:
        """Detect objects in image using VLM."""
        print(f"\nAnalyzing image with {self.vlm_type.upper()}...")
        
        if self.vlm_type == "claude":
            return self._detect_with_claude(image_path)
        elif self.vlm_type == "openai":
            return self._detect_with_openai(image_path)
        elif self.vlm_type == "ollama":
            return self._detect_with_ollama(image_path)
        else:
            raise ValueError(f"Unknown VLM type: {self.vlm_type}")
    
    def _detect_with_claude(self, image_path: str) -> List[str]:
        """Use Claude for object detection."""
        import anthropic
        
        client = anthropic.Anthropic()
        
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.DETECTION_PROMPT
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text
        return self._parse_object_list(response_text)
    
    def _detect_with_openai(self, image_path: str) -> List[str]:
        """Use GPT-4V for object detection."""
        import openai
        
        client = openai.OpenAI()
        
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.DETECTION_PROMPT
                        }
                    ],
                }
            ],
            max_tokens=1024,
        )
        
        response_text = response.choices[0].message.content
        return self._parse_object_list(response_text)
    
    def _detect_with_ollama(self, image_path: str) -> List[str]:
        """Use local LLaVA via Ollama for object detection."""
        import requests
        
        image_data = encode_image_base64(image_path)
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": self.DETECTION_PROMPT,
                "images": [image_data],
                "stream": False
            },
            timeout=120
        )
        
        response_text = response.json()["response"]
        return self._parse_object_list(response_text)
    
    def _parse_object_list(self, response: str) -> List[str]:
        """Parse JSON array from VLM response."""
        # Try to extract JSON array from response
        try:
            # Find JSON array in response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                objects = json.loads(match.group())
                if isinstance(objects, list):
                    return [str(obj).strip() for obj in objects if obj]
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse line by line
        lines = response.strip().split('\n')
        objects = []
        for line in lines:
            # Remove common prefixes
            line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line.strip())
            line = re.sub(r'^"(.*)"$', r'\1', line)
            if line and len(line) > 2 and len(line) < 100:
                objects.append(line)
        
        return objects


# ==============================================================================
# SAM3 Segmentation
# ==============================================================================

class SAM3Segmenter:
    """Segment objects using SAM3 with text prompts."""
    
    # Alternative phrasings to try for common difficult objects
    PROMPT_VARIATIONS = {
        "gripper": ["robot gripper", "robotic arm", "metal gripper", "parallel gripper", "robot end effector"],
        "metallic": ["metal", "silver", "aluminum", "steel"],
        "robotic": ["robot", "mechanical", "automated"],
    }
    
    def __init__(self):
        self._model = None
        self._processor = None
    
    def _ensure_loaded(self):
        """Lazy load SAM3."""
        if self._model is not None:
            return
        
        print("\nLoading SAM3 model...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self._model = build_sam3_image_model()
            self._processor = Sam3Processor(self._model)
            print("✓ SAM3 loaded successfully")
        except ImportError:
            raise RuntimeError(
                "SAM3 not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e .\n"
                "  huggingface-cli login"
            )
    
    def _generate_prompt_variations(self, prompt: str) -> List[str]:
        """Generate alternative phrasings for a prompt."""
        variations = [prompt]
        prompt_lower = prompt.lower()
        
        # Add variations based on keywords
        for keyword, alternatives in self.PROMPT_VARIATIONS.items():
            if keyword in prompt_lower:
                for alt in alternatives:
                    variations.append(prompt_lower.replace(keyword, alt))
        
        # Also try simpler versions
        words = prompt.split()
        if len(words) > 2:
            # Try last two words
            variations.append(" ".join(words[-2:]))
            # Try first two words  
            variations.append(" ".join(words[:2]))
        
        return list(dict.fromkeys(variations))  # Remove duplicates, preserve order
    
    def segment(
        self, 
        image_path: str, 
        object_prompts: List[str],
        threshold: float = 0.3,
        retry_with_variations: bool = True,
        min_threshold: float = 0.1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Segment objects using text prompts.
        
        Args:
            image_path: Path to image
            object_prompts: List of object descriptions
            threshold: Initial confidence threshold
            retry_with_variations: Try alternative phrasings if initial fails
            min_threshold: Minimum threshold to try before giving up
        
        Returns:
            (mask_array, metadata_dict)
        """
        self._ensure_loaded()
        
        image = Image.open(image_path).convert("RGB")
        H, W = np.array(image).shape[:2]
        
        state = self._processor.set_image(image)
        
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        obj_id = 1
        segmented_objects = []
        failed_objects = []
        
        for prompt in object_prompts:
            print(f"  Segmenting: '{prompt}'...")
            
            found = False
            best_mask = None
            best_score = 0
            best_prompt = prompt
            
            # Generate variations to try
            prompts_to_try = [prompt]
            if retry_with_variations:
                prompts_to_try = self._generate_prompt_variations(prompt)
            
            # Try each prompt variation
            for try_prompt in prompts_to_try:
                if found:
                    break
                    
                try:
                    output = self._processor.set_text_prompt(state=state, prompt=try_prompt)
                    
                    masks = output["masks"]
                    scores = output["scores"]
                    
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().numpy()
                    
                    for i, score in enumerate(scores):
                        if score >= threshold:
                            mask = masks[i]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            if mask.ndim == 3:
                                mask = mask.squeeze()
                            
                            if score > best_score:
                                best_mask = mask
                                best_score = float(score)
                                best_prompt = try_prompt
                                found = True
                        elif score >= min_threshold and score > best_score:
                            # Keep track of best sub-threshold result
                            mask = masks[i]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            if mask.ndim == 3:
                                mask = mask.squeeze()
                            best_mask = mask
                            best_score = float(score)
                            best_prompt = try_prompt
                            
                except Exception as e:
                    if try_prompt == prompt:  # Only print error for main prompt
                        print(f"    ✗ Error: {e}")
            
            # Use best result if found
            if found:
                combined_mask[best_mask > 0.5] = obj_id
                segmented_objects.append({
                    "id": obj_id,
                    "prompt": prompt,
                    "matched_prompt": best_prompt,
                    "score": best_score,
                    "pixels": int(np.sum(best_mask > 0.5))
                })
                if best_prompt != prompt:
                    print(f"    ✓ Found with '{best_prompt}' (score: {best_score:.3f}) -> Object ID {obj_id}")
                else:
                    print(f"    ✓ Found (score: {best_score:.3f}) -> Object ID {obj_id}")
                obj_id += 1
            elif best_mask is not None and best_score >= min_threshold:
                # Use sub-threshold result with warning
                combined_mask[best_mask > 0.5] = obj_id
                segmented_objects.append({
                    "id": obj_id,
                    "prompt": prompt,
                    "matched_prompt": best_prompt,
                    "score": best_score,
                    "pixels": int(np.sum(best_mask > 0.5)),
                    "low_confidence": True
                })
                print(f"    ⚠ Found with LOW confidence (score: {best_score:.3f}) -> Object ID {obj_id}")
                obj_id += 1
            else:
                failed_objects.append({"prompt": prompt, "best_score": best_score})
                print(f"    ✗ Not found (best score: {best_score:.3f})")
        
        metadata = {
            "total_objects": obj_id - 1,
            "objects": segmented_objects,
            "failed": failed_objects
        }
        
        return combined_mask, metadata


# ==============================================================================
# Interactive Object Selection
# ==============================================================================

def interactive_select(objects: List[str]) -> List[str]:
    """Let user interactively select which objects to segment."""
    print("\n" + "="*50)
    print("DETECTED OBJECTS")
    print("="*50)
    
    for i, obj in enumerate(objects, 1):
        print(f"  [{i}] {obj}")
    
    print("\n" + "-"*50)
    print("Options:")
    print("  Enter numbers separated by commas (e.g., 1,3,5)")
    print("  Enter 'all' to segment all objects")
    print("  Enter 'q' to quit")
    print("-"*50)
    
    while True:
        choice = input("\nSelect objects to segment: ").strip().lower()
        
        if choice == 'q':
            return []
        
        if choice == 'all':
            return objects
        
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(objects):
                    selected.append(objects[idx - 1])
                else:
                    print(f"  Invalid index: {idx}")
            
            if selected:
                print(f"\nSelected: {selected}")
                confirm = input("Confirm? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected
        except ValueError:
            print("  Invalid input. Enter numbers separated by commas.")


# ==============================================================================
# Visualization with OpenCV
# ==============================================================================

def visualize_detections(image_path: str, objects: List[str]) -> List[str]:
    """Visualize detected objects and let user select via GUI."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, using text-based selection")
        return interactive_select(objects)
    
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    
    # Create selection state
    selected = [False] * len(objects)
    
    window_name = "Select Objects to Segment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, W), min(900, H + 200))
    
    def get_display():
        # Create display with object list
        display = image.copy()
        
        # Add object list at bottom
        list_height = min(30 * len(objects) + 60, 300)
        canvas = np.ones((H + list_height, W, 3), dtype=np.uint8) * 40
        canvas[:H, :W] = display
        
        # Draw object list
        y_offset = H + 30
        cv2.putText(canvas, "Click number to toggle | 'a'=all | 'c'=clear | 'Enter'=confirm | 'q'=quit",
                   (10, H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, (obj, sel) in enumerate(zip(objects, selected)):
            color = (0, 255, 0) if sel else (150, 150, 150)
            marker = "[X]" if sel else "[ ]"
            text = f"{i+1}. {marker} {obj[:50]}"
            cv2.putText(canvas, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return canvas
    
    print("\nGUI Selection Window opened")
    print("  Press number keys (1-9) to toggle objects")
    print("  Press 'a' to select all, 'c' to clear")
    print("  Press Enter to confirm, 'q' to quit")
    
    while True:
        display = get_display()
        cv2.imshow(window_name, display)
        key = cv2.waitKey(50) & 0xFF
        
        # Number keys 1-9
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(objects):
                selected[idx] = not selected[idx]
                status = "selected" if selected[idx] else "deselected"
                print(f"  {objects[idx]}: {status}")
        
        elif key == ord('a'):
            selected = [True] * len(objects)
            print("  Selected all")
        
        elif key == ord('c'):
            selected = [False] * len(objects)
            print("  Cleared all")
        
        elif key == 13:  # Enter
            break
        
        elif key == ord('q'):
            selected = [False] * len(objects)
            break
    
    cv2.destroyAllWindows()
    
    return [obj for obj, sel in zip(objects, selected) if sel]


# ==============================================================================
# Create TubeletGraph Dataset Structure
# ==============================================================================

def create_dataset_structure(
    input_processor: InputProcessor,
    mask: np.ndarray,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Create complete TubeletGraph-compatible dataset structure.
    
    Structure:
        <output_dir>/
        ├── JPEGImages/<video_name>/
        │   ├── 0000000.jpg
        │   └── ...
        ├── Annotations/<video_name>/
        │   └── 0000000.png
        └── splits/
            └── val.txt
    """
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(input_processor.frames_dir).parent.parent
    
    video_name = input_processor.video_name
    
    # Annotation directory
    anno_dir = base_dir / "Annotations" / video_name
    anno_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mask
    mask_path = anno_dir / "0000000.png"
    save_vos_mask(mask, str(mask_path))
    
    # Create splits directory and file
    splits_dir = base_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    split_file = splits_dir / "val.txt"
    existing = []
    if split_file.exists():
        existing = split_file.read_text().strip().split('\n')
    if video_name not in existing:
        with open(split_file, 'a') as f:
            f.write(f"{video_name}\n")
    
    return {
        "base_dir": str(base_dir),
        "frames_dir": str(input_processor.frames_dir),
        "anno_dir": str(anno_dir),
        "mask_path": str(mask_path),
        "split_file": str(split_file),
        "video_name": video_name,
        "num_frames": input_processor.num_frames,
        "fps": input_processor.fps
    }


# ==============================================================================
# Multi-Frame Scanning
# ==============================================================================

def scan_frames_for_objects(
    input_processor: InputProcessor,
    object_prompts: List[str],
    segmenter: SAM3Segmenter,
    num_frames_to_scan: int = 5,
    threshold: float = 0.3
) -> Tuple[int, np.ndarray, Dict]:
    """
    Scan multiple frames to find the best frame where all objects are visible.
    
    Args:
        input_processor: The input processor with extracted frames
        object_prompts: List of object descriptions to find
        segmenter: SAM3 segmenter instance
        num_frames_to_scan: Number of frames to sample
        threshold: Confidence threshold
    
    Returns:
        (best_frame_index, mask, metadata)
    """
    if input_processor.num_frames <= 1:
        # Only one frame, use it
        frame_path = input_processor.get_frame_path(0) or str(input_processor.selected_frame_path)
        mask, metadata = segmenter.segment(frame_path, object_prompts, threshold)
        return 0, mask, metadata
    
    # Sample frames evenly across the video
    frame_indices = []
    step = max(1, input_processor.num_frames // num_frames_to_scan)
    for i in range(0, input_processor.num_frames, step):
        frame_indices.append(i)
        if len(frame_indices) >= num_frames_to_scan:
            break
    
    print(f"\nScanning {len(frame_indices)} frames to find best segmentation...")
    print(f"  Frame indices: {frame_indices}")
    
    best_frame_idx = 0
    best_mask = None
    best_metadata = None
    best_score = -1
    
    for frame_idx in frame_indices:
        frame_path = input_processor.get_frame_path(frame_idx)
        if frame_path is None:
            continue
        
        print(f"\n  Frame {frame_idx}:")
        mask, metadata = segmenter.segment(
            frame_path, 
            object_prompts, 
            threshold,
            retry_with_variations=True,
            min_threshold=0.1
        )
        
        # Score based on: number of objects found and their confidence
        num_found = metadata["total_objects"]
        avg_score = 0
        if metadata["objects"]:
            avg_score = sum(obj["score"] for obj in metadata["objects"]) / len(metadata["objects"])
        
        # Penalize low confidence detections
        low_conf_count = sum(1 for obj in metadata["objects"] if obj.get("low_confidence", False))
        
        # Combined score
        frame_score = num_found * 10 + avg_score * 5 - low_conf_count * 2
        
        print(f"    Found {num_found}/{len(object_prompts)} objects, avg_score={avg_score:.3f}, frame_score={frame_score:.2f}")
        
        if frame_score > best_score:
            best_score = frame_score
            best_frame_idx = frame_idx
            best_mask = mask
            best_metadata = metadata
        
        # If we found all objects with good confidence, stop early
        if num_found == len(object_prompts) and avg_score > 0.5:
            print(f"    ✓ Found all objects with good confidence, using this frame")
            break
    
    print(f"\n  Best frame: {best_frame_idx} (score: {best_score:.2f})")
    return best_frame_idx, best_mask, best_metadata


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate VOS masks using VLM + SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses a Vision-Language Model to identify objects,
then SAM3 to segment them. No manual object naming required!

Input Types Supported:
  - Video files: .mp4, .avi, .mov, .mkv, etc.
  - Image files: .jpg, .png, etc.
  - Frame folders: Directory containing image sequence

VLM Options:
  claude  : Anthropic Claude (requires ANTHROPIC_API_KEY)
  openai  : OpenAI GPT-4V (requires OPENAI_API_KEY)  
  ollama  : Local LLaVA via Ollama (requires: ollama serve && ollama pull llava)

Examples:
  # From video file - detect and segment all objects
  python generate_mask_grounded.py -i video.mp4 --vlm claude --auto
  
  # From video file - interactively select objects
  python generate_mask_grounded.py -i video.mp4 --vlm claude
  
  # From image file
  python generate_mask_grounded.py -i frame.jpg --vlm claude --auto
  
  # From frame folder
  python generate_mask_grounded.py -i ./frames/ --vlm claude --auto
  
  # Specify output directory (creates TubeletGraph structure)
  python generate_mask_grounded.py -i video.mp4 --vlm claude --auto --output_dir ./dataset
  
  # Use local LLaVA (free, no API key needed)
  python generate_mask_grounded.py -i video.mp4 --vlm ollama --auto
        """
    )
    
    parser.add_argument("--input", "-i", required=True,
                       help="Input: video file (.mp4), image file (.jpg), or frame folder")
    parser.add_argument("--output_dir", "-o", default=None,
                       help="Output directory (creates TubeletGraph structure)")
    parser.add_argument("--vlm", "-v", default="claude",
                       choices=["claude", "openai", "ollama"],
                       help="Vision-Language Model to use")
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Automatically segment all detected objects")
    parser.add_argument("--threshold", "-t", type=float, default=0.1,
                       help="SAM3 confidence threshold (default: 0.3, try 0.1 for difficult objects)")
    parser.add_argument("--gui", "-g", action="store_true",
                       help="Use GUI for object selection (requires OpenCV)")
    parser.add_argument("--keep_frames", "-k", action="store_true",
                       help="Keep extracted frames (don't delete temp directory)")
    parser.add_argument("--frame", "-f", type=int, default=0,
                       help="Frame index to use for detection/segmentation (default: 0)")
    parser.add_argument("--scan_frames", "-s", action="store_true",
                       help="Scan multiple frames to find best segmentation for all objects")
    parser.add_argument("--num_scan_frames", type=int, default=5,
                       help="Number of frames to scan when using --scan_frames (default: 5)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)
    
    print("="*60)
    print("VLM-Grounded SAM3 Mask Generator")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir or '(auto)'}")
    print(f"VLM: {args.vlm}")
    print(f"Auto-segment all: {args.auto}")
    print(f"Frame index: {args.frame}")
    print(f"Scan frames: {args.scan_frames}")
    print(f"Threshold: {args.threshold}")
    
    # Step 1: Process input (video/image/folder)
    input_processor = InputProcessor(args.input, args.output_dir, frame_index=args.frame)
    
    try:
        first_frame_path, frames_dir = input_processor.process()
        
        # Step 2: Detect objects with VLM
        detector = VLMObjectDetector(args.vlm)
        objects = detector.detect_objects(first_frame_path)
            
        if not objects:
            print("\nNo objects detected in the image.")
            sys.exit(1)
        
        print(f"\n✓ Detected {len(objects)} objects:")

        # store detected object list
        

        for i, obj in enumerate(objects, 1):
            print(f"  {i}. {obj}")
        
        # Step 3: Select objects to segment
        if args.auto:
            selected_objects = objects
            print(f"\nAuto-selecting all {len(objects)} objects")

        elif args.gui:
            selected_objects = visualize_detections(first_frame_path, objects)
        else:
            selected_objects = interactive_select(objects)
        
        if not selected_objects:
            print("\nNo objects selected. Exiting.")
            sys.exit(0)
        
        print(f"\nWill segment {len(selected_objects)} objects:")
        for obj in selected_objects:
            print(f"  - {obj}")
        
        # Step 4: Segment with SAM3
        print("\n" + "-"*50)
        print("SEGMENTATION")
        print("-"*50)
        
        segmenter = SAM3Segmenter()
        
        if args.scan_frames and input_processor.num_frames > 1:
            # Scan multiple frames to find best segmentation
            best_frame_idx, mask, metadata = scan_frames_for_objects(
                input_processor,
                selected_objects,
                segmenter,
                num_frames_to_scan=args.num_scan_frames,
                threshold=args.threshold
            )
            # Update the selected frame path for mask saving
            frame_path = input_processor.get_frame_path(best_frame_idx)
            if frame_path:
                first_frame_path = frame_path
        else:
            # Use single frame
            mask, metadata = segmenter.segment(
                first_frame_path,
                selected_objects,
                threshold=args.threshold,
                retry_with_variations=True,
                min_threshold=0.1
            )
        
        # Step 5: Save result and create dataset structure
        if np.any(mask > 0):
            dataset_info = create_dataset_structure(
                input_processor,
                mask,
                args.output_dir
            )
            
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Input type: {input_processor.input_type}")
            print(f"Video/Image name: {dataset_info['video_name']}")
            print(f"Frames: {dataset_info['num_frames']}")
            print(f"FPS: {dataset_info['fps']:.2f}")
            print(f"\nRequested: {len(selected_objects)} objects")
            print(f"Segmented: {metadata['total_objects']} objects")
            for obj in metadata['objects']:
                low_conf = " (LOW CONFIDENCE)" if obj.get('low_confidence', False) else ""
                matched = f" [matched: '{obj['matched_prompt']}']" if obj.get('matched_prompt') != obj['prompt'] else ""
                print(f"  ID {obj['id']}: '{obj['prompt']}' (score: {obj['score']:.3f}){matched}{low_conf}")
            
            # Show failed objects and suggestions
            if metadata.get('failed'):
                print(f"\nFailed to segment ({len(metadata['failed'])} objects):")
                for obj in metadata['failed']:
                    print(f"  - '{obj['prompt']}' (best score: {obj['best_score']:.3f})")
                print("\nSuggestions for failed objects:")
                print("  1. Try --frame N to use a different frame where object is more visible")
                print("  2. Try --scan_frames to automatically find best frame")
                print("  3. Try --threshold 0.1 for lower confidence threshold")
                print("  4. Use interactive mode: generate_mask_sam3.py --mode click")
            
            print(f"\nOutput files:")
            print(f"  Frames: {dataset_info['frames_dir']}")
            print(f"  Mask: {dataset_info['mask_path']}")
            print(f"  Split: {dataset_info['split_file']}")
            
            print(f"\nTo run TubeletGraph:")
            print(f"  python quick_run.py \\")
            print(f"      --input_dir {dataset_info['frames_dir']} \\")
            print(f"      --input_mask {dataset_info['mask_path']} \\")
            print(f"      --fps {int(dataset_info['fps'])}")
        else:
            print("\nNo objects could be segmented. Try:")
            print("  - Using more specific descriptions")
            print("  - Lowering the threshold with --threshold 0.2")
            print("  - Using a different VLM")
    
    finally:
        # Cleanup temp directory if not keeping frames
        if not args.keep_frames and not args.output_dir:
            input_processor.cleanup()


if __name__ == "__main__":
    main()