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
    
    # Find all instances of a repeated object (e.g. multiple pipettes)
    python generate_mask_grounded.py --input frame.jpg --vlm claude --auto --max_instances 8
    
    # Find all pipettes, asking VLM for a tight crop first (recommended)
    python generate_mask_grounded.py --input frame.jpg --vlm claude --auto --max_instances 8 --bbox_padding 0.05
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
from PIL import Image, ImageDraw
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

def _create_with_param_fallback(create, **kwargs):
    """
    Call an OpenAI create() endpoint, retrying without parameters the model rejects.

    Reasoning models in the GPT-5 family accept only the default temperature and
    want max_completion_tokens rather than max_tokens. Sending the greedy setting
    first keeps temperature=0 wherever it is supported — the parsers downstream
    are brittle regexes and do not tolerate sampling — and gives it up only where
    the API actually refuses it.
    """
    attempt = dict(kwargs)
    # The API reports rejected parameters one at a time, so drop them in a loop
    # rather than assuming a single retry is enough.
    for _ in range(4):
        try:
            return create(**attempt)
        except Exception as exc:
            msg = str(exc)
            changed = False
            if "temperature" in msg and "temperature" in attempt:
                attempt.pop("temperature")
                changed = True
            if "max_tokens" in msg and "max_tokens" in attempt:
                attempt["max_completion_tokens"] = attempt.pop("max_tokens")
                changed = True
            if not changed:
                raise
    return create(**attempt)


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

    # Prompt template for bounding-box localisation of a repeated object class.
    # The image dimensions are injected at call time so the model can reason in
    # pixel space and we can sanity-check the returned values.
    BBOX_PROMPT_TEMPLATE = """Find a tight bounding box that encloses ALL visible instances of: "{object_desc}"

Rules:
- Include every single instance, even partially visible ones.
- Make the box containing all instances.
- Use NORMALIZED coordinates from 0 to 100, where (0, 0) is the top-left corner and (100, 100) is the bottom-right corner.
- Do NOT use absolute pixel coordinates.
- If no instance is visible, return null.

Return ONLY a JSON object with this exact schema, nothing else:
{{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}}

Where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."""

    def __init__(self, vlm_type: str = "openai",
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        self.vlm_type = vlm_type.lower()
        self.base_url = base_url or os.environ.get("OVAD_VLM_BASE_URL") or None
        self.model = model or os.environ.get("OVAD_VLM_MODEL") or (
            "gpt-4o" if self.vlm_type == "openai" else "Qwen/Qwen3-VL-32B-Instruct")
        self._cached_client = None
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate API keys and dependencies."""
        if self.vlm_type == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY not set. Run:\n"
                    "  export OPENAI_API_KEY='your-key'"
                )
        elif self.vlm_type == "qwen":
            if not self.base_url:
                raise ValueError(
                    "The 'qwen' backbone needs an OpenAI-compatible server. Start one:\n"
                    "  vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000\n"
                    "then set vlm.base_url in the config or $OVAD_VLM_BASE_URL.")
        else:
            raise ValueError(
                f"Unknown VLM backend '{self.vlm_type}' (supported: openai, qwen)")
    
    def detect_objects(self, image_path: str) -> List[str]:
        """Detect objects in image using VLM."""
        print(f"\nAnalyzing image with {self.vlm_type.upper()}...")
        
        if self.vlm_type == "openai":
            return self._detect_with_openai(image_path)
        elif self.vlm_type == "qwen":
            return self._detect_with_qwen(image_path)
        else:
            raise ValueError(
                f"Unknown VLM backend '{self.vlm_type}' (supported: openai, qwen)")
    
    def _detect_with_openai(self, image_path: str) -> List[str]:
        """Use GPT-4o for object detection."""
        return self._parse_object_list(
            self._chat(image_path, self.DETECTION_PROMPT, max_tokens=1024))

    def _client(self):
        """OpenAI-protocol client for this backbone (built once)."""
        if getattr(self, "_cached_client", None) is None:
            import openai
            base_url = os.environ.get("OVAD_VLM_BASE_URL") or self.base_url
            api_key = os.environ.get("OPENAI_API_KEY")
            kwargs = {"timeout": 120.0, "max_retries": 2}
            if base_url:
                kwargs["base_url"] = base_url
                kwargs["api_key"] = api_key or "EMPTY"
            else:
                kwargs["api_key"] = api_key
            self._cached_client = openai.OpenAI(**kwargs)
        return self._cached_client

    def _chat(self, image_path: str, prompt: str, max_tokens: int) -> str:
        """
        Single-image chat/completions call.

        Used by both backbones: 'openai' hits the OpenAI API, 'qwen' hits an
        OpenAI-compatible server (e.g. local vLLM). temperature=0 keeps decoding
        greedy, matching the previous local-transformers behaviour — the parsers
        downstream are brittle regexes and do not tolerate sampling.
        """
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        response = _create_with_param_fallback(
            self._client().chat.completions.create,
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    def _detect_with_qwen(self, image_path: str) -> List[str]:
        """Object detection via an OpenAI-compatible server (e.g. local vLLM)."""
        return self._parse_object_list(
            self._chat(image_path, self.DETECTION_PROMPT, max_tokens=1024))

    def locate_group_bbox(
        self,
        image_path: str,
        object_desc: str,
        padding: float = 0.1,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Ask the VLM to find a bounding box that encloses ALL instances of
        *object_desc* in the image.

        Parameters
        ----------
        image_path  : path to the source image
        object_desc : natural-language description of the object class,
                      e.g. "pipette" or "red apple"
        padding     : extra margin added on each side as a fraction of the
                      image dimension (default 0.05 = 5 %).  Keeps SAM3 from
                      losing context right at the crop edge.

        Returns
        -------
        (x1, y1, x2, y2) in pixel coordinates clipped to the image bounds, or
        None if the VLM reports no instances are visible.
        """
        img = Image.open(image_path)
        W, H = img.size

        # We no longer pass W and H into the prompt because we are using normalized 0-100 coords
        prompt = self.BBOX_PROMPT_TEMPLATE.format(
            object_desc=object_desc
        )

        print(f"\n  [bbox] Asking {self.vlm_type.upper()} for group bounding box of '{object_desc}'…")

        if self.vlm_type == "openai":
            raw = self._bbox_with_openai(image_path, prompt)
        elif self.vlm_type == "qwen":
            raw = self._bbox_with_qwen(image_path, prompt)
        else:
            raise ValueError(
                f"Unknown VLM backend '{self.vlm_type}' (supported: openai, qwen)")

        bbox = self._parse_bbox(raw, W, H, padding)
        if bbox:
            x1, y1, x2, y2 = bbox
            print(f"  [bbox] Raw box: ({x1},{y1}) → ({x2},{y2})  "
                  f"[{x2-x1}×{y2-y1} px]")
        else:
            print(f"  [bbox] VLM returned no bounding box – will use full image.")
        return bbox

    def _parse_bbox(
        self,
        response: str,
        img_w: int,
        img_h: int,
        padding: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract (x1,y1,x2,y2) from the VLM response, map from normalized 0-100 coords 
        to absolute pixels, apply padding, and clip to image bounds.  Returns None if 
        the response contains "null" or cannot be parsed.
        """
        text = response.strip()

        # Model said there are no instances
        if re.search(r'\bnull\b', text, re.IGNORECASE):
            return None

        # Try to parse a JSON object with x1/y1/x2/y2 keys
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())

                # Models vary in how they package the four numbers. Accept the
                # documented {"x1":..,"y1":..,"x2":..,"y2":..} as well as the
                # common variants where all four arrive in one list — e.g.
                # {"x1": [216, 300, 783, 716]} or {"bbox": [...]}, both observed
                # from Qwen3-VL. Without this the parse raises and the caller
                # silently falls back to the full image, quietly disabling the
                # bbox-cropped segmentation path.
                if not all(k in data for k in ("x1", "y1", "x2", "y2")):
                    flat = None
                    for v in data.values():
                        if isinstance(v, (list, tuple)) and len(v) == 4:
                            flat = v
                            break
                    if flat is None:
                        raise ValueError(f"unrecognised bbox payload: {data}")
                    data = dict(zip(("x1", "y1", "x2", "y2"), flat))
                elif isinstance(data.get("x1"), (list, tuple)) and len(data["x1"]) == 4:
                    data = dict(zip(("x1", "y1", "x2", "y2"), data["x1"]))

                raw = [float(data["x1"]), float(data["y1"]),
                       float(data["x2"]), float(data["y2"])]

                # The prompt asks for a normalized 0-100 grid, but not every
                # model obeys it — Qwen3-VL returns absolute pixels. Any value
                # above 100 cannot be a 0-100 coordinate, so treat the whole box
                # as pixels in that case. Without this the box is scaled by
                # img/100 a second time and comes out degenerate, and the caller
                # silently falls back to the full image.
                if max(raw) > 100.0:
                    x1, y1, x2, y2 = (int(v) for v in raw)
                else:
                    x1 = int(raw[0] * img_w / 100.0)
                    y1 = int(raw[1] * img_h / 100.0)
                    x2 = int(raw[2] * img_w / 100.0)
                    y2 = int(raw[3] * img_h / 100.0)

                # The model might accidentally swap corners – normalise
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Apply padding
                pad_x = int(padding * img_w)
                pad_y = int(padding * img_h)
                x1 = max(0,     x1 - pad_x)
                y1 = max(0,     y1 - pad_y)
                x2 = min(img_w, x2 + pad_x)
                y2 = min(img_h, y2 + pad_y)

                # Sanity: box must be non-degenerate and reasonably sized
                if x2 - x1 < 4 or y2 - y1 < 4:
                    print(f"  [bbox] Parsed box is degenerate ({x1},{y1},{x2},{y2}) – ignoring.")
                    return None

                return (x1, y1, x2, y2)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            print(f"  [bbox] Could not parse bbox JSON ({exc}): {text[:120]}")

        return None

    def _bbox_with_openai(self, image_path: str, prompt: str) -> str:
        return self._chat(image_path, prompt, max_tokens=256)

    def _bbox_with_qwen(self, image_path: str, prompt: str) -> str:
        return self._chat(image_path, prompt, max_tokens=256)

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

def _to_numpy_mask(mask_tensor) -> np.ndarray:
    """Convert a SAM3 mask tensor/array to a 2-D boolean numpy array."""
    m = mask_tensor
    if hasattr(m, 'cpu'):
        m = m.cpu().numpy()
    m = np.array(m)
    # Squeeze away leading channel dims (e.g. shape (1,H,W) → (H,W))
    while m.ndim > 2:
        m = m.squeeze(0)
    return m


class SAM3Segmenter:
    """Segment objects using SAM3 with text prompts."""
    
    # Alternative phrasings to try for common difficult objects
    PROMPT_VARIATIONS = {
        "gripper": ["robot gripper", "robotic arm", "metal gripper", "parallel gripper", "robot end effector"],
        "metallic": ["metal", "silver", "aluminum", "steel"],
        "robotic": ["robot", "mechanical", "automated"],
    }

    # Minimum fraction of total image pixels a valid instance mask must cover.
    # Masks smaller than this are assumed to be SAM noise / background bleed.
    MIN_MASK_AREA_FRAC: float = 0.0005

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

    # ------------------------------------------------------------------
    # Iterative instance discovery
    # ------------------------------------------------------------------

    def _apply_exclusion_mask(
        self,
        image: Image.Image,
        exclusion_mask: np.ndarray,
        fill_color: Tuple[int, int, int] = (128, 128, 128),
    ) -> Image.Image:
        """
        Return a copy of *image* with all pixels in *exclusion_mask* (bool or
        uint8>0) filled with *fill_color*.  SAM3 will therefore see the
        already-found instances as a neutral gray patch and segment the *next*
        distinct region instead.
        """
        img_arr = np.array(image).copy()
        binary = exclusion_mask > 0
        img_arr[binary] = fill_color
        return Image.fromarray(img_arr)

    def _set_image_from_array(self, image: Image.Image):
        """Re-initialise SAM3 processor state with a (possibly masked) image."""
        return self._processor.set_image(image)

    # ------------------------------------------------------------------
    # Crop / uncrop helpers for bbox-guided segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def crop_image(
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """Return the sub-image defined by *bbox* = (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def uncrop_mask(
        crop_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        full_h: int,
        full_w: int,
    ) -> np.ndarray:
        """
        Place *crop_mask* (H_crop × W_crop bool/uint8) back into a zero canvas
        of size (full_h × full_w) at the position described by *bbox*.

        This is a simple offset paste – no interpolation needed because the
        crop was taken without rescaling.
        """
        x1, y1, x2, y2 = bbox
        full_mask = np.zeros((full_h, full_w), dtype=crop_mask.dtype)
        # Clip destination region in case the crop was already clipped to
        # image bounds (should never differ, but defensive)
        dst_h = min(y2 - y1, full_h - y1)
        dst_w = min(x2 - x1, full_w - x1)
        full_mask[y1 : y1 + dst_h, x1 : x1 + dst_w] = \
            crop_mask[:dst_h, :dst_w]
        return full_mask

    def _all_masks_for_prompt(
        self,
        state,
        prompt: str,
        min_threshold: float,
        retry_with_variations: bool,
        total_pixels: int,
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Query SAM3 with *prompt* (and optionally prompt variations) against an
        already-set image state and return ALL candidate masks that pass basic
        quality filters.

        Unlike the old _best_mask_for_prompt, this collects every mask whose
        score >= min_threshold and whose area >= MIN_MASK_AREA_FRAC, across all
        prompt variations.  Deduplication across variations is done by area-IoU
        so that the same physical region is not returned twice under two
        different phrasings.

        Returns
        -------
        List of (mask_2d, score, matched_prompt), sorted by score descending.
        """
        prompts_to_try = [prompt]
        if retry_with_variations:
            prompts_to_try = self._generate_prompt_variations(prompt)

        candidates: List[Tuple[np.ndarray, float, str]] = []

        for try_prompt in prompts_to_try:
            try:
                output = self._processor.set_text_prompt(state=state, prompt=try_prompt)
                masks  = output["masks"]
                scores = output["scores"]

                if hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                scores = np.array(scores)

                for i, score in enumerate(scores):
                    if float(score) < min_threshold:
                        continue

                    candidate = _to_numpy_mask(masks[i])
                    binary    = candidate > 0.5
                    area_frac = float(binary.sum()) / max(total_pixels, 1)

                    if area_frac < self.MIN_MASK_AREA_FRAC:
                        continue

                    # Deduplicate against masks already collected from other
                    # prompt variations (IoU > 0.7 means same region).
                    is_dup = False
                    for prev_mask, _, _ in candidates:
                        inter = np.logical_and(binary, prev_mask).sum()
                        union = np.logical_or(binary,  prev_mask).sum()
                        if inter / max(union, 1) > 0.7:
                            is_dup = True
                            break
                    if not is_dup:
                        candidates.append((binary, float(score), try_prompt))

            except Exception as e:
                if try_prompt == prompt:
                    print(f"    ✗ Error with prompt '{try_prompt}': {e}")

        # Sort best-score first so callers can iterate in confidence order.
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def segment_instances(
        self,
        image_path: str,
        prompt: str,
        max_instances: int = 20,
        threshold: float = 0.3,
        min_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        retry_with_variations: bool = True,
        crop_bbox: Optional[Tuple[int, int, int, int]] = None,
        max_iterations: int = 10,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Find all instances of a single object class described by *prompt*.

        Strategy (new overlap-based approach)
        --------------------------------------
        Each iteration runs SAM3 on the ORIGINAL (unmodified) image so that
        previously found instances remain fully visible and can influence the
        model's attention.  All candidate masks returned by SAM3 in that
        iteration are collected and filtered against the already-accepted set
        using IoU:

          IoU <= iou_threshold  →  novel instance, accept it
          IoU >  iou_threshold  →  duplicate of a known instance, discard it

        This means a small amount of physical overlap between two real objects
        is allowed (they share a pixel border, say), while masks that are
        largely the same region as one already accepted are rejected.

        The loop runs for at most *max_iterations* iterations (hard cap = 10).
        It stops early as soon as the accepted count reaches *max_instances*.
        If an entire iteration yields zero new accepted masks the search has
        converged and exits immediately.

        Parameters
        ----------
        image_path        : path to the source image
        prompt            : text description of the object class
        max_instances     : target number of instances (early-stop threshold)
        threshold         : "good" confidence level, used only for log labels
        min_threshold     : candidates whose score < this are discarded
        iou_threshold     : max IoU a candidate may share with any accepted
                            mask and still be counted as a new instance
                            (default 0.3 – small overlap is fine)
        retry_with_variations : try alternate phrasings to widen the candidate pool
        crop_bbox         : optional (x1,y1,x2,y2) in full-image pixels; SAM3
                            operates on this crop and masks are unprojected back
        max_iterations    : hard cap on SAM3 call count (default 10)

        Returns
        -------
        (masks, scores) – masks are 2-D boolean np.ndarrays in FULL-IMAGE space.
        """
        self._ensure_loaded()

        full_image = Image.open(image_path).convert("RGB")
        full_H, full_W = np.array(full_image).shape[:2]

        # ---- Determine working image (cropped or full) ----------------------
        if crop_bbox is not None:
            x1c, y1c, x2c, y2c = crop_bbox
            base_image = self.crop_image(full_image, crop_bbox)
            print(f"  [crop] Operating on crop ({x1c},{y1c})→({x2c},{y2c})  "
                  f"[{x2c-x1c}×{y2c-y1c} px]")
        else:
            base_image = full_image

        H, W         = np.array(base_image).shape[:2]
        total_pixels = H * W

        accepted_masks:  List[np.ndarray] = []   # binary masks in CROP space
        accepted_scores: List[float]      = []

        print(f"\n  [iterative] Searching for instances of '{prompt}' "
              f"(max_instances={max_instances}, max_iter={max_iterations}, "
              f"iou_thresh={iou_threshold}, min_score={min_threshold})")

        # Set the image state ONCE – we never modify the image between iterations
        state = self._set_image_from_array(base_image)

        for iteration in range(max_iterations):
            # ---- Collect all candidate masks from SAM3 ----------------------
            candidates = self._all_masks_for_prompt(
                state=state,
                prompt=prompt,
                min_threshold=min_threshold,
                retry_with_variations=retry_with_variations,
                total_pixels=total_pixels,
            )

            if not candidates:
                print(f"    [iter {iteration+1}] No candidates above threshold – stopping.")
                break

            # ---- Filter candidates by IoU against accepted set --------------
            newly_accepted = 0
            for cand_mask, cand_score, matched_prompt in candidates:
                if len(accepted_masks) >= max_instances:
                    break

                # Compute max IoU of this candidate against every accepted mask
                max_iou = 0.0
                for prev in accepted_masks:
                    inter = np.logical_and(cand_mask, prev).sum()
                    union = np.logical_or(cand_mask,  prev).sum()
                    iou   = inter / max(union, 1)
                    if iou > max_iou:
                        max_iou = iou

                if max_iou > iou_threshold:
                    # Too much overlap with an already-accepted mask → duplicate
                    continue

                # Accept this candidate as a new instance
                conf_label   = "✓" if cand_score >= threshold else "⚠ (low conf)"
                matched_note = (f" [via '{matched_prompt}']"
                                if matched_prompt != prompt else "")
                print(f"    [iter {iteration+1}] #{len(accepted_masks)+1} "
                      f"{conf_label} score={cand_score:.3f}{matched_note} "
                      f"area={100*cand_mask.sum()/total_pixels:.2f}% "
                      f"max_iou_vs_accepted={max_iou:.2f}")

                accepted_masks.append(cand_mask)
                accepted_scores.append(cand_score)
                newly_accepted += 1

            print(f"    [iter {iteration+1}] +{newly_accepted} new  "
                  f"total={len(accepted_masks)}/{max_instances}")

            # Early-stop: target reached
            if len(accepted_masks) >= max_instances:
                print(f"  [iterative] Reached max_instances={max_instances} – stopping early.")
                break

            # Early-stop: iteration added nothing new (converged)
            if newly_accepted == 0:
                print(f"  [iterative] No new instances in iter {iteration+1} – converged.")
                break

        print(f"  [iterative] Found {len(accepted_masks)} instance(s) of '{prompt}'")

        # ---- Unproject masks back to full-image space -----------------------
        if crop_bbox is not None:
            found_masks_full = [
                self.uncrop_mask(m, crop_bbox, full_H, full_W)
                for m in accepted_masks
            ]
        else:
            found_masks_full = accepted_masks

        return found_masks_full, accepted_scores

    # ------------------------------------------------------------------
    # Original single-instance segment() kept intact for backward compat
    # ------------------------------------------------------------------

    def segment(
        self, 
        image_path: str, 
        object_prompts: List[str],
        threshold: float = 0.3,
        retry_with_variations: bool = True,
        min_threshold: float = 0.1,
        max_instances: int = 1,
        iou_threshold: float = 0.3,
        crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Segment objects using text prompts.

        When *max_instances* > 1 the overlap-based iterative loop is used for
        every prompt, discovering all distinct instances of the same object
        class.  SAM3 always sees the unmodified image; new candidates are
        accepted when their IoU with every already-accepted mask is <=
        *iou_threshold*.

        When *crop_bbox* is provided (x1,y1,x2,y2 in full-image pixels), SAM3
        operates on that crop region only.  All returned masks are in
        full-image coordinates regardless.
        
        Args:
            image_path        : Path to image
            object_prompts    : List of object descriptions
            threshold         : Confidence threshold for "high confidence" label
            retry_with_variations : Try alternative phrasings if initial fails
            min_threshold     : Discard candidates with score below this
            max_instances     : Target number of instances per prompt
            iou_threshold     : Max IoU a candidate may share with any accepted
                                mask and still count as a new instance (default
                                0.3).  Raise to allow more overlap; lower to be
                                stricter about duplicates.
            crop_bbox         : Optional (x1,y1,x2,y2) crop region.
        
        Returns:
            (mask_array, metadata_dict)  –  mask_array is always full-image size.
        """
        self._ensure_loaded()
        
        image = Image.open(image_path).convert("RGB")
        H, W = np.array(image).shape[:2]
        
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        obj_id = 1
        segmented_objects = []
        failed_objects = []
        
        for prompt in object_prompts:
            print(f"  Segmenting: '{prompt}'...")

            if max_instances > 1:
                # ---- Multi-instance path ------------------------------------
                masks, scores = self.segment_instances(
                    image_path=image_path,
                    prompt=prompt,
                    max_instances=max_instances,
                    threshold=threshold,
                    min_threshold=min_threshold,
                    iou_threshold=iou_threshold,
                    retry_with_variations=retry_with_variations,
                    crop_bbox=crop_bbox,
                )

                if not masks:
                    failed_objects.append({"prompt": prompt, "best_score": 0.0})
                    print(f"    ✗ No instances found for '{prompt}'")
                    continue

                for inst_mask, inst_score in zip(masks, scores):
                    combined_mask[inst_mask] = obj_id
                    low_conf = inst_score < threshold
                    segmented_objects.append({
                        "id":             obj_id,
                        "prompt":         prompt,
                        "matched_prompt": prompt,
                        "score":          inst_score,
                        "pixels":         int(inst_mask.sum()),
                        "low_confidence": low_conf,
                    })
                    obj_id += 1

            else:
                # ---- Original single-instance path --------------------------
                # Apply crop if requested
                if crop_bbox is not None:
                    work_image  = self.crop_image(image, crop_bbox)
                    cx1, cy1, cx2, cy2 = crop_bbox
                else:
                    work_image  = image

                state = self._processor.set_image(work_image)
                found      = False
                best_mask  = None
                best_score = 0.0
                best_prompt = prompt

                prompts_to_try = [prompt]
                if retry_with_variations:
                    prompts_to_try = self._generate_prompt_variations(prompt)

                for try_prompt in prompts_to_try:
                    if found:
                        break
                        
                    try:
                        output = self._processor.set_text_prompt(state=state, prompt=try_prompt)
                        
                        masks_out  = output["masks"]
                        scores_out = output["scores"]
                        
                        if hasattr(scores_out, 'cpu'):
                            scores_out = scores_out.cpu().numpy()
                        
                        for i, score in enumerate(scores_out):
                            if score >= threshold:
                                mask = _to_numpy_mask(masks_out[i])
                                if score > best_score:
                                    best_mask   = mask
                                    best_score  = float(score)
                                    best_prompt = try_prompt
                                    found = True
                            elif score >= min_threshold and score > best_score:
                                mask = _to_numpy_mask(masks_out[i])
                                best_mask   = mask
                                best_score  = float(score)
                                best_prompt = try_prompt
                                
                    except Exception as e:
                        if try_prompt == prompt:
                            print(f"    ✗ Error: {e}")

                if found or (best_mask is not None and best_score >= min_threshold):
                    # Unproject mask to full-image space if we used a crop
                    if crop_bbox is not None:
                        best_mask = self.uncrop_mask(
                            best_mask > 0.5, crop_bbox, H, W
                        ).astype(best_mask.dtype)

                    if found:
                        combined_mask[best_mask > 0.5] = obj_id
                        segmented_objects.append({
                            "id":             obj_id,
                            "prompt":         prompt,
                            "matched_prompt": best_prompt,
                            "score":          best_score,
                            "pixels":         int(np.sum(best_mask > 0.5))
                        })
                        if best_prompt != prompt:
                            print(f"    ✓ Found with '{best_prompt}' (score: {best_score:.3f}) -> Object ID {obj_id}")
                        else:
                            print(f"    ✓ Found (score: {best_score:.3f}) -> Object ID {obj_id}")
                        obj_id += 1
                    else:
                        combined_mask[best_mask > 0.5] = obj_id
                        segmented_objects.append({
                            "id":             obj_id,
                            "prompt":         prompt,
                            "matched_prompt": best_prompt,
                            "score":          best_score,
                            "pixels":         int(np.sum(best_mask > 0.5)),
                            "low_confidence": True
                        })
                        print(f"    ⚠ Found with LOW confidence (score: {best_score:.3f}) -> Object ID {obj_id}")
                        obj_id += 1
                else:
                    failed_objects.append({"prompt": prompt, "best_score": best_score})
                    print(f"    ✗ Not found (best score: {best_score:.3f})")
        
        metadata = {
            "total_objects": obj_id - 1,
            "objects":       segmented_objects,
            "failed":        failed_objects
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
        "base_dir":    str(base_dir),
        "frames_dir":  str(input_processor.frames_dir),
        "anno_dir":    str(anno_dir),
        "mask_path":   str(mask_path),
        "split_file":  str(split_file),
        "video_name":  video_name,
        "num_frames":  input_processor.num_frames,
        "fps":         input_processor.fps
    }


# ==============================================================================
# Multi-Frame Scanning
# ==============================================================================

def scan_frames_for_objects(
    input_processor: InputProcessor,
    object_prompts: List[str],
    segmenter: SAM3Segmenter,
    detector: "VLMObjectDetector",
    num_frames_to_scan: int = 5,
    threshold: float = 0.3,
    max_instances: int = 8,
    iou_threshold: float = 0.5,
    bbox_padding: float = 0.1,
    bbox_object_desc: Optional[str] = None,
) -> Tuple[int, np.ndarray, Dict]:
    """
    Scan multiple frames to find the best result.

    For each sampled frame the function:
      1. Calls the VLM to obtain a tight bounding box that encloses all
         instances of the target object class in *that specific frame*.
      2. Crops the frame to that box (with padding) and runs SAM3 on the crop.
      3. Unprojects the resulting masks back to full-image coordinates.

    The best frame is chosen as the one whose accepted masks cover the largest
    total pixel area in the full image – this naturally favours frames where
    objects are most prominent and fully visible.

    Args:
        input_processor   : The input processor with extracted frames.
        object_prompts    : List of object descriptions to segment.
        segmenter         : SAM3 segmenter instance.
        detector          : VLMObjectDetector used for per-frame bbox calls.
        num_frames_to_scan: Number of frames to sample evenly across the clip.
        threshold         : SAM3 confidence threshold.
        max_instances     : Max instances per prompt.
        iou_threshold     : IoU threshold for duplicate filtering.
        bbox_padding      : Padding fraction added around the VLM bbox crop.
        bbox_object_desc  : Object description forwarded to locate_group_bbox.
                            Defaults to the first element of object_prompts.

    Returns:
        (best_frame_index, mask, metadata)
        mask is in full-image coordinates; metadata includes per-object info
        plus a "crop_bbox" key showing the VLM bbox used for that frame.
    """
    # Fall back to the first prompt as the bbox query description
    if bbox_object_desc is None:
        bbox_object_desc = object_prompts[0]

    if input_processor.num_frames <= 1:
        # Single-frame input: run bbox + SAM on that one frame
        frame_path = (input_processor.get_frame_path(0)
                      or str(input_processor.selected_frame_path))
        crop_bbox = detector.locate_group_bbox(
            frame_path,
            object_desc=bbox_object_desc,
            padding=bbox_padding,
        )
        mask, metadata = segmenter.segment(
            frame_path, object_prompts, threshold,
            max_instances=max_instances,
            iou_threshold=iou_threshold,
            crop_bbox=crop_bbox,
        )
        metadata["crop_bbox"] = crop_bbox
        return 0, mask, metadata

    # Sample frames evenly across the video
    frame_indices = []
    step = max(1, input_processor.num_frames // num_frames_to_scan)
    for i in range(0, input_processor.num_frames, step):
        frame_indices.append(i)
        if len(frame_indices) >= num_frames_to_scan:
            break

    print(f"\nScanning {len(frame_indices)} frames "
          f"(per-frame VLM bbox + SAM3)...")
    print(f"  Frame indices: {frame_indices}")

    best_frame_idx  = 0
    best_mask       = None
    best_metadata   = None
    best_total_area = -1

    for frame_idx in frame_indices:
        frame_path = input_processor.get_frame_path(frame_idx)
        if frame_path is None:
            continue

        print(f"\n  Frame {frame_idx}:")

        # Step 1: Ask the VLM for a bbox on THIS frame
        crop_bbox = detector.locate_group_bbox(
            frame_path,
            object_desc=bbox_object_desc,
            padding=bbox_padding,
        )

        # Step 2: Run SAM3 on the (possibly cropped) frame
        mask, metadata = segmenter.segment(
            frame_path,
            object_prompts,
            threshold,
            retry_with_variations=True,
            min_threshold=0.1,
            max_instances=max_instances,
            iou_threshold=iou_threshold,
            crop_bbox=crop_bbox,
        )
        metadata["crop_bbox"] = crop_bbox

        # Step 3: Score = total accepted mask area in full-image pixels
        total_area = int((mask > 0).sum())
        num_found  = metadata["total_objects"]
        print(f"    Found {num_found} instance(s), "
              f"total_mask_area={total_area} px  "
              f"crop={'yes' if crop_bbox else 'full image'}")

        if total_area > best_total_area:
            best_total_area = total_area
            best_frame_idx  = frame_idx
            best_mask       = mask
            best_metadata   = metadata

    print(f"\n  Best frame: {best_frame_idx} "
          f"(total_mask_area={best_total_area} px)")
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

  # Find all 8 pipettes – VLM crops to their region first, then SAM3 iterates
  python generate_mask_grounded.py -i lab.jpg --vlm claude --auto --max_instances 8
  
  # Increase crop padding if SAM3 misses instances near the crop edge
  python generate_mask_grounded.py -i lab.jpg --vlm claude --auto --max_instances 8 --bbox_padding 0.08
        """
    )
    
    parser.add_argument("--input", "-i", required=True,
                       help="Input: video file (.mp4), image file (.jpg), or frame folder")
    parser.add_argument("--output_dir", "-o", default=None,
                       help="Output directory (creates TubeletGraph structure)")
    parser.add_argument("--bbox_object_desc", type=str, default=None,
                       help="Object class to localise before cropping (e.g. "
                            "'multi-channel pipette'). Omit to segment the full frame.")
    parser.add_argument("--vlm", "-v", default="openai",
                       choices=["openai", "qwen"],
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
    parser.add_argument("--max_instances", "-m", type=int, default=8,
                       help=(
                           "Maximum number of instances to find per object prompt. "
                           "Set > 1 to enable iterative mask-exclusion loop that finds "
                           "all copies of the same object class (e.g. --max_instances 8 "
                           "to find all 8 pipettes). Default: 1 (original behaviour)."
                       ))
    parser.add_argument("--bbox_padding", type=float, default=0.05,
                       help=(
                           "Padding fraction added around the VLM-returned group bounding "
                           "box before cropping for SAM3 (default: 0.05 = 5%%). "
                           "Only used when --max_instances > 1."
                       ))
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                       help=(
                           "IoU threshold for duplicate detection during multi-instance "
                           "segmentation. A candidate mask is accepted as a new instance "
                           "only if its IoU with every already-accepted mask is <= this "
                           "value. Lower = stricter (fewer duplicates, may miss touching "
                           "objects). Higher = more permissive (finds touching objects, "
                           "may include near-duplicates). Default: 0.3. Only used when "
                           "--max_instances > 1."
                       ))
    
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
    print(f"Max instances per prompt: {args.max_instances}")
    if args.max_instances > 1:
        print(f"Bbox padding: {args.bbox_padding:.0%}")
        print(f"IoU threshold: {args.iou_threshold}")
    
    # Step 1: Process input (video/image/folder)
    input_processor = InputProcessor(args.input, args.output_dir, frame_index=args.frame)
    
    try:
        first_frame_path, frames_dir = input_processor.process()
        
        # Step 2: Detect objects with VLM
        detector = VLMObjectDetector(args.vlm)
        # objects = detector.detect_objects(first_frame_path)
            
        # if not objects:
        #     print("\nNo objects detected in the image.")
        #     sys.exit(1)
        
        # print(f"\n✓ Detected {len(objects)} objects:")
        # for i, obj in enumerate(objects, 1):
        #     print(f"  {i}. {obj}")
        objects = ["single-channel pipette"]
        
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
        if args.max_instances > 1:
            print(f"  (up to {args.max_instances} instances each)")
        
        # Step 3b: For the non-scan path, ask the VLM for a bbox on the
        # selected frame.  When scan_frames is active, bbox localisation is
        # done per-frame inside scan_frames_for_objects instead.
        crop_bboxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
        for obj in selected_objects:
            if obj not in crop_bboxes:
                bbox = detector.locate_group_bbox(
                    first_frame_path,
                    object_desc=obj,
                    padding=args.bbox_padding,
                )
                crop_bboxes[obj] = bbox

        # Step 4: Segment with SAM3
        print("\n" + "-"*50)
        print("SEGMENTATION")
        print("-"*50)
        
        segmenter = SAM3Segmenter()
        
        if args.scan_frames and input_processor.num_frames > 1:
            # Per-frame VLM bbox + SAM3 scan: the VLM is called once per
            # sampled frame to get a tight crop for that frame, then SAM3
            # segments the crop.  The best frame is chosen by total mask area.
            best_frame_idx, mask, metadata = scan_frames_for_objects(
                input_processor,
                selected_objects,
                segmenter,
                detector=detector,
                num_frames_to_scan=args.num_scan_frames,
                threshold=args.threshold,
                max_instances=args.max_instances,
                iou_threshold=args.iou_threshold,
                bbox_padding=args.bbox_padding,
                bbox_object_desc=args.bbox_object_desc,
            )
            frame_path = input_processor.get_frame_path(best_frame_idx)
            if frame_path:
                first_frame_path = frame_path  # We are now looking at the best frame
                
            # Expose the winning frame's crop bbox in the summary
            winning_bbox = metadata.get("crop_bbox")
            if winning_bbox:
                crop_bboxes[selected_objects[0]] = winning_bbox
        else:
            if args.max_instances > 1 and crop_bboxes:
                # Segment each prompt separately so each gets its own bbox,
                # then merge the resulting masks into one combined mask.
                image_tmp  = Image.open(first_frame_path).convert("RGB")
                H_full, W_full = np.array(image_tmp).shape[:2]
                combined_mask = np.zeros((H_full, W_full), dtype=np.uint8)
                all_objects   = []
                all_failed    = []
                obj_id_offset = 1

                for obj in selected_objects:
                    bbox = crop_bboxes.get(obj)
                    part_mask, part_meta = segmenter.segment(
                        first_frame_path,
                        [obj],
                        threshold=args.threshold,
                        retry_with_variations=True,
                        min_threshold=0.1,
                        max_instances=args.max_instances,
                        iou_threshold=args.iou_threshold,
                        crop_bbox=bbox,
                    )
                    # Re-number object IDs so they don't collide across prompts
                    for seg_obj in part_meta["objects"]:
                        old_id  = seg_obj["id"]
                        new_id  = old_id + obj_id_offset - 1
                        combined_mask[part_mask == old_id] = new_id
                        seg_obj["id"] = new_id
                        all_objects.append(seg_obj)
                    all_failed.extend(part_meta.get("failed", []))
                    if part_meta["objects"]:
                        obj_id_offset += len(part_meta["objects"])

                mask     = combined_mask
                metadata = {
                    "total_objects": len(all_objects),
                    "objects":       all_objects,
                    "failed":        all_failed,
                }
            else:
                # Original single-pass path (max_instances == 1, no bbox)
                mask, metadata = segmenter.segment(
                    first_frame_path,
                    selected_objects,
                    threshold=args.threshold,
                    retry_with_variations=True,
                    min_threshold=0.1,
                    max_instances=args.max_instances,
                    iou_threshold=args.iou_threshold,
                )
        
        # Step 5: Save result and create dataset structure
        if np.any(mask > 0):
            dataset_info = create_dataset_structure(
                input_processor,
                mask,
                args.output_dir
            )

            # ------------------------------------------------------------------
            # ADDED: Save bounding box visualization on the original image
            # ------------------------------------------------------------------
            if crop_bboxes:
                try:
                    # Open original image and initialize drawing context
                    vis_img = Image.open(first_frame_path).convert("RGB")
                    draw = ImageDraw.Draw(vis_img)
                    
                    for obj_name, bbox in crop_bboxes.items():
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            # Draw the bounding box
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            # Draw the label above the bounding box
                            draw.text((x1, max(0, y1 - 15)), obj_name, fill="red")
                            
                    # Determine path to save (same dir as the generated mask)
                    out_dir = os.path.dirname(dataset_info['mask_path'])
                    bbox_vis_path = os.path.join(out_dir, "bbox_visualize.png")
                    
                    vis_img.save(bbox_vis_path)
                    dataset_info['bbox_visualize'] = bbox_vis_path
                except Exception as e:
                    print(f"\nWarning: Failed to save bbox visualization: {e}")
            # ------------------------------------------------------------------
            
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Input type: {input_processor.input_type}")
            print(f"Video/Image name: {dataset_info['video_name']}")
            print(f"Frames: {dataset_info['num_frames']}")
            print(f"FPS: {dataset_info['fps']:.2f}")
            print(f"\nRequested: {len(selected_objects)} object class(es)"
                  + (f" × up to {args.max_instances} instances each" if args.max_instances > 1 else ""))
            if crop_bboxes:
                for obj, bbox in crop_bboxes.items():
                    if bbox:
                        x1,y1,x2,y2 = bbox
                        print(f"  Crop for '{obj}': ({x1},{y1})→({x2},{y2}) [{x2-x1}×{y2-y1}px]")
                    else:
                        print(f"  Crop for '{obj}': none (full image used)")
            print(f"Segmented: {metadata['total_objects']} object(s) total")
            for obj in metadata['objects']:
                low_conf = " (LOW CONFIDENCE)" if obj.get('low_confidence', False) else ""
                matched  = (f" [matched: '{obj['matched_prompt']}']"
                            if obj.get('matched_prompt') != obj['prompt'] else "")
                print(f"  ID {obj['id']}: '{obj['prompt']}' "
                      f"(score: {obj['score']:.3f}){matched}{low_conf}")
            
            # Show failed objects and suggestions
            if metadata.get('failed'):
                print(f"\nFailed to segment ({len(metadata['failed'])} objects):")
                for obj in metadata['failed']:
                    print(f"  - '{obj['prompt']}' (best score: {obj['best_score']:.3f})")
                print("\nSuggestions for failed objects:")
                print("  1. Try --frame N to use a different frame where object is more visible")
                print("  2. Try --scan_frames to automatically find best frame")
                print("  3. Try --threshold 0.1 for lower confidence threshold")
                print("  4. Try --max_instances N if there are multiple copies of the object")
                print("  5. Use interactive mode: generate_mask_sam3.py --mode click")
            
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
            print("  - Lowering the threshold with --threshold 0.1")
            print("  - Using a different VLM")
    
    finally:
        # Cleanup temp directory if not keeping frames
        if not args.keep_frames and not args.output_dir:
            input_processor.cleanup()


if __name__ == "__main__":
    main()
    