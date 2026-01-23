#!/usr/bin/env python3
"""
VOST-TAS Object Discovery & Annotation Tool (Fixed for SAM3)
=============================================================

For when you DON'T know what objects are in your video frames.

INPUT: A folder containing JPEG frames (0000000.jpg, 0000001.jpg, ...)

Discovery approaches:
1. Auto (text-based) - Use generic prompts to find common objects
2. Interactive clicking - Click on objects, label them
3. Custom prompts - Provide your own text prompts

Usage:
------
    # Interactive mode - click on objects
    python discover_and_annotate.py -f ./JPEGImages/my_video -m interactive
    
    # Auto mode - find common objects automatically  
    python discover_and_annotate.py -f ./JPEGImages/my_video -m auto
    
    # Custom prompts mode
    python discover_and_annotate.py -f ./JPEGImages/my_video -m custom \
        --prompts "thing" "object" "item"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class DiscoveredObject:
    """An object discovered through segmentation."""
    temp_id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    centroid: Tuple[int, int]
    score: float = 0.0
    source_prompt: str = ""
    
    # To be filled after labeling
    final_id: Optional[int] = None
    label: Optional[str] = None
    track_through_video: bool = False
    is_tool: bool = False


# ==============================================================================
# Frame Utilities  
# ==============================================================================

def find_first_frame(frames_dir: str) -> str:
    """Find the first frame in a directory."""
    frames_dir = Path(frames_dir)
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Try common patterns
    patterns = [
        "0000000.jpg", "0000000.png",
        "000000.jpg", "000000.png", 
        "00000.jpg", "00000.png",
        "0000.jpg", "0000.png",
        "000.jpg", "000.png",
        "0.jpg", "0.png",
    ]
    
    for pattern in patterns:
        candidate = frames_dir / pattern
        if candidate.exists():
            return str(candidate)
    
    # Find first file by sorting
    files = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    if not files:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    
    return str(files[0])


# ==============================================================================
# SAM3 Object Discovery
# ==============================================================================

class SAM3Discoverer:
    """
    Object discovery using SAM3's text-based segmentation.
    
    SAM3 excels at finding objects from text descriptions.
    For discovery without knowing names, we use generic prompts.
    """
    
    # Generic prompts to discover common objects
    GENERIC_PROMPTS = [
        "object", "thing", "item",
        "tool", "utensil", "instrument",
        "food", "fruit", "vegetable",
        "container", "bowl", "plate", "cup",
        "person", "hand", "body part",
    ]
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Load SAM3 model."""
        if self._model is not None:
            return
        
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print("Loading SAM3...")
            self._model = build_sam3_image_model()
            self._processor = Sam3Processor(self._model)
            print("✓ SAM3 loaded")
            
        except ImportError as e:
            raise RuntimeError(
                f"SAM3 not found: {e}\n"
                "Install from: https://github.com/facebookresearch/sam3"
            )
    
    def discover_with_prompts(
        self,
        image_path: str,
        prompts: List[str],
        min_area: int = 500,
        score_threshold: float = 0.3
    ) -> List[DiscoveredObject]:
        """
        Discover objects using text prompts.
        
        Args:
            image_path: Path to image
            prompts: List of text prompts to try
            min_area: Minimum object size in pixels
            score_threshold: Minimum confidence score
        
        Returns:
            List of discovered objects
        """
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(image)
        
        H, W = np.array(image).shape[:2]
        
        all_objects = []
        seen_regions = np.zeros((H, W), dtype=bool)
        
        print(f"Trying {len(prompts)} prompts...")
        
        for prompt in prompts:
            try:
                output = self._processor.set_text_prompt(
                    state=state,
                    prompt=prompt
                )
                
                masks = output.get("masks", [])
                scores = output.get("scores", [])
                
                if len(masks) == 0:
                    continue
                
                # Convert to numpy
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score < score_threshold:
                        continue
                    
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    mask = mask.astype(bool)
                    
                    # Check minimum area
                    area = mask.sum()
                    if area < min_area:
                        continue
                    
                    # Check overlap with already found objects
                    overlap = (mask & seen_regions).sum() / area
                    if overlap > 0.5:  # >50% overlap, skip
                        continue
                    
                    # Calculate properties
                    ys, xs = np.where(mask)
                    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                    centroid = (int(np.mean(xs)), int(np.mean(ys)))
                    
                    obj = DiscoveredObject(
                        temp_id=len(all_objects) + 1,
                        mask=mask,
                        bbox=bbox,
                        area=int(area),
                        centroid=centroid,
                        score=float(score),
                        source_prompt=prompt
                    )
                    all_objects.append(obj)
                    seen_regions |= mask
                    
                    print(f"  '{prompt}': Found object {obj.temp_id} "
                          f"(score={score:.2f}, area={area:,})")
                    
            except Exception as e:
                print(f"  '{prompt}': Error - {e}")
                continue
        
        # Sort by area (largest first)
        all_objects.sort(key=lambda x: x.area, reverse=True)
        
        # Renumber
        for i, obj in enumerate(all_objects):
            obj.temp_id = i + 1
        
        print(f"✓ Discovered {len(all_objects)} objects")
        return all_objects
    
    def discover_auto(
        self,
        image_path: str,
        min_area: int = 500
    ) -> List[DiscoveredObject]:
        """
        Auto-discover using generic prompts.
        """
        return self.discover_with_prompts(
            image_path,
            self.GENERIC_PROMPTS,
            min_area=min_area
        )
    
    def discover_interactive(
        self,
        image_path: str,
        min_area: int = 500
    ) -> List[DiscoveredObject]:
        """
        Interactive discovery - click on objects, segment with SAM3.
        
        This uses SAM3's text prompt with location hint.
        """
        import cv2
        
        self._load_model()
        
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        H, W = image_np.shape[:2]
        
        state = self._processor.set_image(image_pil)
        
        discovered = []
        display_img = image_cv.copy()
        click_points = []
        
        def segment_region(x, y, prompt="object"):
            """Segment using text prompt, filter by proximity to click."""
            output = self._processor.set_text_prompt(
                state=state,
                prompt=prompt
            )
            
            masks = output.get("masks", [])
            scores = output.get("scores", [])
            
            if len(masks) == 0:
                return None, 0
            
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            
            # Find mask that contains the clicked point
            best_mask = None
            best_score = 0
            best_area = float('inf')
            
            for mask, score in zip(masks, scores):
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()
                mask = mask.astype(bool)
                
                # Check if click point is inside mask
                if mask[y, x]:
                    area = mask.sum()
                    # Prefer smaller masks that contain the point (more specific)
                    if best_mask is None or area < best_area:
                        best_mask = mask
                        best_score = float(score)
                        best_area = area
            
            return best_mask, best_score
        
        def on_mouse(event, x, y, flags, param):
            nonlocal discovered, display_img, click_points
            
            if event == cv2.EVENT_LBUTTONDOWN:
                click_points.append((x, y))
                print(f"\n  Click at ({x}, {y})")
                
                # Try different prompts to find best segmentation
                prompts_to_try = ["object", "thing", "item", "tool", "food"]
                
                best_mask = None
                best_score = 0
                best_prompt = ""
                
                for prompt in prompts_to_try:
                    mask, score = segment_region(x, y, prompt)
                    if mask is not None and score > best_score:
                        best_mask = mask
                        best_score = score
                        best_prompt = prompt
                
                if best_mask is None or best_mask.sum() < min_area:
                    print(f"    ✗ No valid mask found at this point")
                    return
                
                # Check overlap with existing objects
                for existing in discovered:
                    overlap = (best_mask & existing.mask).sum()
                    if overlap > 0.5 * best_mask.sum():
                        print(f"    ✗ Overlaps with object {existing.temp_id}")
                        return
                
                ys, xs = np.where(best_mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                centroid = (int(np.mean(xs)), int(np.mean(ys)))
                
                obj = DiscoveredObject(
                    temp_id=len(discovered) + 1,
                    mask=best_mask,
                    bbox=bbox,
                    area=int(best_mask.sum()),
                    centroid=centroid,
                    score=best_score,
                    source_prompt=best_prompt
                )
                discovered.append(obj)
                
                # Update display
                color = get_color_bgr(len(discovered))
                overlay = np.zeros_like(image_cv)
                overlay[best_mask] = color
                display_img = cv2.addWeighted(display_img, 1.0, overlay, 0.4, 0)
                
                cv2.putText(display_img, str(obj.temp_id),
                           (centroid[0]-10, centroid[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                print(f"    ✓ Object {obj.temp_id}: {obj.area:,} pixels")
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                if discovered:
                    removed = discovered.pop()
                    print(f"  ✗ Removed object {removed.temp_id}")
                    
                    # Redraw
                    display_img = image_cv.copy()
                    for obj in discovered:
                        color = get_color_bgr(obj.temp_id)
                        overlay = np.zeros_like(image_cv)
                        overlay[obj.mask] = color
                        display_img = cv2.addWeighted(display_img, 1.0, overlay, 0.4, 0)
                        cv2.putText(display_img, str(obj.temp_id),
                                   (obj.centroid[0]-10, obj.centroid[1]+10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        window = "Click to Segment | Q=Done | RightClick=Undo"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("="*50)
        print("  Left-click:  Segment object at point")
        print("  Right-click: Remove last object")
        print("  Q or ESC:    Finish")
        print("="*50)
        
        while True:
            cv2.imshow(window, display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        
        print(f"\n✓ Discovered {len(discovered)} objects")
        return discovered


# ==============================================================================
# Object Labeling
# ==============================================================================

class ObjectLabeler:
    """Label discovered objects."""
    
    def label_interactive(
        self,
        image_path: str,
        objects: List[DiscoveredObject]
    ) -> List[DiscoveredObject]:
        """Show each object and prompt for labels."""
        import cv2
        
        image = cv2.imread(image_path)
        
        print("\n" + "="*50)
        print("LABELING")
        print("="*50)
        print("For each object:")
        print("  - Enter a label (e.g., 'apple')")
        print("  - Say if it should be tracked (y/n)")
        print("  - Say if it's a tool (y/n)")
        print("  - Or type 'skip' to ignore")
        print("="*50)
        
        labeled = []
        next_id = 1
        
        for obj in objects:
            # Highlight this object
            display = image.copy()
            overlay = np.zeros_like(display)
            overlay[obj.mask] = (0, 255, 0)
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Show already labeled
            for prev in labeled:
                prev_overlay = np.zeros_like(display)
                prev_overlay[prev.mask] = get_color_bgr(prev.final_id)
                display = cv2.addWeighted(display, 1.0, prev_overlay, 0.2, 0)
            
            window = f"Object {obj.temp_id}"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.imshow(window, display)
            cv2.waitKey(100)
            
            print(f"\n--- Object {obj.temp_id} ---")
            print(f"    Area: {obj.area:,} px")
            print(f"    Found via: '{obj.source_prompt}'")
            
            label = input("    Label (or 'skip'): ").strip()
            
            if label.lower() == 'skip' or label == '':
                print("    → Skipped")
                cv2.destroyWindow(window)
                continue
            
            track = input("    Track? (y/n) [y]: ").strip().lower() != 'n'
            is_tool = input("    Tool? (y/n) [n]: ").strip().lower() == 'y'
            
            obj.label = label
            obj.track_through_video = track
            obj.is_tool = is_tool
            obj.final_id = next_id
            next_id += 1
            
            labeled.append(obj)
            print(f"    → ID {obj.final_id}: '{label}'")
            
            cv2.destroyWindow(window)
        
        cv2.destroyAllWindows()
        return labeled
    
    def label_cli(
        self,
        objects: List[DiscoveredObject],
        preview_path: str = None
    ) -> List[DiscoveredObject]:
        """CLI-only labeling."""
        print("\n" + "="*50)
        print("LABELING (CLI)")
        print("="*50)
        
        if preview_path:
            print(f"Preview: {preview_path}")
        
        print(f"\n{len(objects)} objects found:")
        for obj in objects:
            print(f"  [{obj.temp_id}] {obj.area:,} px, via '{obj.source_prompt}'")
        
        print("\nOptions:")
        print("  1. Type 'auto' to label all as object_1, object_2, ...")
        print("  2. Batch: '1:apple:y:n 2:knife:y:y' (id:label:track:tool)")
        print("  3. Press Enter for one-by-one")
        
        response = input("\nChoice: ").strip()
        
        if response.lower() == 'auto':
            for i, obj in enumerate(objects):
                obj.label = f"object_{i+1}"
                obj.final_id = i + 1
                obj.track_through_video = True
            return objects
        
        if ':' in response:
            labeled = []
            for item in response.split():
                parts = item.split(':')
                if len(parts) < 2:
                    continue
                tid = int(parts[0])
                for obj in objects:
                    if obj.temp_id == tid:
                        obj.label = parts[1]
                        obj.final_id = len(labeled) + 1
                        obj.track_through_video = len(parts) <= 2 or parts[2].lower() == 'y'
                        obj.is_tool = len(parts) > 3 and parts[3].lower() == 'y'
                        labeled.append(obj)
                        break
            return labeled
        
        # One by one
        labeled = []
        for obj in objects:
            print(f"\n[{obj.temp_id}] {obj.area:,} px")
            label = input("  Label (or skip): ").strip()
            if not label or label == 'skip':
                continue
            
            obj.label = label
            obj.track_through_video = input("  Track? [y]: ").strip().lower() != 'n'
            obj.is_tool = input("  Tool? [n]: ").strip().lower() == 'y'
            obj.final_id = len(labeled) + 1
            labeled.append(obj)
        
        return labeled


# ==============================================================================
# Utilities
# ==============================================================================

def get_color_bgr(idx: int) -> Tuple[int, int, int]:
    """Get distinct BGR color."""
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 0, 128), (0, 128, 0), (128, 0, 0),
    ]
    return colors[(idx - 1) % len(colors)]


def get_vos_palette() -> List[int]:
    """VOS color palette."""
    palette = [0, 0, 0]
    for i in range(1, 255):
        palette.extend([(i*67+100)%256, (i*137+80)%256, (i*37+120)%256])
    palette.extend([255, 255, 255])
    return palette


def save_vos_mask(objects: List[DiscoveredObject], path: str, shape: Tuple[int, int]):
    """Save as VOS-format indexed PNG."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    for obj in objects:
        if obj.final_id and obj.track_through_video:
            mask[obj.mask] = obj.final_id
    
    img = Image.fromarray(mask, mode='P')
    img.putpalette(get_vos_palette())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_preview(image_path: str, objects: List[DiscoveredObject], output_path: str):
    """Save preview with objects highlighted."""
    import cv2
    
    image = cv2.imread(image_path)
    
    for obj in objects:
        idx = obj.final_id if obj.final_id else obj.temp_id
        color = get_color_bgr(idx)
        
        overlay = np.zeros_like(image)
        overlay[obj.mask] = color
        image = cv2.addWeighted(image, 0.85, overlay, 0.15, 0)
        
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = str(obj.temp_id)
        if obj.label:
            label = f"{obj.final_id}: {obj.label}"
        cv2.putText(image, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, image)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Discover and annotate objects for TubeletGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  interactive  Click on objects to segment them
  auto         Try generic prompts to find objects
  custom       Use your own text prompts

Examples:
  python discover_and_annotate.py -f ./JPEGImages/my_video -m interactive
  python discover_and_annotate.py -f ./JPEGImages/my_video -m auto
  python discover_and_annotate.py -f ./JPEGImages/my_video -m custom --prompts "apple" "knife"
        """
    )
    
    parser.add_argument("--frames_dir", "-f", required=True,
                       help="Directory with JPEG frames")
    parser.add_argument("--output_dir", "-o", default=None,
                       help="Output directory")
    parser.add_argument("--mode", "-m", default="interactive",
                       choices=["interactive", "auto", "custom"])
    parser.add_argument("--prompts", "-p", nargs="+", default=None,
                       help="Custom prompts for 'custom' mode")
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--no-gui", action="store_true",
                       help="CLI-only labeling")
    
    args = parser.parse_args()
    
    frames_dir = Path(args.frames_dir)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif frames_dir.parent.name == "JPEGImages":
        output_dir = frames_dir.parent.parent / "Annotations" / frames_dir.name
    else:
        output_dir = frames_dir / "annotations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find first frame
    print("\n[1/4] Finding first frame...")
    first_frame = find_first_frame(str(frames_dir))
    print(f"✓ {first_frame}")
    
    image = Image.open(first_frame)
    H, W = image.size[1], image.size[0]
    
    # Discover
    print(f"\n[2/4] Discovering objects (mode: {args.mode})...")
    discoverer = SAM3Discoverer(args.device)
    
    if args.mode == "interactive":
        objects = discoverer.discover_interactive(first_frame)
    elif args.mode == "auto":
        objects = discoverer.discover_auto(first_frame)
    else:  # custom
        if not args.prompts:
            print("Error: --prompts required for custom mode")
            sys.exit(1)
        objects = discoverer.discover_with_prompts(first_frame, args.prompts)
    
    if not objects:
        print("No objects found!")
        sys.exit(1)
    
    # Save preview
    preview_path = output_dir / "discovery_preview.jpg"
    save_preview(first_frame, objects, str(preview_path))
    print(f"✓ Preview: {preview_path}")
    
    # Label
    print(f"\n[3/4] Labeling...")
    labeler = ObjectLabeler()
    
    if args.no_gui or args.mode != "interactive":
        labeled = labeler.label_cli(objects, str(preview_path))
    else:
        labeled = labeler.label_interactive(first_frame, objects)
    
    if not labeled:
        print("No objects labeled!")
        sys.exit(1)
    
    # Save
    print(f"\n[4/4] Saving...")
    
    frame_name = Path(first_frame).stem
    mask_path = output_dir / f"{frame_name}.png"
    save_vos_mask(labeled, str(mask_path), (H, W))
    print(f"✓ Mask: {mask_path}")
    
    # Save info
    info = {
        "frames_dir": str(frames_dir),
        "objects": [
            {
                "id": o.final_id,
                "label": o.label,
                "track": o.track_through_video,
                "is_tool": o.is_tool,
                "area": o.area
            }
            for o in labeled
        ]
    }
    info_path = output_dir / "objects_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Summary
    print("\n" + "="*50)
    print("DONE")
    print("="*50)
    print(f"\n{len(labeled)} objects:")
    for o in labeled:
        flags = []
        if o.track_through_video:
            flags.append("track")
        if o.is_tool:
            flags.append("tool")
        print(f"  {o.final_id}: {o.label} [{', '.join(flags)}]")
    
    print(f"\nRun TubeletGraph:")
    print(f"  python quick_run.py --input_dir {frames_dir} --input_mask {mask_path}")


if __name__ == "__main__":
    main()