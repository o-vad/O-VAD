#!/usr/bin/env python3
"""
SAM3 Mask Generator - No Object Names Required
==============================================

Generate VOS-compatible masks when you don't know the exact object names.

Methods available:
1. INTERACTIVE: Click on objects to segment them (recommended)
2. AUTOMATIC: Segment everything in the image, then select which masks to keep
3. BOX: Draw bounding boxes around objects of interest
4. GROUNDING: Use a vision-language model to describe what you see first

Requirements:
    conda create -n sam3 python=3.12
    pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3 && pip install -e .
    pip install opencv-python matplotlib
    huggingface-cli login

Usage:
    # Interactive mode - click on objects you want to track
    python generate_mask_interactive.py --image frame.jpg --mode interactive
    
    # Automatic mode - segment everything, then choose
    python generate_mask_interactive.py --image frame.jpg --mode auto
    
    # GUI box drawing mode
    python generate_mask_interactive.py --image frame.jpg --mode draw_boxes
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def get_vos_palette() -> List[int]:
    """Generate VOS-compatible color palette."""
    palette = [0, 0, 0]  # Background: black
    
    # Distinct colors for objects
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
    
    palette.extend([255, 255, 255])  # Void: white
    return palette


def save_vos_mask(mask: np.ndarray, output_path: str) -> None:
    """Save mask in VOS format (8-bit indexed PNG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    mask_img.putpalette(get_vos_palette())
    mask_img.save(str(output_path))
    
    # Also save colored version for visualization
    vis_path = output_path.parent / f"{output_path.stem}_vis.png"
    mask_img.convert('RGB').save(str(vis_path))
    
    # Print stats
    unique = np.unique(mask)
    obj_ids = unique[(unique > 0) & (unique < 255)]
    print(f"\n✓ Saved: {output_path}")
    print(f"  Objects: {len(obj_ids)} (IDs: {list(obj_ids)})")
    for oid in obj_ids:
        pct = 100 * np.sum(mask == oid) / mask.size
        print(f"    Object {oid}: {pct:.1f}% of image")


# ==============================================================================
# Method 1: Interactive Click-based Segmentation
# ==============================================================================

def interactive_click_mode(image_path: str, output_path: str, device: str = "cuda"):
    """
    Click on objects to segment them.
    
    Controls:
    - Left click: Add point to current object (positive)
    - Right click: Exclude region from current object (negative)  
    - 'n': Finish current object, start next
    - 's': Save current segmentation and show preview
    - 'r': Reset current object's points
    - 'u': Undo last saved object
    - 'q': Quit and save final mask
    """
    import cv2
    import torch
    
    print("\n" + "="*60)
    print("INTERACTIVE CLICK MODE")
    print("="*60)
    print("Controls:")
    print("  Left-click  : Include this point in object")
    print("  Right-click : Exclude this point from object")
    print("  'n'         : Confirm object & start next")
    print("  'r'         : Reset current object points")
    print("  'u'         : Undo last object")
    print("  'q'         : Quit and save")
    print("="*60 + "\n")
    
    # Load SAM3
    print("Loading SAM3...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    H, W = image_cv.shape[:2]
    
    # Set image in SAM3
    state = processor.set_image(image_pil)
    
    # State variables
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    current_obj_id = 1
    positive_points = []
    negative_points = []
    saved_objects = []  # List of (obj_id, mask) for undo
    
    # Display
    window_name = "SAM3 Interactive - Click to Segment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, W), min(720, H))
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal positive_points, negative_points
        if event == cv2.EVENT_LBUTTONDOWN:
            positive_points.append([x, y])
            print(f"  + Point ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            negative_points.append([x, y])
            print(f"  - Point ({x}, {y})")
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Create overlay image
    def get_display():
        display = image_cv.copy()
        
        # Draw existing mask overlay
        if np.any(combined_mask > 0):
            overlay = np.zeros_like(display)
            for oid in range(1, current_obj_id):
                if np.any(combined_mask == oid):
                    color = [(oid*67+100)%256, (oid*137+80)%256, (oid*37+120)%256]
                    overlay[combined_mask == oid] = color
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        
        # Draw current points
        for pt in positive_points:
            cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
            cv2.circle(display, tuple(pt), 8, (255, 255, 255), 2)
        for pt in negative_points:
            cv2.circle(display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.circle(display, tuple(pt), 8, (255, 255, 255), 2)
        
        # Add text
        cv2.putText(display, f"Object {current_obj_id} | Points: {len(positive_points)}+ {len(negative_points)}-",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "n=next | r=reset | u=undo | q=quit",
                   (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display
    
    print(f"\nClick on Object {current_obj_id}...")
    
    while True:
        display = get_display()
        cv2.imshow(window_name, display)
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('n') and positive_points:
            # Segment current object
            print(f"\n  Segmenting object {current_obj_id}...")
            
            all_points = positive_points + negative_points
            all_labels = [1]*len(positive_points) + [0]*len(negative_points)
            
            points_t = torch.tensor([all_points], dtype=torch.float32)
            labels_t = torch.tensor([all_labels], dtype=torch.int32)
            
            output = processor.set_point_prompt(state=state, points=points_t, labels=labels_t)
            
            mask = output["masks"][0]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            # Save for undo
            saved_objects.append((current_obj_id, mask.copy()))
            
            # Add to combined mask
            combined_mask[mask.astype(bool)] = current_obj_id
            
            print(f"  ✓ Object {current_obj_id} saved")
            
            current_obj_id += 1
            positive_points = []
            negative_points = []
            
            print(f"\nClick on Object {current_obj_id} (or 'q' to finish)...")
        
        elif key == ord('r'):
            positive_points = []
            negative_points = []
            print("  Points reset")
        
        elif key == ord('u') and saved_objects:
            # Undo last object
            last_id, last_mask = saved_objects.pop()
            combined_mask[combined_mask == last_id] = 0
            current_obj_id = last_id
            print(f"  Undid object {last_id}")
        
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Save final mask
    if np.any(combined_mask > 0):
        save_vos_mask(combined_mask, output_path)
    else:
        print("\nNo objects segmented. Mask not saved.")


# ==============================================================================
# Method 2: Automatic Segmentation (Segment Everything)
# ==============================================================================

def automatic_mode(image_path: str, output_path: str, device: str = "cuda"):
    """
    Automatically segment all objects, then let user select which to keep.
    
    Uses SAM3's automatic mask generation to find all possible objects,
    then displays them for selection.
    """
    import cv2
    import torch
    
    print("\n" + "="*60)
    print("AUTOMATIC SEGMENTATION MODE")
    print("="*60)
    print("Will segment all objects, then you select which to keep.")
    print("="*60 + "\n")
    
    # Load SAM3
    print("Loading SAM3...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    H, W = image_np.shape[:2]
    
    state = processor.set_image(image_pil)
    
    # Generate automatic masks using grid of points
    print("Generating automatic masks (this may take a moment)...")
    
    # Create grid of points
    grid_size = 32
    points = []
    for y in range(grid_size//2, H, grid_size):
        for x in range(grid_size//2, W, grid_size):
            points.append([x, y])
    
    all_masks = []
    all_scores = []
    
    # Process in batches
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        
        for pt in batch_points:
            try:
                points_t = torch.tensor([[pt]], dtype=torch.float32)
                labels_t = torch.tensor([[1]], dtype=torch.int32)
                
                output = processor.set_point_prompt(state=state, points=points_t, labels=labels_t)
                
                masks = output["masks"]
                scores = output["scores"]
                
                for j in range(len(masks)):
                    mask = masks[j]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    score = scores[j].item() if hasattr(scores[j], 'item') else float(scores[j])
                    
                    # Filter by size and score
                    area = np.sum(mask > 0.5)
                    if area > 100 and score > 0.7:  # Minimum area and score
                        all_masks.append(mask > 0.5)
                        all_scores.append(score)
            except:
                continue
        
        print(f"  Processed {min(i+batch_size, len(points))}/{len(points)} points, found {len(all_masks)} masks")
    
    if not all_masks:
        print("No masks found! Try interactive mode instead.")
        return
    
    # Remove duplicate/overlapping masks (keep highest score)
    print(f"\nFiltering {len(all_masks)} masks...")
    unique_masks = []
    unique_scores = []
    
    # Sort by score descending
    sorted_indices = np.argsort(all_scores)[::-1]
    
    for idx in sorted_indices:
        mask = all_masks[idx]
        score = all_scores[idx]
        
        # Check overlap with existing masks
        is_duplicate = False
        for existing_mask in unique_masks:
            intersection = np.sum(mask & existing_mask)
            union = np.sum(mask | existing_mask)
            iou = intersection / (union + 1e-6)
            
            if iou > 0.5:  # Too similar
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_masks.append(mask)
            unique_scores.append(score)
            
            if len(unique_masks) >= 20:  # Limit to top 20
                break
    
    print(f"Found {len(unique_masks)} unique objects")
    
    # Interactive selection
    print("\n" + "-"*40)
    print("SELECT OBJECTS TO INCLUDE")
    print("-"*40)
    print("Controls:")
    print("  Click on mask: Toggle selection")
    print("  'a': Select all")
    print("  'c': Clear all")
    print("  'q': Confirm and save")
    print("-"*40 + "\n")
    
    import cv2
    
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    selected = [False] * len(unique_masks)
    
    window_name = "Select Objects (click to toggle)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, W), min(720, H))
    
    def get_mask_at_point(x, y):
        for i, mask in enumerate(unique_masks):
            if mask[y, x]:
                return i
        return None
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = get_mask_at_point(x, y)
            if idx is not None:
                selected[idx] = not selected[idx]
                status = "selected" if selected[idx] else "deselected"
                print(f"  Object {idx+1} {status}")
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    def get_display():
        display = image_cv.copy()
        
        for i, (mask, is_selected) in enumerate(zip(unique_masks, selected)):
            if is_selected:
                color = [(i*67+100)%256, (i*137+80)%256, (i*37+120)%256]
                overlay = np.zeros_like(display)
                overlay[mask] = color
                display = cv2.addWeighted(display, 1, overlay, 0.5, 0)
                
                # Draw number
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    cv2.putText(display, str(i+1), (cx, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            else:
                # Show unselected masks with outline
                contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (128, 128, 128), 1)
        
        n_selected = sum(selected)
        cv2.putText(display, f"Selected: {n_selected}/{len(unique_masks)} | a=all c=clear q=save",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    while True:
        display = get_display()
        cv2.imshow(window_name, display)
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('a'):
            selected = [True] * len(unique_masks)
            print("  Selected all")
        elif key == ord('c'):
            selected = [False] * len(unique_masks)
            print("  Cleared all")
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Create final mask
    combined = np.zeros((H, W), dtype=np.uint8)
    obj_id = 1
    for i, (mask, is_selected) in enumerate(zip(unique_masks, selected)):
        if is_selected:
            combined[mask] = obj_id
            obj_id += 1
    
    if np.any(combined > 0):
        save_vos_mask(combined, output_path)
    else:
        print("\nNo objects selected. Mask not saved.")


# ==============================================================================
# Method 3: Draw Bounding Boxes
# ==============================================================================

def draw_boxes_mode(image_path: str, output_path: str, device: str = "cuda"):
    """
    Draw bounding boxes around objects to segment.
    
    Controls:
    - Click and drag: Draw box
    - 'n': Confirm current boxes and segment
    - 'r': Remove last box
    - 'c': Clear all boxes
    - 'q': Quit and save
    """
    import cv2
    import torch
    
    print("\n" + "="*60)
    print("BOUNDING BOX MODE")
    print("="*60)
    print("Controls:")
    print("  Click+drag : Draw box around object")
    print("  'n'        : Segment all boxes")
    print("  'r'        : Remove last box")
    print("  'c'        : Clear all boxes")
    print("  'q'        : Quit and save")
    print("="*60 + "\n")
    
    # Load image first (before SAM3 to show preview faster)
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    H, W = image_cv.shape[:2]
    
    # State
    boxes = []
    drawing = False
    start_point = None
    current_point = None
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    sam3_loaded = False
    processor = None
    state = None
    
    window_name = "Draw Boxes (click and drag)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, W), min(720, H))
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, current_point, boxes
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x1, y1 = start_point
            x2, y2 = x, y
            
            # Normalize coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Minimum box size
            if x2 - x1 > 10 and y2 - y1 > 10:
                boxes.append((x1, y1, x2, y2))
                print(f"  Box {len(boxes)}: ({x1}, {y1}) - ({x2}, {y2})")
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    def get_display():
        display = image_cv.copy()
        
        # Draw mask overlay
        if np.any(combined_mask > 0):
            overlay = np.zeros_like(display)
            for oid in np.unique(combined_mask):
                if oid > 0:
                    color = [(oid*67+100)%256, (oid*137+80)%256, (oid*37+120)%256]
                    overlay[combined_mask == oid] = color
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        
        # Draw existing boxes
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = ((i*67+100)%256, (i*137+80)%256, (i*37+120)%256)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, str(i+1), (x1+5, y1+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw current box being drawn
        if drawing and start_point and current_point:
            cv2.rectangle(display, start_point, current_point, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(display, f"Boxes: {len(boxes)} | n=segment r=remove c=clear q=save",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    while True:
        display = get_display()
        cv2.imshow(window_name, display)
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('n') and boxes:
            # Load SAM3 if not loaded
            if not sam3_loaded:
                print("\nLoading SAM3...")
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                
                model = build_sam3_image_model()
                processor = Sam3Processor(model)
                state = processor.set_image(image_pil)
                sam3_loaded = True
            
            # Segment each box
            print(f"\nSegmenting {len(boxes)} boxes...")
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            
            for i, box in enumerate(boxes):
                box_t = torch.tensor([list(box)], dtype=torch.float32)
                output = processor.set_box_prompt(state=state, box=box_t)
                
                mask = output["masks"][0]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                combined_mask[mask.astype(bool)] = i + 1
                print(f"  ✓ Box {i+1} segmented")
            
            boxes = []  # Clear boxes after segmentation
        
        elif key == ord('r') and boxes:
            removed = boxes.pop()
            print(f"  Removed box: {removed}")
        
        elif key == ord('c'):
            boxes = []
            print("  Cleared all boxes")
        
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if np.any(combined_mask > 0):
        save_vos_mask(combined_mask, output_path)
    else:
        print("\nNo objects segmented. Mask not saved.")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate VOS masks without knowing object names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  interactive  : Click on objects to segment (recommended)
  auto         : Automatically find all objects, then select which to keep
  draw_boxes   : Draw bounding boxes around objects

Examples:
  python generate_mask_interactive.py --image frame.jpg --mode interactive
  python generate_mask_interactive.py --image frame.jpg --mode auto
  python generate_mask_interactive.py --image frame.jpg --mode draw_boxes
        """
    )
    
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default=None, 
                       help="Output mask path (default: <image>_mask.png)")
    parser.add_argument("--mode", "-m", default="interactive",
                       choices=["interactive", "auto", "draw_boxes"],
                       help="Segmentation mode")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    if args.output is None:
        p = Path(args.image)
        args.output = str(p.parent / f"{p.stem}_mask.png")
    
    print("="*60)
    print("SAM3 Mask Generator (No Object Names Required)")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "interactive":
        interactive_click_mode(args.image, args.output, args.device)
    elif args.mode == "auto":
        automatic_mode(args.image, args.output, args.device)
    elif args.mode == "draw_boxes":
        draw_boxes_mode(args.image, args.output, args.device)


if __name__ == "__main__":
    main()