"""
Visualize VOS (Video Object Segmentation) mask trajectories with colored
contour overlays.  Automatically discovers all object mask files in a
directory and assigns distinct colors.

--video accepts either a video FILE or a FOLDER of frame images.
  When a folder is given the frames are sorted naturally (frame_00000.jpg,
  frame_00001.jpg, …) and the output defaults to a sibling folder of
  annotated frames (override with --output).  You can also force video
  output from frame-folder input with --output out.mp4 --out_fps 30.

Mask directory layout (auto-detected):
    masks_dir/
        <video_name>_1.json      e.g. 0000_1.json
        <video_name>_2.json      e.g. 0000_2.json
        ...
  OR
    masks_dir/
        1.json / obj_1.json      any file with a numeric id
        2.json / obj_2.json
        ...

Each JSON must contain:
    { "prediction": { "<frame_idx>": { "<sub_id>": {
          "counts": <COCO_RLE_string>, "size": [H, W] } } } }

Usage examples:
    # Video file input
    python visualize_vos.py --video 0000.mp4 --mask_dir ./masks/

    # Frame folder input → outputs annotated frames
    python visualize_vos.py --video ./frames/ --mask_dir ./masks/

    # Frame folder input → force video output
    python visualize_vos.py --video ./frames/ --mask_dir ./masks/ \
        --output result.mp4 --out_fps 30

    # Explicit mask files
    python visualize_vos.py --video 0000.mp4 --masks 0000_1.json 0000_2.json

    # Custom colors, contour only, with labels
    python visualize_vos.py --video 0000.mp4 --mask_dir ./masks/ \
        --colors "#00FF00" "#FF8000" --alpha 0 --show_labels \
        --label_names toothpaste gripper

    python visualize_vos.py --video /home/yizhou/Mprojects/VAD/TubeletGraph/assets/example/IPAD-1/JPEGImages/IPAD-01 --mask_dir /home/yizhou/Mprojects/VAD/TubeletGraph/_pred_out/custom-IPAD-01-Ours \
    --alpha 0 
"""

import argparse
import glob
import json
import os
import re
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_sort_key(s):
    """Sort key that handles embedded numbers naturally (frame_2 < frame_10)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', str(s))]


# ═══════════════════════════════════════════════════════════════════════════
# FrameSource — unified reader for video files and image folders
# ═══════════════════════════════════════════════════════════════════════════

class FrameSource:
    """
    Transparent frame iterator that wraps either:
      • a video file  (via cv2.VideoCapture), or
      • a directory of frame images (sorted naturally).
    """

    def __init__(self, path: str):
        self.path = path
        self.is_dir = os.path.isdir(path)

        if self.is_dir:
            files = [
                f for f in sorted(os.listdir(path), key=natural_sort_key)
                if Path(f).suffix.lower() in IMAGE_EXTS
            ]
            if not files:
                raise FileNotFoundError(
                    f"No image files found in {path}  "
                    f"(looked for {', '.join(IMAGE_EXTS)})"
                )
            self._frame_paths = [os.path.join(path, f) for f in files]
            first = cv2.imread(self._frame_paths[0])
            self.height, self.width = first.shape[:2]
            self.total = len(self._frame_paths)
            self.fps = 0.0  # unknown for image folder
            self.stem = Path(path.rstrip("/\\")).stem
            self._idx = 0
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps    = self._cap.get(cv2.CAP_PROP_FPS)
            self.stem   = Path(path).stem

    def read(self):
        """Return (success: bool, frame: ndarray|None)."""
        if self.is_dir:
            if self._idx >= self.total:
                return False, None
            frame = cv2.imread(self._frame_paths[self._idx])
            self._idx += 1
            return (frame is not None), frame
        else:
            return self._cap.read()

    def release(self):
        if not self.is_dir:
            self._cap.release()

    @property
    def source_label(self):
        return "frame dir" if self.is_dir else "video file"


# ═══════════════════════════════════════════════════════════════════════════
# COCO compressed-RLE decoder  (no pycocotools dependency)
# Mirrors maskApi.c :: rleFrString exactly.
# ═══════════════════════════════════════════════════════════════════════════

def _rle_str_to_counts(s: str):
    """Decode COCO compressed-RLE string → list of run lengths."""
    counts = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= (~0) << (5 * k)
        if len(counts) > 2:          # delta decode (C: if(m>2))
            x += counts[-2]
        counts.append(x)
    return counts


def rle_decode(rle_dict: dict) -> np.ndarray:
    """
    Decode a COCO compressed-RLE dict {'counts': str, 'size': [h, w]}
    into a binary mask of shape (h, w).
    """
    h, w = rle_dict["size"]
    counts = _rle_str_to_counts(rle_dict["counts"])

    mask_flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if c < 0:
            c = 0
        end = min(pos + c, h * w)
        if i % 2 == 1:               # odd runs = foreground
            mask_flat[pos:end] = 1
        pos = end

    return mask_flat.reshape((h, w), order="F")   # column-major


# ═══════════════════════════════════════════════════════════════════════════
# Color palette – 20 visually distinct colors (RGB)
# ═══════════════════════════════════════════════════════════════════════════

PALETTE_RGB = [
    (0,   255, 0),    # green
    (255, 128, 0),    # orange
    (0,   128, 255),  # sky blue
    (255, 0,   0),    # red
    (255, 255, 0),    # yellow
    (255, 0,   255),  # magenta
    (0,   255, 255),  # cyan
    (128, 0,   255),  # violet
    (0,   255, 128),  # spring green
    (255, 128, 128),  # salmon
    (128, 255, 0),    # chartreuse
    (0,   128, 128),  # teal
    (255, 200, 0),    # gold
    (200, 0,   128),  # raspberry
    (64,  224, 208),  # turquoise
    (255, 105, 180),  # hot pink
    (173, 216, 230),  # light blue
    (144, 238, 144),  # light green
    (218, 112, 214),  # orchid
    (250, 128, 114),  # salmon2
]


def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def parse_color(s: str):
    """
    Parse a color string into an (R, G, B) tuple.
    Accepts:  "#FF8000"  |  "255,128,0"  |  "(255,128,0)"
    """
    s = s.strip()
    # Hex
    if s.startswith("#"):
        s = s.lstrip("#")
        return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
    # Tuple-like
    s = s.strip("() ")
    parts = [x.strip() for x in s.split(",")]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return tuple(int(p) for p in parts)
    raise ValueError(f"Cannot parse color: '{s}'. Use '#RRGGBB' or 'R,G,B'.")


# ═══════════════════════════════════════════════════════════════════════════
# Mask discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_masks(mask_dir: str, video_stem: str = None):
    """
    Auto-discover mask JSON files in *mask_dir* and return them sorted by
    object id.

    Supported naming patterns (all case-insensitive):
        <video_stem>_<id>.json        e.g. 0000_1.json
        <anything>_<id>.json          e.g. obj_1.json, mask_02.json
        <id>.json                     e.g. 1.json, 02.json

    Returns:
        list of (object_id: int, filepath: str) sorted by object_id.
    """
    json_files = sorted(glob.glob(os.path.join(mask_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {mask_dir}")

    results = []
    for fp in json_files:
        name = Path(fp).stem                     # e.g. "0000_2"

        # Try <stem>_<id> pattern first
        if video_stem and name.lower().startswith(video_stem.lower()):
            suffix = name[len(video_stem):]
            m = re.search(r'(\d+)', suffix)
            if m:
                results.append((int(m.group(1)), fp))
                continue

        # Generic: last group of digits in the filename
        m = re.findall(r'(\d+)', name)
        if m:
            results.append((int(m[-1]), fp))
        else:
            # No numeric id found; assign sequentially
            results.append((len(results) + 1, fp))

    results.sort(key=lambda x: x[0])
    return results


def load_predictions(json_path: str) -> dict:
    """Load mask predictions from a VOS JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    if "prediction" in data:
        return data["prediction"]
    # Fallback: treat the entire file as the prediction dict
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict):
        return data
    raise ValueError(
        f"Unrecognized JSON structure in {json_path}. "
        "Expected a 'prediction' key or frame-indexed dict."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_mask_overlay(frame, mask, color_bgr, alpha=0.3):
    """Semi-transparent color fill on the masked region."""
    overlay = frame.copy()
    overlay[mask > 0] = color_bgr
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_contours(frame, mask, color_bgr, thickness=2):
    """Draw anti-aliased contours of the binary mask."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        frame, contours, -1, color_bgr, thickness, lineType=cv2.LINE_AA
    )


def draw_label(frame, mask, text, color_bgr, font_scale=0.7, thickness=2):
    """Draw a text label at the centroid of the mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return
    cx, cy = int(xs.mean()), int(ys.mean())
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    # Background rectangle for readability
    cv2.rectangle(
        frame,
        (cx - tw // 2 - 4, cy - th - 6),
        (cx + tw // 2 + 4, cy + 4),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, text, (cx - tw // 2, cy),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness,
        lineType=cv2.LINE_AA
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Overlay VOS mask contours on video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Input ──────────────────────────────────────────────────────────
    p.add_argument("--video", required=True,
                   help="Input video file OR folder of frame images.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--mask_dir",
                     help="Directory containing per-object mask JSONs. "
                          "Files are auto-discovered and sorted by object id.")
    grp.add_argument("--masks", nargs="+",
                     help="Explicit list of mask JSON files (one per object).")

    # ── Output ─────────────────────────────────────────────────────────
    p.add_argument("--output", default=None,
                   help="Output path.  When input is a video, defaults to "
                        "<stem>_vis.mp4.  When input is a frame folder, "
                        "defaults to <folder>_vis/ (frame images).  "
                        "Set to a .mp4 path to force video output.")
    p.add_argument("--out_fps", type=float, default=30.0,
                   help="FPS for video output when input is a frame folder "
                        "(default: 30). Ignored when input is already a video.")
    p.add_argument("--out_fmt", default="jpg", choices=["jpg", "png", "bmp"],
                   help="Image format when outputting frames (default: jpg).")

    # ── Appearance ─────────────────────────────────────────────────────
    p.add_argument("--colors", nargs="*", default=None,
                   help="Per-object colors as '#RRGGBB' or 'R,G,B'.  "
                        "Auto-assigned from a 20-color palette if omitted.")
    p.add_argument("--thickness", type=int, default=2,
                   help="Contour line thickness (default: 2).")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Mask fill opacity, 0 = contour only (default: 0.3).")
    p.add_argument("--show_labels", action="store_true",
                   help="Draw object id labels at mask centroids.")
    p.add_argument("--label_names", nargs="*", default=None,
                   help="Custom label names per object, e.g. "
                        "--label_names toothpaste gripper")
    return p


def main():
    args = build_parser().parse_args()

    # ── Open frame source (video file or image folder) ─────────────────
    src = FrameSource(args.video)
    video_stem = src.stem

    # ── Discover / load masks ──────────────────────────────────────────
    if args.mask_dir:
        found = discover_masks(args.mask_dir, video_stem)
        obj_ids    = [oid for oid, _ in found]
        mask_paths = [fp  for _, fp  in found]
    else:
        mask_paths = args.masks
        obj_ids    = list(range(1, len(mask_paths) + 1))

    n_objects = len(mask_paths)
    all_preds = [load_predictions(p) for p in mask_paths]

    print(f"Found {n_objects} object(s):")
    for idx, (oid, fp) in enumerate(zip(obj_ids, mask_paths)):
        n_frames = len(all_preds[idx])
        print(f"  Object {oid}: {Path(fp).name}  ({n_frames} frames)")

    # ── Assign colors ──────────────────────────────────────────────────
    if args.colors:
        colors_rgb = [parse_color(c) for c in args.colors]
    else:
        colors_rgb = []

    while len(colors_rgb) < n_objects:
        colors_rgb.append(PALETTE_RGB[len(colors_rgb) % len(PALETTE_RGB)])

    colors_bgr = [rgb_to_bgr(c) for c in colors_rgb]

    # ── Label names ────────────────────────────────────────────────────
    if args.label_names:
        labels = list(args.label_names)
    else:
        labels = [f"obj {oid}" for oid in obj_ids]
    while len(labels) < n_objects:
        labels.append(f"obj {len(labels) + 1}")

    # ── Determine output mode (video file vs frame folder) ─────────────
    W, H, total = src.width, src.height, src.total

    if args.output is None:
        if src.is_dir:
            # default: sibling folder of annotated frames
            args.output = str(Path(args.video.rstrip("/\\")).parent
                              / (video_stem + "_vis"))
            output_is_video = False
        else:
            args.output = str(Path(args.video).with_name(video_stem + "_vis.mp4"))
            output_is_video = True
    else:
        # user-specified: decide by extension
        output_is_video = Path(args.output).suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv", ".wmv"
        }

    if output_is_video:
        fps = src.fps if src.fps > 0 else args.out_fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    else:
        os.makedirs(args.output, exist_ok=True)
        out_writer = None
        out_ext = f".{args.out_fmt}"

    # ── Print summary ──────────────────────────────────────────────────
    fps_str = f"{src.fps:.1f}" if src.fps > 0 else "N/A"
    print(f"\nInput : {args.video}  [{src.source_label}]  "
          f"({W}x{H}, fps={fps_str}, {total} frames)")
    out_kind = "video" if output_is_video else "frames"
    print(f"Output: {args.output}  [{out_kind}]")
    print(f"Style : alpha={args.alpha}, thickness={args.thickness}, "
          f"labels={'ON' if args.show_labels else 'OFF'}")
    color_info = ", ".join(
        f"{labels[i]}=rgb{colors_rgb[i]}" for i in range(n_objects)
    )
    print(f"Colors: {color_info}\n")

    # ── Process frames ─────────────────────────────────────────────────
    frame_idx = 0
    while True:
        ret, frame = src.read()
        if not ret:
            break

        fkey = str(frame_idx)

        for i, preds in enumerate(all_preds):
            if fkey not in preds:
                continue

            # Merge sub-masks for this object on this frame
            combined = np.zeros((H, W), dtype=np.uint8)
            for sub_key, rle in preds[fkey].items():
                combined = np.maximum(combined, rle_decode(rle))

            color = colors_bgr[i]

            if args.alpha > 0:
                draw_mask_overlay(frame, combined, color, args.alpha)

            draw_contours(frame, combined, color, args.thickness)

            if args.show_labels:
                draw_label(frame, combined, labels[i], color)

        if output_is_video:
            out_writer.write(frame)
        else:
            fname = f"frame_{str(frame_idx).zfill(5)}{out_ext}"
            cv2.imwrite(os.path.join(args.output, fname), frame)

        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == total:
            print(f"\rProcessed {frame_idx}/{total} frames", end="", flush=True)

    src.release()
    if out_writer is not None:
        out_writer.release()
    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()


# """
# Visualize VOS (Video Object Segmentation) mask trajectories with colored
# contour overlays.  Automatically discovers all object mask files in a
# directory and assigns distinct colors.

# Mask directory layout (auto-detected):
#     masks_dir/
#         <video_name>_1.json      e.g. 0000_1.json
#         <video_name>_2.json      e.g. 0000_2.json
#         ...
#   OR
#     masks_dir/
#         1.json / obj_1.json      any file with a numeric id
#         2.json / obj_2.json
#         ...

# Each JSON must contain:
#     { "prediction": { "<frame_idx>": { "<sub_id>": {
#           "counts": <COCO_RLE_string>, "size": [H, W] } } } }

# Usage examples:
#     # Auto-discover masks from directory, auto-assign colors
#     python visualize_vos.py --video 0000.mp4 --mask_dir ./masks/

#     # Explicit mask files
#     python visualize_vos.py --video 0000.mp4 --masks 0000_1.json 0000_2.json

#     # Custom colors (hex or RGB tuple), overlay alpha, contour thickness
#     python visualize_vos.py --video 0000.mp4 --mask_dir ./masks/ \
#         --colors "#00FF00" "#FF8000" --alpha 0.35 --thickness 3

#     # Contour only (no fill), with object labels
#     python visualize_vos.py --video 0000.mp4 --mask_dir ./masks/ \
#         --alpha 0 --show_labels

#     # Custom label names
#     python visualize_vos.py --video 0000.mp4 --mask_dir ./masks/ \
#         --show_labels --label_names toothpaste gripper
# """

# import argparse
# import glob
# import json
# import os
# import re
# import cv2
# import numpy as np
# from pathlib import Path


# # ═══════════════════════════════════════════════════════════════════════════
# # COCO compressed-RLE decoder  (no pycocotools dependency)
# # Mirrors maskApi.c :: rleFrString exactly.
# # ═══════════════════════════════════════════════════════════════════════════

# def _rle_str_to_counts(s: str):
#     """Decode COCO compressed-RLE string → list of run lengths."""
#     counts = []
#     p = 0
#     while p < len(s):
#         x = 0
#         k = 0
#         more = True
#         while more:
#             c = ord(s[p]) - 48
#             x |= (c & 0x1F) << (5 * k)
#             more = (c & 0x20) != 0
#             p += 1
#             k += 1
#             if not more and (c & 0x10):
#                 x |= (~0) << (5 * k)
#         if len(counts) > 2:          # delta decode (C: if(m>2))
#             x += counts[-2]
#         counts.append(x)
#     return counts


# def rle_decode(rle_dict: dict) -> np.ndarray:
#     """
#     Decode a COCO compressed-RLE dict {'counts': str, 'size': [h, w]}
#     into a binary mask of shape (h, w).
#     """
#     h, w = rle_dict["size"]
#     counts = _rle_str_to_counts(rle_dict["counts"])

#     mask_flat = np.zeros(h * w, dtype=np.uint8)
#     pos = 0
#     for i, c in enumerate(counts):
#         if c < 0:
#             c = 0
#         end = min(pos + c, h * w)
#         if i % 2 == 1:               # odd runs = foreground
#             mask_flat[pos:end] = 1
#         pos = end

#     return mask_flat.reshape((h, w), order="F")   # column-major


# # ═══════════════════════════════════════════════════════════════════════════
# # Color palette – 20 visually distinct colors (RGB)
# # ═══════════════════════════════════════════════════════════════════════════

# PALETTE_RGB = [
#     (0,   255, 0),    # green
#     (255, 128, 0),    # orange
#     (0,   128, 255),  # sky blue
#     (255, 0,   0),    # red
#     (255, 255, 0),    # yellow
#     (255, 0,   255),  # magenta
#     (0,   255, 255),  # cyan
#     (128, 0,   255),  # violet
#     (0,   255, 128),  # spring green
#     (255, 128, 128),  # salmon
#     (128, 255, 0),    # chartreuse
#     (0,   128, 128),  # teal
#     (255, 200, 0),    # gold
#     (200, 0,   128),  # raspberry
#     (64,  224, 208),  # turquoise
#     (255, 105, 180),  # hot pink
#     (173, 216, 230),  # light blue
#     (144, 238, 144),  # light green
#     (218, 112, 214),  # orchid
#     (250, 128, 114),  # salmon2
# ]


# def rgb_to_bgr(rgb):
#     return (rgb[2], rgb[1], rgb[0])


# def parse_color(s: str):
#     """
#     Parse a color string into an (R, G, B) tuple.
#     Accepts:  "#FF8000"  |  "255,128,0"  |  "(255,128,0)"
#     """
#     s = s.strip()
#     # Hex
#     if s.startswith("#"):
#         s = s.lstrip("#")
#         return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
#     # Tuple-like
#     s = s.strip("() ")
#     parts = [x.strip() for x in s.split(",")]
#     if len(parts) == 3 and all(p.isdigit() for p in parts):
#         return tuple(int(p) for p in parts)
#     raise ValueError(f"Cannot parse color: '{s}'. Use '#RRGGBB' or 'R,G,B'.")


# # ═══════════════════════════════════════════════════════════════════════════
# # Mask discovery
# # ═══════════════════════════════════════════════════════════════════════════

# def discover_masks(mask_dir: str, video_stem: str = None):
#     """
#     Auto-discover mask JSON files in *mask_dir* and return them sorted by
#     object id.

#     Supported naming patterns (all case-insensitive):
#         <video_stem>_<id>.json        e.g. 0000_1.json
#         <anything>_<id>.json          e.g. obj_1.json, mask_02.json
#         <id>.json                     e.g. 1.json, 02.json

#     Returns:
#         list of (object_id: int, filepath: str) sorted by object_id.
#     """
#     json_files = sorted(glob.glob(os.path.join(mask_dir, "*.json")))
#     if not json_files:
#         raise FileNotFoundError(f"No .json files found in {mask_dir}")

#     results = []
#     for fp in json_files:
#         name = Path(fp).stem                     # e.g. "0000_2"

#         # Try <stem>_<id> pattern first
#         if video_stem and name.lower().startswith(video_stem.lower()):
#             suffix = name[len(video_stem):]
#             m = re.search(r'(\d+)', suffix)
#             if m:
#                 results.append((int(m.group(1)), fp))
#                 continue

#         # Generic: last group of digits in the filename
#         m = re.findall(r'(\d+)', name)
#         if m:
#             results.append((int(m[-1]), fp))
#         else:
#             # No numeric id found; assign sequentially
#             results.append((len(results) + 1, fp))

#     results.sort(key=lambda x: x[0])
#     return results


# def load_predictions(json_path: str) -> dict:
#     """Load mask predictions from a VOS JSON file."""
#     with open(json_path) as f:
#         data = json.load(f)
#     if "prediction" in data:
#         return data["prediction"]
#     # Fallback: treat the entire file as the prediction dict
#     first_val = next(iter(data.values()))
#     if isinstance(first_val, dict):
#         return data
#     raise ValueError(
#         f"Unrecognized JSON structure in {json_path}. "
#         "Expected a 'prediction' key or frame-indexed dict."
#     )


# # ═══════════════════════════════════════════════════════════════════════════
# # Drawing helpers
# # ═══════════════════════════════════════════════════════════════════════════

# def draw_mask_overlay(frame, mask, color_bgr, alpha=0.3):
#     """Semi-transparent color fill on the masked region."""
#     overlay = frame.copy()
#     overlay[mask > 0] = color_bgr
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# def draw_contours(frame, mask, color_bgr, thickness=5):
#     """Draw anti-aliased contours of the binary mask."""
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     cv2.drawContours(
#         frame, contours, -1, color_bgr, thickness, lineType=cv2.LINE_AA
#     )


# def draw_label(frame, mask, text, color_bgr, font_scale=3, thickness=4):
#     """Draw a text label at the centroid of the mask."""
#     ys, xs = np.where(mask > 0)
#     if len(xs) == 0:
#         return
#     cx, cy = int(xs.mean()), int(ys.mean())
#     (tw, th), _ = cv2.getTextSize(
#         text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
#     )
#     # Background rectangle for readability
#     cv2.rectangle(
#         frame,
#         (cx - tw // 2 - 4, cy - th - 6),
#         (cx + tw // 2 + 4, cy + 4),
#         (0, 0, 0), -1
#     )
#     cv2.putText(
#         frame, text, (cx - tw // 2, cy),
#         cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness,
#         lineType=cv2.LINE_AA
#     )


# # ═══════════════════════════════════════════════════════════════════════════
# # Main
# # ═══════════════════════════════════════════════════════════════════════════

# def build_parser():
#     p = argparse.ArgumentParser(
#         description="Overlay VOS mask contours on video.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     # ── Input ──────────────────────────────────────────────────────────
#     p.add_argument("--video", required=True,
#                    help="Input video path.")
#     grp = p.add_mutually_exclusive_group(required=True)
#     grp.add_argument("--mask_dir",
#                      help="Directory containing per-object mask JSONs. "
#                           "Files are auto-discovered and sorted by object id.")
#     grp.add_argument("--masks", nargs="+",
#                      help="Explicit list of mask JSON files (one per object).")

#     # ── Output ─────────────────────────────────────────────────────────
#     p.add_argument("--output", default=None,
#                    help="Output video path.  Default: <video_stem>_vis.mp4")

#     # ── Appearance ─────────────────────────────────────────────────────
#     p.add_argument("--colors", nargs="*", default=None,
#                    help="Per-object colors as '#RRGGBB' or 'R,G,B'.  "
#                         "Auto-assigned from a 20-color palette if omitted.")
#     p.add_argument("--thickness", type=int, default=15,
#                    help="Contour line thickness (default: 2).")
#     p.add_argument("--alpha", type=float, default=0,
#                    help="Mask fill opacity, 0 = contour only (default: 0.3).")
#     p.add_argument("--show_labels", action="store_true",
#                    help="Draw object id labels at mask centroids.")
#     p.add_argument("--label_names", nargs="*", default=None,
#                    help="Custom label names per object, e.g. "
#                         "--label_names toothpaste gripper")
#     return p


# def main():
#     args = build_parser().parse_args()

#     video_stem = Path(args.video).stem          # e.g. "0000"

#     # ── Discover / load masks ──────────────────────────────────────────
#     if args.mask_dir:
#         found = discover_masks(args.mask_dir, video_stem)
#         obj_ids    = [oid for oid, _ in found]
#         mask_paths = [fp  for _, fp  in found]
#     else:
#         mask_paths = args.masks
#         obj_ids    = list(range(1, len(mask_paths) + 1))

#     n_objects = len(mask_paths)
#     all_preds = [load_predictions(p) for p in mask_paths]

#     print(f"Found {n_objects} object(s):")
#     for idx, (oid, fp) in enumerate(zip(obj_ids, mask_paths)):
#         n_frames = len(all_preds[idx])
#         print(f"  Object {oid}: {Path(fp).name}  ({n_frames} frames)")

#     # ── Assign colors ──────────────────────────────────────────────────
#     if args.colors:
#         colors_rgb = [parse_color(c) for c in args.colors]
#     else:
#         colors_rgb = []

#     while len(colors_rgb) < n_objects:
#         colors_rgb.append(PALETTE_RGB[len(colors_rgb) % len(PALETTE_RGB)])

#     colors_bgr = [rgb_to_bgr(c) for c in colors_rgb]

#     # ── Label names ────────────────────────────────────────────────────
#     if args.label_names:
#         labels = list(args.label_names)
#     else:
#         labels = [f"obj {oid}" for oid in obj_ids]
#     while len(labels) < n_objects:
#         labels.append(f"obj {len(labels) + 1}")

#     # ── Open video ─────────────────────────────────────────────────────
#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {args.video}")

#     fps   = cap.get(cv2.CAP_PROP_FPS)
#     W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # ── Output path ────────────────────────────────────────────────────
#     if args.output is None:
#         args.output = str(Path(args.video).with_name(video_stem + "_vis.mp4"))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

#     # ── Print summary ──────────────────────────────────────────────────
#     print(f"\nVideo : {args.video}  ({W}x{H}, {fps:.1f} fps, {total} frames)")
#     print(f"Output: {args.output}")
#     print(f"Style : alpha={args.alpha}, thickness={args.thickness}, "
#           f"labels={'ON' if args.show_labels else 'OFF'}")
#     color_info = ", ".join(
#         f"{labels[i]}=rgb{colors_rgb[i]}" for i in range(n_objects)
#     )
#     print(f"Colors: {color_info}\n")

#     # ── Process frames ─────────────────────────────────────────────────
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         fkey = str(frame_idx)

#         for i, preds in enumerate(all_preds):
#             if fkey not in preds:
#                 continue

#             # Merge sub-masks for this object on this frame
#             combined = np.zeros((H, W), dtype=np.uint8)
#             for sub_key, rle in preds[fkey].items():
#                 combined = np.maximum(combined, rle_decode(rle))

#             color = colors_bgr[i]

#             if args.alpha > 0:
#                 draw_mask_overlay(frame, combined, color, args.alpha)

#             draw_contours(frame, combined, color, args.thickness)

#             if args.show_labels:
#                 draw_label(frame, combined, labels[i], color)

#         out.write(frame)
#         frame_idx += 1

#         if frame_idx % 30 == 0 or frame_idx == total:
#             print(f"\rProcessed {frame_idx}/{total} frames", end="", flush=True)

#     cap.release()
#     out.release()
#     print(f"\nDone → {args.output}")


# if __name__ == "__main__":
#     main()





# # """
# # Visualize VOS (Video Object Segmentation) mask trajectories for 2 objects
# # with colored contours overlaid on the original video.

# # Usage:
# #     python visualize_vos.py \
# #         --video 0000.mp4 \
# #         --masks 0000_1.json 0000_2.json \
# #         --output output_vis.mp4 \
# #         --colors "(0,255,0)" "(255,0,0)" \
# #         --thickness 2 \
# #         --alpha 0.3
# # """

# # import argparse
# # import json
# # import cv2
# # import numpy as np


# # # ---------------------------------------------------------------------------
# # # Custom COCO compressed-RLE decoder (no pycocotools dependency)
# # # ---------------------------------------------------------------------------
# # def _rle_str_to_counts(s: str):
# #     """
# #     Decode COCO's compressed RLE string into a list of run lengths.
# #     Mirrors the C implementation in maskApi.c :: rleFrString exactly.
# #     """
# #     counts = []
# #     p = 0
# #     while p < len(s):
# #         x = 0
# #         k = 0
# #         more = True
# #         while more:
# #             c = ord(s[p]) - 48
# #             x |= (c & 0x1F) << (5 * k)
# #             more = (c & 0x20) != 0
# #             p += 1
# #             k += 1
# #             if not more and (c & 0x10):
# #                 x |= (~0) << (5 * k)
# #         # Delta decoding: each count adds the value two positions back
# #         # C code: if(m>2) x += cnts[m-2], where m is 0-indexed count index
# #         if len(counts) > 2:
# #             x += counts[-2]
# #         counts.append(x)

# #     return counts


# # def rle_decode(rle_dict: dict) -> np.ndarray:
# #     """
# #     Decode a COCO compressed-RLE dict {'counts': str, 'size': [h, w]}
# #     into a binary mask of shape (h, w).
# #     """
# #     h, w = rle_dict["size"]
# #     counts = _rle_str_to_counts(rle_dict["counts"])

# #     # Build flat mask – runs alternate 0, 1, 0, 1, ...
# #     mask_flat = np.zeros(h * w, dtype=np.uint8)
# #     pos = 0
# #     for i, c in enumerate(counts):
# #         if c < 0:
# #             c = 0
# #         end = min(pos + c, h * w)
# #         if i % 2 == 1:  # odd runs are foreground
# #             mask_flat[pos:end] = 1
# #         pos = end

# #     # COCO stores masks in column-major (Fortran) order
# #     mask = mask_flat.reshape((h, w), order="F")
# #     return mask


# # # ---------------------------------------------------------------------------
# # # Drawing helpers
# # # ---------------------------------------------------------------------------
# # def draw_mask_overlay(frame, mask, color, alpha=0.3):
# #     """Draw a semi-transparent color fill on the masked region."""
# #     overlay = frame.copy()
# #     overlay[mask > 0] = color
# #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# # def draw_contours(frame, mask, color, thickness=2):
# #     """Draw contours of the binary mask on the frame."""
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     cv2.drawContours(frame, contours, -1, color, thickness, lineType=cv2.LINE_AA)


# # # ---------------------------------------------------------------------------
# # # Main
# # # ---------------------------------------------------------------------------
# # def load_predictions(json_path: str) -> dict:
# #     with open(json_path) as f:
# #         data = json.load(f)
# #     return data["prediction"]


# # def parse_color(s: str):
# #     """Parse a color string like '(0,255,0)' into a BGR tuple for OpenCV."""
# #     s = s.strip("() ")
# #     r, g, b = [int(x.strip()) for x in s.split(",")]
# #     return (b, g, r)  # OpenCV uses BGR


# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Overlay VOS mask contours on video."
# #     )
# #     parser.add_argument("--video", required=True, help="Input video path")
# #     parser.add_argument(
# #         "--masks", nargs="+", required=True,
# #         help="Paths to mask JSON files (one per object)"
# #     )
# #     parser.add_argument("--output", default="output_vis.mp4", help="Output video path")
# #     parser.add_argument(
# #         "--colors", nargs="+", default=["(0,255,0)", "(0,128,255)"],
# #         help="Per-object colors as '(R,G,B)' strings"
# #     )
# #     parser.add_argument("--thickness", type=int, default=2, help="Contour line thickness")
# #     parser.add_argument("--alpha", type=float, default=0.3, help="Mask overlay opacity (0=off)")
# #     args = parser.parse_args()

# #     # Load mask predictions
# #     all_preds = [load_predictions(p) for p in args.masks]
# #     colors = [parse_color(c) for c in args.colors]

# #     # Extend colors if fewer than objects
# #     default_palette = [
# #         (0, 255, 0), (0, 128, 255), (255, 0, 0),
# #         (255, 255, 0), (255, 0, 255), (0, 255, 255),
# #     ]
# #     while len(colors) < len(all_preds):
# #         colors.append(default_palette[len(colors) % len(default_palette)])

# #     # Open video
# #     cap = cv2.VideoCapture(args.video)
# #     if not cap.isOpened():
# #         raise RuntimeError(f"Cannot open video: {args.video}")

# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# #     out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

# #     print(f"Video: {w}x{h}, {fps:.1f} fps, {total} frames")
# #     print(f"Objects: {len(all_preds)}, colors (BGR): {colors}")

# #     frame_idx = 0
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         fkey = str(frame_idx)

# #         for obj_idx, preds in enumerate(all_preds):
# #             if fkey not in preds:
# #                 continue
# #             frame_masks = preds[fkey]
# #             # Each frame may have multiple sub-masks (e.g., key "0");
# #             # merge them into one binary mask for this object.
# #             combined = np.zeros((h, w), dtype=np.uint8)
# #             for sub_key, rle in frame_masks.items():
# #                 m = rle_decode(rle)
# #                 combined = np.maximum(combined, m)

# #             color = colors[obj_idx]

# #             # Semi-transparent fill
# #             if args.alpha > 0:
# #                 draw_mask_overlay(frame, combined, color, args.alpha)

# #             # Contour outline
# #             draw_contours(frame, combined, color, args.thickness)

# #         out.write(frame)
# #         frame_idx += 1

# #         if frame_idx % 30 == 0 or frame_idx == total:
# #             print(f"\rProcessed {frame_idx}/{total} frames", end="", flush=True)

# #     print("\nDone!")
# #     cap.release()
# #     out.release()


# # if __name__ == "__main__":
# #     main()
