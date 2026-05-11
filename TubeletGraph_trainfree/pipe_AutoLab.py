#!/usr/bin/env python3
"""
ST-VAD Framework: Spatio-Temporal Video Anomaly Detection  — AutoLab edition
Three-stage pipeline:

  Stage 1 – Object Grounding   : VLM detects objects; shared SAM3 segments them.
             After object detection, a second VLM call produces:
               • A temporal caption describing object interactions / motion across frames.
               • A recommended FPS (int, 2–10) for Stage 2 graph construction.
  Stage 2 – Object Tracking    : quick_run.py / TubeletGraph builds tubelets + state graph.
             Uses the VLM-recommended FPS (falling back to --fps if unavailable).
  Stage 3 – Anomaly Detection  : prompt_vad.py reasons over state graph with VLM.
             Receives the Stage 1 caption via --caption so the VLM can cross-reference
             observed object motion with the graph-derived state changes.

Dataset: AutoLab
  VQA lines example:
    {"video_path": "train/abnormal_16/000.mp4", ...,
     "context": [63, 208]}    ← [start_frame, end_frame] (int list, not a string)

  Input --video arg points to a root directory, e.g.:
    /work/nvme/bgiv/qilong/datasets/AutoLab/test/abnormal_1
  containing .mp4 files.

NVMe permanent storage anchored at "AutoLab":
  Stage 1 → /work/nvme/bgiv/qilong/stage1/AutoLab/<split>/<subset>/<stem>_mask/0000000.png
  Stage 2 → /work/nvme/bgiv/qilong/stage2/AutoLab/<split>/<subset>/<stem>/
  Stage 3 → /work/nvme/bgiv/qilong/stage3/AutoLab/<split>/<subset>/<stem>_report.json

VQA video_path keys are relative (e.g. "train/abnormal_16/000.mp4").
The framework resolves them against the dataset root so lookups work whether
the caller passes an absolute or relative path.

Usage:
    python pipe_autolab.py analyze <video_dir> -c <config> [options]
"""

import os
import os.path as osp
import sys
import re
import base64
import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

from tqdm import trange, tqdm

# ---------------------------------------------------------------------------
# Global Storage Roots
# ---------------------------------------------------------------------------
STAGE1_ROOT = Path("/work/nvme/bgiv/qilong/stage1")
STAGE2_ROOT = Path("/work/nvme/bgiv/qilong/stage2")
STAGE3_ROOT = Path("/work/nvme/bgiv/qilong/stage3")

# Dataset anchor – used for NVMe path construction
DATASET_ANCHOR = "AutoLab"

# Max frames sent to the VLM for caption + FPS selection
CAPTION_MAX_FRAMES = 16

# ---------------------------------------------------------------------------
# Optional OpenAI import
# ---------------------------------------------------------------------------
try:
    import openai as _openai_mod
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def suppress_output(verbose: bool):
    """Silences stdout and stderr at the OS level when verbose is False."""
    if verbose:
        yield
        return
    import warnings, logging
    sys.stdout.flush(); sys.stderr.flush()
    orig_out = sys.stdout.fileno(); orig_err = sys.stderr.fileno()
    saved_out = os.dup(orig_out);  saved_err = os.dup(orig_err)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            with open(os.devnull, 'w') as fnull:
                os.dup2(fnull.fileno(), orig_out)
                os.dup2(fnull.fileno(), orig_err)
                yield
        finally:
            os.dup2(saved_out, orig_out); os.dup2(saved_err, orig_err)
            os.close(saved_out);          os.close(saved_err)
            logging.disable(logging.NOTSET)


def _unique_stem(video_path: str, dataset_root: Optional[str] = None) -> str:
    """
    Globally-unique local scratch identifier for one AutoLab video.

    Strategy (in order):
      1. If `dataset_root` is given, make the path relative to it and join
         all parts with underscores, e.g.:
           root=/…/AutoLab/test, video=/…/AutoLab/test/abnormal_1/000.mp4
           → "abnormal_1_000"
      2. Anchor at the "AutoLab" directory component.
      3. Fall back to the bare file stem.
    """
    p = Path(video_path).resolve()

    if dataset_root:
        root = Path(dataset_root).resolve()
        try:
            rel   = p.relative_to(root)
            parts = list(rel.parent.parts) + [rel.stem]
            return "_".join(parts)
        except ValueError:
            pass

    parts = p.parts
    if DATASET_ANCHOR in parts:
        idx       = parts.index(DATASET_ANCHOR)
        rel_parts = parts[idx + 1:]
        if rel_parts:
            dirs = rel_parts[:-1]
            stem = Path(rel_parts[-1]).stem
            return "_".join(list(dirs) + [stem])

    return p.stem


def _nvme_rel(video_path: str, vqa_metadata: Dict,
              dataset_root: Optional[str] = None) -> Tuple[Path, str]:
    """
    Returns (parent_dir, stem) for NVMe paths, always RELATIVE.

    AutoLab VQA video_path is relative (e.g. "train/abnormal_16/000.mp4").
    NVMe paths are always:  AutoLab/<split>/<subset>/<stem>
    """
    def _build_from_vqa_path(vqa_vp: str) -> Tuple[Path, str]:
        p     = Path(vqa_vp)
        parts = p.parts
        if parts and parts[0] == DATASET_ANCHOR:
            p = Path(*parts[1:])
        return Path(DATASET_ANCHOR) / p.parent, p.stem

    for key in (str(video_path), str(Path(video_path).resolve())):
        entry = vqa_metadata.get(key)
        if entry:
            return _build_from_vqa_path(entry["video_path"])

    parts = Path(video_path).resolve().parts
    if DATASET_ANCHOR in parts:
        idx       = parts.index(DATASET_ANCHOR)
        rel_parts = (DATASET_ANCHOR,) + parts[idx + 1:]
        p         = Path(*rel_parts)
        return p.parent, p.stem

    return Path(Path(video_path).parent.name), Path(video_path).stem


def _mask_nvme_dir(video_path: str, vqa_metadata: Dict,
                   dataset_root: Optional[str] = None) -> Path:
    parent, stem = _nvme_rel(video_path, vqa_metadata, dataset_root)
    return STAGE1_ROOT / parent / f"{stem}_mask"


def _pred_nvme_dir(video_path: str, vqa_metadata: Dict,
                   dataset_root: Optional[str] = None) -> Path:
    parent, stem = _nvme_rel(video_path, vqa_metadata, dataset_root)
    return STAGE2_ROOT / parent / stem


def _report_nvme_path(video_path: str, vqa_metadata: Dict,
                      dataset_root: Optional[str] = None) -> Path:
    parent, stem = _nvme_rel(video_path, vqa_metadata, dataset_root)
    return STAGE3_ROOT / parent / f"{stem}_report.json"


# ---------------------------------------------------------------------------
# VQA metadata helpers
# ---------------------------------------------------------------------------

def _build_vqa_lookup(vqa_metadata: Dict, dataset_root: Optional[str]) -> Dict:
    """
    Augment vqa_metadata with absolute-path keys so lookups succeed regardless
    of whether callers use relative or absolute video paths.
    """
    if not dataset_root:
        return vqa_metadata
    root  = Path(dataset_root).resolve()
    extra: Dict = {}
    for k, v in list(vqa_metadata.items()):
        vp = v.get("video_path", "")
        if not vp:
            continue
        abs_path = str((root / vp).resolve())
        if abs_path not in vqa_metadata:
            extra[abs_path] = v
    vqa_metadata.update(extra)
    return vqa_metadata


import os
import re
import base64
import torch
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict, Any, Tuple

# Fallback definitions to ensure the code is ready-to-run
try:
    import openai as _openai_mod
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

CAPTION_MAX_FRAMES = 8  # Adjust this default if needed


# ---------------------------------------------------------------------------
# OpenAI Responses API client
# ---------------------------------------------------------------------------

class _OpenAIResponsesClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        if not HAS_OPENAI:
            raise ImportError(
                "'openai' package not installed — cannot run caption generation. "
                "Install it with: pip install openai"
            )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model   = model
        self.client  = _openai_mod.OpenAI(api_key=self.api_key)

    def _encode_image(self, image_path: str) -> Dict:
        ext        = Path(image_path).suffix.lower().lstrip(".")
        media_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
        }.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type":      "input_image",
            "image_url": f"data:{media_type};base64,{data}",
        }

    def query(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        content: List[Dict] = []
        if image_paths:
            for p in image_paths:
                content.append(self._encode_image(p))
        content.append({"type": "input_text", "text": prompt})

        kwargs: Dict[str, Any] = {
            "model":             self.model,
            "input":             [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
            "temperature":       temperature,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt

        response = self.client.responses.create(**kwargs)
        return response.output_text


# ---------------------------------------------------------------------------
# Caption + FPS generation
# ---------------------------------------------------------------------------

def _select_caption_frames(frames_dir: str, max_frames: int = CAPTION_MAX_FRAMES) -> List[str]:
    exts       = {".jpg", ".jpeg", ".png"}
    all_frames = sorted(
        str(f) for f in Path(frames_dir).iterdir() if f.suffix.lower() in exts
    )
    if not all_frames:
        return []
    if len(all_frames) <= max_frames:
        return all_frames
    step = (len(all_frames) - 1) / (max_frames - 1)
    return [all_frames[round(i * step)] for i in range(max_frames)]


def generate_caption_and_fps(
    frames_dir: str,
    objects: List[str],
    vlm_model: str = "openai",
    verbose: bool = False,
    hint_str: str = "",
) -> Tuple[str, Optional[int]]:
    """
    Send sampled frames to the OpenAI Responses API and ask for:
      (a) A temporal caption describing object interactions and motion.
      (b) A recommended FPS integer (2–10) for Stage 2 TubeletGraph.

    When hint_str is non-empty the secret-mission prompt variant is used,
    steering the VLM toward caption content consistent with the known anomaly.

    Returns (caption, dynamic_fps).  On any failure returns ("", None).
    """
    if vlm_model != "openai":
        if verbose:
            print(f"  [Caption] Skipped — vlm_model='{vlm_model}' "
                  f"(only 'openai' supported for caption+FPS generation).")
        return "", None

    frame_paths = _select_caption_frames(frames_dir, max_frames=CAPTION_MAX_FRAMES)
    if not frame_paths:
        if verbose:
            print("  [Caption] No frames found in frames_dir — skipping.")
        return "", None

    objects_str = ", ".join(str(o) for o in objects) if objects else "the objects in the scene"

    system_prompt = (
        "You are an expert video analyst specialising in physical anomaly detection "
        "in lab automation procedures. "
        "You receive a sequence of frames sampled uniformly from a video and a list "
        "of detected objects. Your job is to describe the video content with temporal "
        "precision and to recommend an appropriate processing frame rate."
    )

    user_prompt = (
        f"I am providing {len(frame_paths)} evenly-spaced frames from a lab automation video.\n"
        f"The following objects have been detected in the scene: {objects_str}.\n\n"
        "Please provide TWO things:\n\n"
        "1. A detailed TEMPORAL CAPTION describing what happens across these frames.\n"
        "   Requirements:\n"
        "   - Use description tone (firstly, then, after that, ...) and do NOT use detailed frame id.\n"
        "   - Focus on the detected objects and how they interact or change state over time.\n"
        "   - For EACH object, explicitly state at every temporal stage whether it is\n"
        "     MOVING or STATIC, e.g. 'Object A continues to move forward, but Object B\n"
        "     stops moving and remains still from this point onward.'\n"
        "   - Describe any deformation, contact, separation, acceleration, or other\n"
        "     physical state changes in chronological order.\n"
        "   - Note any unusual or potentially anomalous behaviours observed.\n\n"
        "2. A recommended FPS (frames per second) integer for video graph construction. Default 3.\n"
        "   - Range: integer between 2 and 10 (inclusive).\n"
        "   - Use a HIGHER value (e.g. 5-10) for fast-moving or highly dynamic scenes.\n"
        "   - Use a LOWER value (e.g. 2–5) for slow, static, or gradually changing scenes.\n"
        "   - Choose the value that best captures the temporal density of meaningful\n"
        "     state changes without redundant frames.\n\n"
        "Respond using EXACTLY this format (no text outside the tags):\n\n"
        "<CAPTION>\n"
        "[Your detailed temporal caption here]\n"
        "</CAPTION>\n"
        "<FPS>\n"
        "[Integer between 2 and 10]\n"
        "</FPS>"
    )

    try:
        client = _OpenAIResponsesClient()
        if verbose:
            print(f"  [Caption] Querying {client.model} with {len(frame_paths)} "
                  f"frame(s) for caption + FPS …")
        raw_response = client.query(
            prompt=user_prompt,
            image_paths=frame_paths,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception as e:
        if verbose:
            print(f"  [Caption] OpenAI call failed: {e}")
        return "", None

    caption_match = re.search(r"<CAPTION>(.*?)</CAPTION>", raw_response, re.DOTALL)
    caption       = caption_match.group(1).strip() if caption_match else ""

    fps_match = re.search(r"<FPS>\s*(\d+)\s*</FPS>", raw_response)
    if fps_match:
        fps_raw     = int(fps_match.group(1))
        dynamic_fps: Optional[int] = max(2, min(10, fps_raw))
    else:
        dynamic_fps = None

    if verbose:
        fps_display = str(dynamic_fps) if dynamic_fps is not None else "N/A (will use --fps)"
        print(f"  [Caption] Recommended FPS: {fps_display}")
        if caption:
            preview = caption[:140].replace("\n", " ")
            print(f"  [Caption] '{preview}{'…' if len(caption) > 140 else ''}'")
        else:
            print("  [Caption] WARNING: <CAPTION> tag missing in VLM response.")
            print(f"  [Caption] Raw response (first 400 chars):\n{raw_response[:400]}")

    return caption, dynamic_fps


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

class STVADFramework:
    def __init__(self, config_path: str, input_root: str,
                 output_dir: str = "output", verbose: bool = False,
                 dataset_root: Optional[str] = None):
        self.config_path  = config_path
        self.input_root   = input_root
        self.output_dir   = output_dir
        self.verbose      = verbose
        self.dataset_root = dataset_root
        self.base_dir     = osp.dirname(osp.abspath(__file__))

        self.vlm_mask_script   = osp.join(self.base_dir, "annotate", "vlm_mask_grounded.py")
        self.quick_run_script  = osp.join(self.base_dir, "quick_run.py")
        self.prompt_vad_script = osp.join(self.base_dir, "TubeletGraph", "vlm", "prompt_vad.py")

        self.project_root    = Path(self.quick_run_script).parent
        self._sam3_segmenter = None
        self._stage1_mod: Optional[object] = None

        self._verify_scripts()
        STAGE1_ROOT.mkdir(parents=True, exist_ok=True)
        STAGE2_ROOT.mkdir(parents=True, exist_ok=True)
        STAGE3_ROOT.mkdir(parents=True, exist_ok=True)

    def _verify_scripts(self):
        required = {
            "annotate/vlm_mask_grounded.py":  self.vlm_mask_script,
            "quick_run.py":                    self.quick_run_script,
            "TubeletGraph/vlm/prompt_vad.py": self.prompt_vad_script,
        }
        missing = [f"{n}  (expected: {p})" for n, p in required.items() if not osp.isfile(p)]
        if missing:
            print("Missing required scripts:")
            for m in missing: print(f"  {m}")
            sys.exit(1)

    def _run(self, cmd: List[str], description: str, check: bool = True,
             env: Optional[Dict] = None, cwd: Optional[str] = None) -> bool:
        if self.verbose:
            print(f"[CMD] {' '.join(cmd)}")
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
        merged_env = {**os.environ, **(env or {})}
        result = subprocess.run(cmd, env=merged_env, check=False,
                                stdout=stdout, stderr=stderr, cwd=cwd)
        if result.returncode != 0:
            if self.verbose:
                print(f"[FAIL] {description} (exit {result.returncode})")
            if check:
                sys.exit(1)
            return False
        return True

    def _load_stage1_module(self):
        if self._stage1_mod is not None:
            return self._stage1_mod
        annotate_dir = osp.dirname(self.vlm_mask_script)
        if annotate_dir not in sys.path:
            sys.path.insert(0, annotate_dir)
        spec = importlib.util.spec_from_file_location("vlm_mask_grounded", self.vlm_mask_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._stage1_mod = mod
        return mod

    def _get_segmenter(self):
        if self._sam3_segmenter is None:
            mod = self._load_stage1_module()
            self._sam3_segmenter = mod.SAM3Segmenter()
            self._sam3_segmenter._ensure_loaded()
        return self._sam3_segmenter

    def preload_sam3(self):
        if self.verbose:
            print("Pre-loading SAM3 model (once for all videos)...")
        with suppress_output(self.verbose):
            self._get_segmenter()
        if self.verbose:
            print("SAM3 ready.\n")

    # ------------------------------------------------------------------
    # Stage 1 – Object Grounding & Segmentation  +  Caption/FPS generation
    # ------------------------------------------------------------------

    def stage1_object_grounding(
        self,
        video_path: str,
        vqa_metadata: Dict,
        vlm_model: str = "openai",
        auto_mode: bool = True,
        scan_frames: bool = True,
        threshold: float = 0.1,
        num_scan_frames: int = 5,
        fps: Optional[int] = None,
    ) -> Tuple[str, str, bool, List[int], str, Optional[int], int]:
        """
        Returns:
          (local_frames_dir, nvme_mask_path, success, keep_indices,
           caption, dynamic_fps, mask_frame_id)
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 1 – Object Grounding: {Path(video_path).name}")

        mod       = self._load_stage1_module()
        segmenter = self._get_segmenter()

        stem             = _unique_stem(video_path, self.dataset_root)
        _tmp_parent      = Path(tempfile.mkdtemp(prefix="stvad_s1_"))
        local_frames_dir = _tmp_parent / stem
        local_frames_dir.mkdir(parents=True, exist_ok=True)

        nvme_mask_dir = _mask_nvme_dir(video_path, vqa_metadata, self.dataset_root)
        tmp_extract   = Path(tempfile.mkdtemp(prefix=f"stvad_extract_{stem}_"))
        is_dir        = Path(video_path).is_dir()
        keep_indices: List[int] = []

        caption:     str           = ""
        dynamic_fps: Optional[int] = None
        best_frame_idx: int        = 0  # Initialize here to ensure it exists for all return paths

        try:
            import numpy as np

            local_frames_dir.mkdir(parents=True, exist_ok=True)

            if is_dir:
                frame_files = sorted([
                    f for f in Path(video_path).iterdir()
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                ])
                if not frame_files:
                    return str(local_frames_dir), "", False, [], "", None, best_frame_idx

                keep_indices = list(range(len(frame_files)))
                for new_idx, old_idx in enumerate(keep_indices):
                    old_f = frame_files[old_idx]
                    new_f = local_frames_dir / f"{new_idx:07d}{old_f.suffix}"
                    try:
                        os.link(old_f, new_f)
                    except OSError:
                        shutil.copy2(old_f, new_f)

            else:
                input_proc = mod.InputProcessor(video_path, str(tmp_extract), frame_index=0)
                _, extracted_frames_dir = input_proc.process()

                frame_files = sorted([
                    f for f in Path(extracted_frames_dir).iterdir()
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                ])

                keep_indices = list(range(len(frame_files)))
                for new_idx, old_idx in enumerate(keep_indices):
                    old_f = frame_files[old_idx]
                    new_f = local_frames_dir / f"{new_idx:07d}{old_f.suffix}"
                    shutil.copy2(str(old_f), str(new_f))

            input_proc_thinned = mod.InputProcessor(str(local_frames_dir), None, frame_index=0)
            first_frame_path, _ = input_proc_thinned.process()

            detector = mod.VLMObjectDetector(vlm_model)
            # raw_objects  = detector.detect_objects(first_frame_path)
            # objects = ["multichannel pipette"]
            objects = ["single-channel pipette", "liquid"]
            # targets = ["multichannel", "multi-channel", "pipette", "pipettor", "tips", "liquid"]
            # for obj in raw_objects:
            #     for t in targets:
            #         if t in obj:
            #             objects.append(obj)
            # if not objects:
            #     return str(local_frames_dir), "", False, [], "", None, best_frame_idx

            caption, dynamic_fps = generate_caption_and_fps(
                frames_dir=str(local_frames_dir),
                objects=["multichannel pipette"],            # hard code for autolab !!!!!!!
                vlm_model=vlm_model,
                verbose=self.verbose,
            )

            crop_bboxes = {}

            if scan_frames and input_proc_thinned.num_frames > 1:
                best_frame_idx, mask, metadata = mod.scan_frames_for_objects(
                    input_processor=input_proc_thinned, 
                    object_prompts=objects, 
                    segmenter=segmenter, 
                    detector=detector,
                    num_frames_to_scan=num_scan_frames, 
                    threshold=threshold,
                    max_instances=8,
                    bbox_padding=0.1,
                    bbox_object_desc="multi-channel pipette"            # hard code for autolab !!!!!!!
                )
                winning_bbox = metadata.get("crop_bbox")
                if winning_bbox:
                    crop_bboxes[objects[0]] = winning_bbox
            else:
                bbox = detector.locate_group_bbox(
                    first_frame_path,
                    object_desc="multi-channel pipette",                # hard code for autolab !!!!!!!
                    padding=0.1,
                )
                if bbox:
                    crop_bboxes[objects[0]] = bbox
                mask, metadata = segmenter.segment(
                    first_frame_path, 
                    objects, 
                    threshold=threshold,
                    retry_with_variations=True, 
                    min_threshold=0.1,
                    crop_bbox=bbox,
                    max_instances=8
                )

            if not np.any(mask > 0):
                return str(local_frames_dir), "", False, [], caption, dynamic_fps, best_frame_idx
            
            mask_filename = f"{best_frame_idx:07d}.png"
            mask_path = nvme_mask_dir / mask_filename
            mod.save_vos_mask(mask, str(mask_path))
            
            vis_path = nvme_mask_dir / f"{best_frame_idx:07d}_vis.png"
            if vis_path.exists():
                vis_path.unlink(missing_ok=True)

            if crop_bboxes:
                try:
                    from PIL import Image, ImageDraw
                    actual_frame_path = input_proc_thinned.get_frame_path(best_frame_idx)
                    
                    vis_img = Image.open(actual_frame_path).convert("RGB")
                    draw = ImageDraw.Draw(vis_img)
                    
                    for obj_name, b in crop_bboxes.items():
                        if b:
                            x1, y1, x2, y2 = b
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            draw.text((x1, max(0, y1 - 15)), obj_name, fill="red")
                            
                    bbox_vis_path = nvme_mask_dir / "bbox_visualize.png"
                    vis_img.save(str(bbox_vis_path))
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to save bbox visualization: {e}")

            obj_path = nvme_mask_dir / "obj.json"
            with open(obj_path, "w") as f:
                json.dump(objects, f, indent=4)

            return str(local_frames_dir), str(mask_path), True, keep_indices, caption, dynamic_fps, best_frame_idx

        except Exception:
            if self.verbose:
                traceback.print_exc()
            return str(local_frames_dir), "", False, [], caption, dynamic_fps, best_frame_idx
        finally:
            shutil.rmtree(tmp_extract, ignore_errors=True)

    # ------------------------------------------------------------------
    # Stage 2 – TubeletGraph Tracking
    # ------------------------------------------------------------------

    def stage2_tracking(
        self,
        video_path: str,
        frames_dir: str,
        mask_path: str,
        fps: Optional[int],
        mask_frame_id: int = 0,    # NEW ARGUMENT
        method: str = "Ours",
    ) -> Tuple[str, str, str, str, bool]:
        if self.verbose:
            fps_display = str(fps) if fps is not None else "default"
            print(f"\n{'='*60}\nSTAGE 2 – Tracking: {Path(video_path).name}  "
                  f"(fps={fps_display}, mask_frame_id={mask_frame_id})")

        pred_name       = Path(frames_dir).name
        jpeg_images_dir = str(self.project_root / "_custom_dataset" / "JPEGImages" / pred_name)
        pred_out_dir    = str(self.project_root / "_pred_out" / pred_name)
        src_pred        = self.project_root / "_pred_out" / pred_name

        cmd = [
            "python3", self.quick_run_script,
            "-c", self.config_path,
            "--input_dir", frames_dir,
            "--input_mask", mask_path,
            "--method", method,
            "--mask_frame_id", str(mask_frame_id)   # NEW FLAG PASSED DOWN
        ]
        if fps is not None:
            cmd += ["--fps", str(fps)]

        ok = self._run(cmd, "quick_run.py (TubeletGraph)", check=False,
                       cwd=str(self.project_root))
        return pred_name, str(src_pred), jpeg_images_dir, pred_out_dir, ok

    # ------------------------------------------------------------------
    # Pre-Stage 3: Temporal Alignment
    # ------------------------------------------------------------------

    def _remap_prediction_indices(self, pred_dir: str, keep_indices: List[int]):
        pred_path = Path(pred_dir)
        if not pred_path.exists():
            return

        def remap_data(data):
            if isinstance(data, dict):
                new_data = {}
                for k, v in data.items():
                    if k.isdigit():
                        idx   = int(k)
                        new_k = str(keep_indices[idx]) if idx < len(keep_indices) else k
                        new_data[new_k] = remap_data(v)
                    else:
                        if k in ("start_frame", "end_frame", "frame", "frame_idx",
                                 "start", "end", "frame_id") and isinstance(v, int):
                            new_data[k] = keep_indices[v] if v < len(keep_indices) else v
                        else:
                            new_data[k] = remap_data(v)
                return new_data
            elif isinstance(data, list):
                return [remap_data(i) for i in data]
            return data

        for json_file in pred_path.glob("*.json"):
            if "report" in json_file.name:
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                with open(json_file, 'w') as f:
                    json.dump(remap_data(data), f, indent=4)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to remap {json_file}: {e}")


    # ------------------------------------------------------------------
    # Stage 3 – Anomaly Detection
    # ------------------------------------------------------------------

    def stage3_anomaly_detection(
        self,
        video_path: str,
        pred_name: str,
        src_pred_path: str,
        vqa_metadata: Dict,
        frames_dir: str,
        sample_interval: int = 10,
        vlm_model: str = "openai",
        caption: str = "",
    ) -> Tuple[Dict, bool]:
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 3 – Anomaly Detection: {Path(video_path).name}")
            if caption:
                preview = caption[:100].replace("\n", " ")
                print(f"  Caption: '{preview}{'…' if len(caption) > 100 else ''}'")
            else:
                print("  Caption: (none)")

        rpt_path = _report_nvme_path(video_path, vqa_metadata, self.dataset_root)
        rpt_path.parent.mkdir(parents=True, exist_ok=True)

        _, stem = _nvme_rel(video_path, vqa_metadata, self.dataset_root)

        cmd = [
            "python3", self.prompt_vad_script,
            "-c", self.config_path,
            "-p", pred_name,
            "--sample_interval", str(sample_interval),
            "--video_path", frames_dir,
            "--vlm", vlm_model,
            "--output_dir", str(rpt_path.parent),
            "--detect_anomalies",
        ]

        if caption:
            cmd += ["--caption", caption]

        if self.verbose:
            cmd.append("-v")

        ok = self._run(cmd, "prompt_vad.py", check=False)

        candidates = [
            rpt_path,
            rpt_path.parent / f"{pred_name}_report.json",
            rpt_path.parent / f"{Path(frames_dir).name}_report.json",
            rpt_path.parent / f"{stem}_report.json",
        ]
        report: Dict = {}
        for p in candidates:
            if p.is_file():
                try:
                    with open(p) as fh:
                        report = json.load(fh)
                    if p.resolve() != rpt_path.resolve():
                        shutil.move(str(p), str(rpt_path))
                    break
                except (json.JSONDecodeError, IOError):
                    pass

        if not report:
            report = {
                "prediction_name":   pred_name,
                "anomaly_detected":  False,
                "num_anomalies":     0,
                "overall_severity":  "N/A",
                "anomalies":         [],
                "reasoning_trace":   [],
                "identified_events": [],
                "summary":           "",
            }

        return report, ok

    # ------------------------------------------------------------------
    # Local scratch cleanup
    # ------------------------------------------------------------------

    def _cleanup_local_scratch(self, pred_name: str, local_frames_dir: str,
                               jpeg_images_dir: str = "", pred_out_dir: str = ""):
        
        return  # skip cleanup for debugging
        root = self.project_root

        if jpeg_images_dir and Path(jpeg_images_dir).exists():
            shutil.rmtree(jpeg_images_dir, ignore_errors=True)
        else:
            jpegs = root / "_custom_dataset" / "JPEGImages" / pred_name
            if jpegs.exists():
                shutil.rmtree(jpegs, ignore_errors=True)

        annos = root / "_custom_dataset" / "Annotations" / pred_name
        if annos.exists():
            shutil.rmtree(annos, ignore_errors=True)

        interm = root / "_interm_out"
        if interm.exists():
            for cat in interm.iterdir():
                if not cat.is_dir():
                    continue
                t = cat / pred_name
                if t.exists():
                    shutil.rmtree(t, ignore_errors=True)

        if pred_out_dir and Path(pred_out_dir).exists():
            shutil.rmtree(pred_out_dir, ignore_errors=True)
        pred_out_root = root / "_pred_out"
        if pred_out_root.is_dir():
            for candidate in list(pred_out_root.iterdir()):
                if candidate.is_dir() and pred_name in candidate.name:
                    shutil.rmtree(candidate, ignore_errors=True)

        splits_dir = root / "splits"
        if splits_dir.is_dir():
            sf = splits_dir / f"custom_{pred_name}.txt"
            if sf.exists():
                sf.unlink(missing_ok=True)

        configs_dir = root / "configs"
        if configs_dir.is_dir():
            for cfg in list(configs_dir.iterdir()):
                if cfg.suffix == ".yaml" and cfg.name != "default.yaml" \
                        and pred_name in cfg.name:
                    cfg.unlink(missing_ok=True)

        if local_frames_dir:
            tmp_parent = Path(local_frames_dir).parent
            if str(tmp_parent).startswith(tempfile.gettempdir()) and tmp_parent.exists():
                shutil.rmtree(tmp_parent, ignore_errors=True)
            elif Path(local_frames_dir).exists():
                shutil.rmtree(local_frames_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Per-video pipeline
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        vqa_metadata: Dict,
        vlm_model: str = "openai",
        fps: Optional[int] = None,
        sample_interval: int = 10,
        method: str = "Ours",
        auto_mode: bool = True,
    ) -> Dict:
        vp = str(video_path)
        MAX_RETRIES = 100

        local_frames_dir = ""
        mask_path        = ""
        pred_name        = ""
        src_pred_path    = ""
        jpeg_images_dir  = ""
        pred_out_dir     = ""
        keep_indices:  List[int]     = []
        caption:       str           = ""
        dynamic_fps:   Optional[int] = None
        mask_frame_id: int           = 0  # NEW: Initialize mask_frame_id

        # ── STAGE 1 ────────────────────────────────────────────────
        ok1 = False
        for attempt in range(MAX_RETRIES):
            try:
                with suppress_output(self.verbose):
                    # NEW: Unpack the 7th returned value (mask_frame_id)
                    (local_frames_dir, mask_path, ok1,
                     keep_indices, caption, dynamic_fps, mask_frame_id) = \
                        self.stage1_object_grounding(
                            vp, vqa_metadata=vqa_metadata,
                            vlm_model=vlm_model, auto_mode=auto_mode,
                            fps=fps)
                if ok1:
                    break
            except Exception as e:
                ok1 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 1 failed: {e}")
                time.sleep(3)

        if not ok1:
            return {"success": False, "video_path": vp, "stage": 1,
                    "error": f"Object grounding failed after {MAX_RETRIES} attempts"}

        # ── FPS resolution ─────────────────────────────────────────
        effective_fps: Optional[int] = dynamic_fps if dynamic_fps is not None else fps
        if self.verbose:
            if dynamic_fps is not None:
                tqdm.write(f"  [FPS] Using VLM-recommended FPS={dynamic_fps} for Stage 2.")
            else:
                tqdm.write(f"  [FPS] VLM FPS unavailable; using CLI fps={fps} for Stage 2.")

        # ── STAGE 2 ────────────────────────────────────────────────
        ok2 = False
        for attempt in range(MAX_RETRIES):
            try:
                with suppress_output(self.verbose):
                    pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok2 = \
                        self.stage2_tracking(
                            vp, local_frames_dir, mask_path,
                            fps=effective_fps, 
                            mask_frame_id=mask_frame_id, # NEW: Pass mask_frame_id to Stage 2
                            method=method)
                if ok2:
                    break
            except Exception as e:
                ok2 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 2 failed: {e}")
                time.sleep(3)

        if not ok2:
            self._cleanup_local_scratch(pred_name, local_frames_dir,               # Test for stage 2 !!!!!!!!!!!!!!!!
                                        jpeg_images_dir, pred_out_dir)
            return {"success": False, "video_path": vp, "stage": 2,
                    "error": f"Tracking failed after {MAX_RETRIES} attempts"}

        # ── TEMPORAL ALIGNMENT ─────────────────────────────────────
        if keep_indices:
            self._remap_prediction_indices(src_pred_path, keep_indices)

        # ── STAGE 3 ────────────────────────────────────────────────
        ok3 = False
        for attempt in range(MAX_RETRIES):
            try:
                with suppress_output(self.verbose):
                    _, ok3 = self.stage3_anomaly_detection(
                        vp, pred_name, src_pred_path,
                        vqa_metadata=vqa_metadata,
                        frames_dir=local_frames_dir,
                        sample_interval=sample_interval,
                        vlm_model=vlm_model,
                        caption=caption)
                if ok3:
                    break
            except Exception as e:
                ok3 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 3 failed: {e}")
                time.sleep(3)

        if not ok3:
            self._cleanup_local_scratch(pred_name, local_frames_dir,              # Test for stage 2 !!!!!!!!!!!!!!!!
                                        jpeg_images_dir, pred_out_dir)
            return {"success": False, "video_path": vp, "stage": 3,
                    "error": f"Anomaly detection failed after {MAX_RETRIES} attempts"}

        # ── POST-PROCESSING ────────────────────────────────────────               # Test for stage 2 !!!!!!!!!!!!!!!!
        # with suppress_output(self.verbose):
        #     dst_pred = Path(_pred_nvme_dir(vp, vqa_metadata, self.dataset_root))
        #     if osp.exists(pred_out_dir):
        #         dst_pred.parent.mkdir(parents=True, exist_ok=True)
        #         for item in Path(pred_out_dir).iterdir():
        #             dst_item = dst_pred.parent / item.name
        #             if item.is_dir():
        #                 shutil.copytree(str(item), str(dst_item), dirs_exist_ok=True)
        #             else:
        #                 shutil.copy2(str(item), str(dst_item))
        #     self._pred_out(pred_name, local_frames_dir,
        #                                 jpeg_images_dir, pred_out_dir)
        #     if osp.exists(pred_out_dir):
        #         shutil.rmtree(pred_out_dir, ignore_errors=True)
        with suppress_output(self.verbose):
            dst_pred = _pred_nvme_dir(vp, vqa_metadata, self.dataset_root)
            if osp.exists(src_pred_path):
                dst_pred.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_pred_path, str(dst_pred), dirs_exist_ok=True)
            self._cleanup_local_scratch(pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)
        return {
            "success": True,
            "video_path": vp,
            "prediction_name": pred_name,
            "report_path": str(_report_nvme_path(vp, vqa_metadata, self.dataset_root)),
            "dynamic_fps": dynamic_fps,
        }


# ---------------------------------------------------------------------------
# Video file discovery
# ---------------------------------------------------------------------------

def _discover_videos(
    video_root: Path,
    vqa_metadata: Dict,
    split: str,
    dataset_root: Optional[str],
) -> List[Path]:
    """
    Discover .mp4 files under video_root that match the requested split.

    A video is included when:
      • vqa_metadata is empty  →  include everything with split/ in the path.
      • vqa_metadata is set    →  the video has a matching VQA entry AND its
                                  org_split matches the requested split.
    """
    video_files: List[Path] = []
    seen: set = set()

    for candidate in sorted(video_root.rglob("*.mp4")):
        abs_str = str(candidate)
        if abs_str in seen:
            continue

        if vqa_metadata:
            entry = vqa_metadata.get(abs_str) or vqa_metadata.get(str(candidate.resolve()))

            if entry is None and dataset_root:
                root_p = Path(dataset_root).resolve()
                try:
                    rel_str = str(candidate.relative_to(root_p))
                    entry   = vqa_metadata.get(rel_str)
                except ValueError:
                    pass

            if entry is None:
                for k, v in vqa_metadata.items():
                    if not Path(k).is_absolute() and abs_str.endswith(k.lstrip("./")):
                        entry = v
                        break

            if entry is None:
                continue

            org_split = entry.get("org_split", entry.get("split", "")).lower()
            if org_split != split.lower():
                continue
        else:
            continue

        video_files.append(candidate)
        seen.add(abs_str)

    return video_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ST-VAD AutoLab: Spatio-Temporal Video Anomaly Detection (3-stage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", choices=["analyze"])
    p.add_argument("video",
                   help="Root directory containing AutoLab videos (searched recursively "
                        "for .mp4 files filtered by --split and --vqa_file).")
    p.add_argument("-c", "--config",     required=True)
    p.add_argument("-o", "--output",     default="output",
                   help="Kept for CLI compat; actual outputs go to NVMe stage1/2/3.")
    p.add_argument("--vlm",              default="openai",
                   choices=["openai", "claude", "ollama"])
    p.add_argument("--fps",              type=int, default=None,
                   help="Fallback FPS for Stage 2 when VLM-recommended FPS is unavailable.")
    p.add_argument("--sample_interval",  type=int, default=10)
    p.add_argument("--method",           default="Ours")
    p.add_argument("--split",            type=str, default="test")
    p.add_argument("--vqa_file",         type=str, default=None)
    p.add_argument("--dataset_root",     type=str, default=None,
                   help="Absolute path to the AutoLab dataset root (e.g. "
                        "/work/nvme/bgiv/qilong/datasets/AutoLab). "
                        "Used to resolve relative VQA video_path values. "
                        "Defaults to the AutoLab/ component found in --video.")
    p.add_argument("--no-auto",          action="store_true")
    p.add_argument("-v", "--verbose",    action="store_true")
    return p


def main():
    args = get_parser().parse_args()

    video_root = Path(args.video).resolve()
    if not video_root.is_dir():
        print(f"Error: video directory does not exist: {video_root}")
        sys.exit(1)

    dataset_root: Optional[str] = args.dataset_root
    if not dataset_root:
        parts = video_root.parts
        if "AutoLab" in parts: # NOTE: Assuming DATASET_ANCHOR was "AutoLab", fallback used here.
            idx          = parts.index("AutoLab")
            dataset_root = str(Path(*parts[:idx + 1]))
        else:
            dataset_root = str(video_root)

    framework = STVADFramework(
        config_path=args.config,
        input_root=str(video_root),
        output_dir=args.output,
        verbose=args.verbose,
        dataset_root=dataset_root,
    )

    vqa_metadata: Dict = {}
    if args.vqa_file:
        if args.verbose:
            print(f"Reading VQA file: {args.vqa_file}")
        try:
            with open(args.vqa_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    vp   = data.get("video_path", "")
                    if not vp:
                        continue
                    vqa_metadata[vp] = data
        except Exception as e:
            print(f"Error reading VQA file: {e}")
            sys.exit(1)

        vqa_metadata = _build_vqa_lookup(vqa_metadata, dataset_root)

        if args.verbose:
            raw_count = sum(1 for k in vqa_metadata if not Path(k).is_absolute())
            print(f"Loaded {raw_count} VQA entries "
                  f"({len(vqa_metadata)} total keys after alias expansion).")

    video_files = _discover_videos(video_root, vqa_metadata, args.split, dataset_root)

    if args.verbose:
        print(f"Found {len(video_files)} video(s) matching split='{args.split}'.")

    if not video_files:
        print(f"No videos found under '{video_root}' matching split='{args.split}'.")
        sys.exit(0)

    framework.preload_sam3()
    failed: List[Path] = []

    for i in trange(len(video_files), desc="Processing Videos", dynamic_ncols=True):
        vp = video_files[i]

        rpt = _report_nvme_path(str(vp), vqa_metadata, dataset_root)
        if rpt.is_file():
            tqdm.write(f"  [SKIP] {rpt.name}")
            continue

        try:
            result = framework.process_video(
                video_path=str(vp),
                vqa_metadata=vqa_metadata,
                vlm_model=args.vlm,
                fps=args.fps,
                sample_interval=args.sample_interval,
                method=args.method,
                auto_mode=not args.no_auto,
            )
            if not result["success"]:
                failed.append(vp)
                tqdm.write(f"  [ERROR] Stage {result.get('stage','?')} – "
                           f"{vp.name}: {result.get('error','')}")
        except Exception as e:
            failed.append(vp)
            tqdm.write(f"  [CRITICAL] {vp.name}: {e}")
            if args.verbose:
                tqdm.write(traceback.format_exc())

    total = len(video_files)
    done  = total - len(failed)
    print(f"\nDone. {done}/{total} videos processed successfully.")

    if failed:
        print(f"{len(failed)} failure(s):")
        for ff in failed:
            print(f"  {ff}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    