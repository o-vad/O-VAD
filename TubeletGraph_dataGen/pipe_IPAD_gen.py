#!/usr/bin/env python3
"""
ST-VAD Framework: IPAD Dataset — Spatio-Temporal Video Anomaly Detection
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
             Also receives the path to a known-normal reference video (same subset,
             training/frames/01) so the VLM can compare the test video against a
             concrete normal baseline.

IPAD input: each sample is a DIRECTORY of image frames (not a video file).
  e.g.  .../IPAD_dataset/S02/testing/frames/01/  →  0000000.jpg, 0000001.jpg, ...

Normal reference video convention:
  For a test sequence under  .../IPAD_dataset/XXX/...
  the normal reference is    .../IPAD_dataset/XXX/training/frames/01/
  e.g. for S01 → /u/qilong/anomaly_detect/datasets/IPAD/IPAD_dataset/S01/training/frames/01

NVMe permanent storage mirrors the VQA source path:
  Stage 1 → /work/nvme/bgiv/qilong/stage1/IPAD/IPAD_dataset/S02/testing/frames/01_mask/0000000.png
            (mask ONLY — frames stay in local /tmp, never written to NVMe)
  Stage 2 → /work/nvme/bgiv/qilong/stage2/IPAD/IPAD_dataset/S02/testing/frames/01/
            (TubeletGraph prediction files)
  Stage 3 → /work/nvme/bgiv/qilong/stage3/IPAD/IPAD_dataset/S02/testing/frames/01_report.json

NVMe writes happen ONLY after IOU > 0.7 validation passes (or after all
MAX_CORRECTIVE_RETRIES are exhausted, in which case the best result is kept).
Local TubeletGraph scratch (_custom_dataset, _interm_out, _pred_out) and local
tmp frames are cleaned between retries and after NVMe storage is confirmed.

Naming convention for local scratch dirs (globally unique across all IPAD subsets):
  unique_stem = parts of seq_dir after "IPAD_dataset" joined by "_"
  e.g.  .../IPAD_dataset/S02/testing/frames/01  →  "S02_testing_frames_01"
  This prevents collisions when running S01 and S02 in parallel.

Usage:
    python pipe_IPAD.py analyze /path/to/IPAD_dataset/S02 \\
        -c configs/default.yaml \\
        --vqa_file /path/to/IPAD_VQA.jsonl \\
        --split test \\
        --vlm openai --fps 3 --sample_interval 1
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
# Global Storage Roots  — update to match your NVMe mount point
# ---------------------------------------------------------------------------
STAGE1_ROOT = Path("/work/nvme/bgiv/qilong/stage1")
STAGE2_ROOT = Path("/work/nvme/bgiv/qilong/stage2")
STAGE3_ROOT = Path("/work/nvme/bgiv/qilong/stage3")

# Max frames sent to the VLM for caption + FPS selection (limits cost / tokens)
CAPTION_MAX_FRAMES = 8

# ---------------------------------------------------------------------------
# Optional OpenAI import
# ---------------------------------------------------------------------------
try:
    import openai as _openai_mod
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Suppress output helper
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


# ---------------------------------------------------------------------------
# Path / naming helpers
# ---------------------------------------------------------------------------

def _unique_stem(seq_dir: str) -> str:
    """
    Globally-unique local scratch identifier for one IPAD sample.

    Anchored at the "IPAD_dataset" directory so the name is unique across
    all subsets (S01, S02, …) even when running concurrently.

    Examples:
      .../IPAD_dataset/S02/testing/frames/01   →  "S02_testing_frames_01"
      .../IPAD_dataset/S06/training/frames/01  →  "S06_training_frames_01"

    Used as:
      - basename of the local /tmp frames dir  → quick_run.py uses it as pred_name
      - key for _custom_dataset/{Annotations,JPEGImages}/<unique_stem>/
      - key for _pred_out/<unique_stem>/
    """
    parts = Path(seq_dir).resolve().parts
    if "IPAD_dataset" in parts:
        idx = parts.index("IPAD_dataset")
        rel_parts = parts[idx + 1:]
        if rel_parts:
            return "_".join(rel_parts)
    p = Path(seq_dir).resolve()
    return f"{p.parent.name}_{p.name}"


def _nvme_rel(seq_dir: str, vqa_metadata: Dict) -> Tuple[Path, str]:
    """
    Returns (parent_dir, stem) for NVMe paths — always a RELATIVE Path so that
      STAGE*_ROOT / parent_dir / stem
    is always correct.
    """
    for key in (str(seq_dir), str(Path(seq_dir).resolve())):
        entry = vqa_metadata.get(key)
        if entry:
            parts = Path(entry["video_path"]).parts
            for marker in ("IPAD", "IPAD_dataset"):
                if marker in parts:
                    idx = parts.index(marker)
                    rel_parts = parts[idx:]
                    rel = Path(*rel_parts)
                    return rel.parent, rel.name
            p = Path(entry["video_path"])
            if p.is_absolute():
                stripped = p.parts[1:]
                p = Path(*stripped) if stripped else Path(".")
            return p.parent, p.name

    # Filesystem fallback: anchor at IPAD or IPAD_dataset
    parts = Path(seq_dir).resolve().parts
    for marker in ("IPAD", "IPAD_dataset"):
        if marker in parts:
            idx = parts.index(marker)
            rel_parts = parts[idx:]
            rel = Path(*rel_parts)
            return rel.parent, rel.name

    return Path(Path(seq_dir).parent.name), Path(seq_dir).name


def _mask_nvme_dir(seq_dir: str, vqa_metadata: Dict) -> Path:
    parent, stem = _nvme_rel(seq_dir, vqa_metadata)
    return STAGE1_ROOT / parent / f"{stem}_mask"


def _pred_nvme_dir(seq_dir: str, vqa_metadata: Dict) -> Path:
    parent, stem = _nvme_rel(seq_dir, vqa_metadata)
    return STAGE2_ROOT / parent / stem


def _report_nvme_path(seq_dir: str, vqa_metadata: Dict) -> Path:
    parent, stem = _nvme_rel(seq_dir, vqa_metadata)
    return STAGE3_ROOT / parent / f"{stem}_report.json"


# ---------------------------------------------------------------------------
# Normal reference video resolution
# ---------------------------------------------------------------------------

def _resolve_normal_video_path(seq_dir: str) -> Optional[str]:
    """
    Derive the path to the known-normal reference video for a given test sequence.

    Convention: for any sequence under .../IPAD_dataset/XXX/...
    the normal reference is    .../IPAD_dataset/XXX/training/frames/01

    For example:
      seq_dir = .../IPAD_dataset/S01/testing/frames/05
      normal  = .../IPAD_dataset/S01/training/frames/01

    Returns None if the derived path does not exist on disk.
    """
    parts = Path(seq_dir).resolve().parts
    if "IPAD_dataset" not in parts:
        return None

    idx = parts.index("IPAD_dataset")
    # parts[idx]   == "IPAD_dataset"
    # parts[idx+1] == subset, e.g. "S01"
    if len(parts) <= idx + 1:
        return None

    dataset_root = Path(*parts[: idx + 1])  # .../IPAD_dataset
    subset       = parts[idx + 1]           # e.g. "S01"

    normal_path = dataset_root / subset / "training" / "frames" / "01"
    if normal_path.is_dir():
        return str(normal_path)
    return None


# ---------------------------------------------------------------------------
# OpenAI Responses API client  (used only for caption + FPS generation)
# ---------------------------------------------------------------------------

class _OpenAIResponsesClient:
    """
    Minimal OpenAI Responses-API client used for the Stage 1 captioning call.
    """

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
        """Encode a local image file as a base64 input_image block."""
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
# Caption + FPS generation  (called inside Stage 1, after object detection)
# ---------------------------------------------------------------------------

def _select_caption_frames(frames_dir: str, max_frames: int = CAPTION_MAX_FRAMES) -> List[str]:
    """
    Return up to `max_frames` evenly-spaced frame file paths from `frames_dir`.
    """
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
    Send a sample of frames from `frames_dir` to the OpenAI Responses API and ask for:

      (a) A temporal caption describing object interactions and motion across the frames.
      (b) A recommended FPS integer (2–20) for Stage 2 TubeletGraph graph construction.

    Returns (caption, dynamic_fps).  On failure returns ("", None).
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
        "You are an expert video analyst specialising in physical anomaly detection. "
        "You receive a sequence of frames sampled uniformly from a video and a list "
        "of detected objects. Your job is to describe the video content with temporal "
        "precision and to recommend an appropriate processing frame rate."
    )

    user_prompt = (
        f"I am providing {len(frame_paths)} evenly-spaced frames from a video sequence.\n"
        f"The following objects have been detected in the scene: {objects_str}.\n\n"
        "Please provide TWO things:\n\n"
        "1. A detailed TEMPORAL CAPTION describing what happens across these frames.\n"
        "   Requirements:\n"
        "   - Use description tune (firstly, then, after that, ...) and do NOT use detailed frame id.\n"
        "   - Focus on the detected objects and how they interact or change state over time.\n"
        "   - Name EACH object in your caption with appearance features like color and shape.\n"
        "   - For EACH object, explicitly state at every temporal stage whether it is\n"
        "     MOVING or STATIC, e.g. 'Object A continues to move forward, but Object B\n"
        "     stops moving and remains still from this point onward.'\n"
        "   - Describe any deformation, contact, separation, acceleration, or other\n"
        "     physical state changes in chronological order.\n"
        "   - Reference approximate frame ranges where relevant.\n\n"
        "2. A recommended FPS (frames per second) integer for video graph construction. Default 3.\n"
        "   - Range: integer between 2 and 10 (inclusive).\n"
        "   - Use a HIGHER value (e.g. 5–10) for fast-moving or highly dynamic scenes.\n"
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

    # ── Parse <CAPTION> ────────────────────────────────────────────────────
    caption_match = re.search(r"<CAPTION>(.*?)</CAPTION>", raw_response, re.DOTALL)
    caption       = caption_match.group(1).strip() if caption_match else ""

    # ── Parse <FPS> and clamp to [2, 10] ──────────────────────────────────
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
            if verbose:
                print(f"  [Caption] Raw response (first 400 chars):\n{raw_response[:400]}")

    return caption, dynamic_fps


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

class IPADFramework:
    def __init__(self, config_path: str, output_dir: str = "output", verbose: bool = False):
        self.config_path = config_path
        self.output_dir  = output_dir   # kept for CLI compat only
        self.verbose     = verbose
        self.base_dir    = osp.dirname(osp.abspath(__file__))

        self.vlm_mask_script   = osp.join(self.base_dir, "annotate", "vlm_mask_grounded_org.py")
        self.quick_run_script  = osp.join(self.base_dir, "quick_run.py")
        self.prompt_vad_script = osp.join(self.base_dir, "TubeletGraph", "vlm", "prompt_vad.py")

        self.project_root = Path(self.quick_run_script).parent
        self._sam3_segmenter = None
        self._stage1_mod: Optional[object] = None

        self._verify_scripts()
        STAGE1_ROOT.mkdir(parents=True, exist_ok=True)
        STAGE2_ROOT.mkdir(parents=True, exist_ok=True)
        STAGE3_ROOT.mkdir(parents=True, exist_ok=True)

    def _verify_scripts(self):
        required = {
            "annotate/vlm_mask_grounded_org.py":  self.vlm_mask_script,
            "quick_run.py":                    self.quick_run_script,
            "TubeletGraph/vlm/prompt_vad.py": self.prompt_vad_script,
        }
        missing = [f"{n}  (expected: {p})" for n, p in required.items() if not osp.isfile(p)]
        if missing:
            print("Missing required scripts:")
            for m in missing:
                print(f"  {m}")
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
        spec = importlib.util.spec_from_file_location("vlm_mask_grounded_org", self.vlm_mask_script)
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
            print("Pre-loading SAM3 model (once for all sequences)...")
        with suppress_output(self.verbose):
            self._get_segmenter()
        if self.verbose:
            print("SAM3 ready.\n")

    # ------------------------------------------------------------------
    # Stage 1 – Object Grounding & Segmentation  +  Caption/FPS generation
    # ------------------------------------------------------------------

    def stage1_object_grounding(
        self,
        seq_dir: str,
        vqa_metadata: Dict,
        vlm_model: str = "openai",
        auto_mode: bool = True,
        scan_frames: bool = True,
        threshold: float = 0.1,
        num_scan_frames: int = 5,
        hint_str: str = "",
        fps: Optional[int] = None,
    ) -> Tuple[str, str, bool, List[int], str, Optional[int], int]:
        """
        seq_dir : path to image-sequence directory (one IPAD sample).
        Returns  : (local_frames_tmp_dir, nvme_mask_path, success, keep_indices,
                    caption, dynamic_fps, mask_frame_id)
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 1 – Object Grounding: {Path(seq_dir).name}")

        mod       = self._load_stage1_module()
        segmenter = self._get_segmenter()

        stem = _unique_stem(seq_dir)

        _tmp_parent      = Path(tempfile.mkdtemp(prefix="stvad_ipad_s1_"))
        local_frames_dir = _tmp_parent / stem
        local_frames_dir.mkdir(parents=True, exist_ok=True)

        nvme_mask_dir = _mask_nvme_dir(seq_dir, vqa_metadata)
        keep_indices: List[int] = []

        caption:      str           = ""
        dynamic_fps:  Optional[int] = None
        mask_frame_id: int          = 0   # always defined; updated after scan

        try:
            import numpy as np

            frame_files = sorted([
                f for f in Path(seq_dir).iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ])
            if not frame_files:
                return str(local_frames_dir), "", False, [], "", None, 0

            keep_indices = list(range(len(frame_files)))

            # Hard-link (or copy) frames into local tmp dir
            for new_idx, old_idx in enumerate(keep_indices):
                old_f = frame_files[old_idx]
                new_f = local_frames_dir / f"{new_idx:07d}{old_f.suffix}"
                try:
                    os.link(old_f, new_f)
                except OSError:
                    shutil.copy2(old_f, new_f)

            # Run object detection on the frame set
            input_proc = mod.InputProcessor(str(local_frames_dir), None, frame_index=0)
            first_frame_path, _ = input_proc.process()

            detector = mod.VLMObjectDetector(vlm_model)
            objects  = detector.detect_objects(first_frame_path)
            if not objects:
                return str(local_frames_dir), "", False, [], "", None, 0

            # ── Caption + FPS generation ──────────────────────────────────
            caption, dynamic_fps = generate_caption_and_fps(
                frames_dir=str(local_frames_dir),
                objects=objects,
                vlm_model=vlm_model,
                verbose=self.verbose,
                hint_str=hint_str
            )

            # ── SAM3 segmentation ─────────────────────────────────────────
            if scan_frames and input_proc.num_frames > 1:
                best_frame_idx, mask, _ = mod.scan_frames_for_objects(
                    input_proc, objects, segmenter,
                    num_frames_to_scan=num_scan_frames, threshold=threshold)
            else:
                mask, _ = segmenter.segment(
                    first_frame_path, objects, threshold=threshold,
                    retry_with_variations=True, min_threshold=0.1)
                best_frame_idx = 0

            mask_frame_id = best_frame_idx   # update return value

            if not np.any(mask > 0):
                return str(local_frames_dir), "", False, [], caption, dynamic_fps, mask_frame_id

            # Write mask to NVMe stage1 only
            nvme_mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = nvme_mask_dir / "0000000.png"
            mod.save_vos_mask(mask, str(mask_path))
            # save_vos_mask may emit a _vis.png sibling — remove it immediately
            vis = nvme_mask_dir / "0000000_vis.png"
            if vis.exists():
                vis.unlink(missing_ok=True)

            # ── Save detected obj to NVMe stage1 ──────────────────────────
            obj_path = nvme_mask_dir / "obj.json"
            with open(obj_path, "w") as f:
                json.dump(objects, f, indent=4)

            return (str(local_frames_dir), str(mask_path), True,
                    keep_indices, caption, dynamic_fps, mask_frame_id)

        except Exception:
            if self.verbose:
                traceback.print_exc()
            return str(local_frames_dir), "", False, [], caption, dynamic_fps, mask_frame_id

    # ------------------------------------------------------------------
    # Stage 2 – TubeletGraph Tracking
    # ------------------------------------------------------------------

    def stage2_tracking(
        self,
        seq_dir: str,
        frames_dir: str,
        mask_path: str,
        fps: Optional[int],
        mask_frame_id: int = 0,
        method: str = "Ours",
        hint_str: str = "",
    ) -> Tuple[str, str, str, str, bool]:
        """
        Returns (pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok).
        """
        if self.verbose:
            fps_display = str(fps) if fps is not None else "default"
            print(f"\n{'='*60}\nSTAGE 2 – Tracking: {Path(seq_dir).name}  "
                  f"(fps={fps_display})")

        pred_name = Path(frames_dir).name

        jpeg_images_dir = str(self.project_root / "_custom_dataset" / "JPEGImages" / pred_name)
        pred_out_dir    = str(self.project_root / "_pred_out" / pred_name)
        src_pred        = self.project_root / "_pred_out" / pred_name

        cmd = [
            "python3", self.quick_run_script,
            "-c", self.config_path,
            "--input_dir", frames_dir,
            "--input_mask", mask_path,
            "--method", method,
            "--mask_frame_id", str(mask_frame_id),
            "--hint_str", hint_str,
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
        """Remaps continuous TubeletGraph frame IDs back to original dataset frame indices."""
        pred_path = Path(pred_dir)
        if not pred_path.exists():
            return

        def remap_data(data):
            if isinstance(data, dict):
                new_data = {}
                for k, v in data.items():
                    if k.isdigit():
                        idx = int(k)
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
        seq_dir: str,
        pred_name: str,
        src_pred_path: str,
        vqa_metadata: Dict,
        frames_dir: str,
        sample_interval: int = 10,
        vlm_model: str = "openai",
        caption: str = "",
        normal_video_path: Optional[str] = None,
        hint_str: str = "",
    ) -> Tuple[Dict, bool]:
        """
        Runs prompt_vad.py for anomaly detection.

        The report is written to a LOCAL temporary directory only.
        The caller (process_sample) is responsible for copying it to NVMe
        after IOU validation passes — this method never touches NVMe.

        When non-empty, the caption produced by generate_caption_and_fps() in
        Stage 1 is forwarded to prompt_vad.py via the --caption argument.

        When a normal_video_path is available, it is forwarded via
        --normal_video_path so the VLM can compare the test video against a
        known-normal baseline. No caption is generated for the reference video.
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 3 – Anomaly Detection: {Path(seq_dir).name}")
            if caption:
                preview = caption[:100].replace("\n", " ")
                print(f"  Caption: '{preview}{'…' if len(caption) > 100 else ''}'")
            else:
                print("  Caption: (none)")
            if normal_video_path:
                print(f"  Normal reference video: {normal_video_path}")
            else:
                print("  Normal reference video: (none)")

        # Use a local temp directory — never write directly to NVMe here.
        local_out_dir = Path(tempfile.mkdtemp(prefix="stvad_ipad_s3_"))
        _, stem = _nvme_rel(seq_dir, vqa_metadata)

        cmd = [
            "python3", self.prompt_vad_script,
            "-c", self.config_path,
            "-p", pred_name,
            "--sample_interval", str(sample_interval),
            "--video_path", frames_dir,
            "--vlm", vlm_model,
            "--output_dir", str(local_out_dir),
            "--detect_anomalies",
        ]

        if caption:
            cmd += ["--caption", caption]

        if normal_video_path:
            cmd += ["--normal_video_path", normal_video_path]

        if self.verbose:
            cmd.append("-v")

        if len(hint_str) > 0:
            cmd += ["--hint_str", hint_str]

        ok = self._run(cmd, "prompt_vad.py", check=False)

        # Search for the report file in the local output directory.
        candidates = [
            local_out_dir / f"{stem}_report.json",
            local_out_dir / f"{pred_name}_report.json",
            local_out_dir / f"{Path(frames_dir).name}_report.json",
        ]
        # Also pick up any *_report.json that prompt_vad.py may have written.
        for p in sorted(local_out_dir.glob("*_report.json")):
            if p not in candidates:
                candidates.append(p)

        report: Dict = {}
        for p in candidates:
            if p.is_file():
                try:
                    with open(p) as fh:
                        report = json.load(fh)
                    break
                except (json.JSONDecodeError, IOError):
                    pass

        # Clean up the local temp dir — the report dict is now in memory.
        shutil.rmtree(str(local_out_dir), ignore_errors=True)

        if not report:
            report = {
                "prediction_name": pred_name,
                "anomaly_detected": False,
                "num_anomalies": 0,
                "overall_severity": "N/A",
                "anomalies": [],
                "reasoning_trace": [],
                "identified_events": [],
                "summary": "",
            }

        return report, ok

    # ------------------------------------------------------------------
    # Local scratch cleanup  (called between retries and after success)
    # ------------------------------------------------------------------

    def _cleanup_local_scratch(self, pred_name: str, local_frames_dir: str,
                                jpeg_images_dir: str = "", pred_out_dir: str = ""):
        """Removes all per-sample local scratch (never touches NVMe)."""
        root = self.project_root

        if jpeg_images_dir and Path(jpeg_images_dir).exists():
            shutil.rmtree(jpeg_images_dir, ignore_errors=True)
        else:
            p = root / "_custom_dataset" / "JPEGImages" / pred_name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

        p = root / "_custom_dataset" / "Annotations" / pred_name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

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
    # Per-sample pipeline
    # ------------------------------------------------------------------

    def process_sample(
        self,
        seq_dir: str,
        vqa_metadata: Dict,
        vlm_model: str = "openai",
        fps: Optional[int] = None,
        sample_interval: int = 10,
        method: str = "Ours",
        auto_mode: bool = True,
        hint: bool = False,
    ) -> Dict:
        """
        Full three-stage pipeline for one IPAD sequence directory.

        NVMe writes happen ONLY after IOU > 0.7 validation passes.  On each
        corrective retry all local scratch from the previous attempt is wiped
        before re-running all three stages.  If no attempt achieves IOU > 0.7
        the best-scoring result (highest IOU) is persisted to NVMe at the end.
        """
        vp = str(seq_dir)
        MAX_RETRIES            = 50   # per-stage retry limit (transient errors)
        MAX_CORRECTIVE_RETRIES = 5    # full pipeline re-runs for IOU < 0.7

        normal_video_path = None

        # ── Process Hint ───────────────────────────────────────────────────
        hint_str = ""
        if hint:
            video_level_hint = vqa_metadata.get("answer", "Normal")
            if video_level_hint.lower() == "abnormal":
                context = vqa_metadata.get("context", [])
                indices = [i for i, val in enumerate(context) if val == 1]
                if not indices:
                    frame_level_hint = ""
                else:
                    ranges = []
                    start = indices[0]
                    prev  = indices[0]
                    for current in indices[1:]:
                        if current != prev + 1:
                            ranges.append(
                                f"{start}-{prev}" if start != prev else str(start))
                            start = current
                        prev = current
                    ranges.append(
                        f"{start}-{prev}" if start != prev else str(start))
                    frame_level_hint = ", ".join(ranges)
                hint_str = (
                    f"The video is {video_level_hint}, "
                    f"with abnormal frames [{frame_level_hint}]."
                )
            else:
                hint_str = f"The video is {video_level_hint}, all frames are normal."

        # ── Track the best result across corrective retries ───────────────
        best_iou:    float = -1.0
        best_report: Dict  = {}
        best_pred_name:     str = ""
        best_src_pred_path: str = ""

        for iteration in range(MAX_CORRECTIVE_RETRIES):

            if self.verbose:
                print(f"\n[Corrective iteration {iteration + 1}/{MAX_CORRECTIVE_RETRIES}]")

            # Reset per-iteration locals so cleanup is always consistent
            local_frames_dir = ""
            mask_path        = ""
            pred_name        = ""
            src_pred_path    = ""
            jpeg_images_dir  = ""
            pred_out_dir     = ""
            keep_indices:  List[int]     = []
            caption:       str           = ""
            dynamic_fps:   Optional[int] = None
            mask_frame_id: int           = 0
            report:        Dict          = {}

            # ── STAGE 1 ────────────────────────────────────────────────────
            ok1 = False
            for attempt in range(MAX_RETRIES):
                try:
                    with suppress_output(self.verbose):
                        (local_frames_dir, mask_path, ok1,
                         keep_indices, caption, dynamic_fps, mask_frame_id) = \
                            self.stage1_object_grounding(
                                vp, vqa_metadata=vqa_metadata,
                                vlm_model=vlm_model, auto_mode=auto_mode,
                                hint_str=hint_str, fps=fps)
                    if ok1:
                        break
                except Exception as e:
                    ok1 = False
                    if self.verbose:
                        print(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 1 failed: {e}")
                    time.sleep(3)

            if not ok1:
                # Clean up whatever was created before giving up on this iteration
                self._cleanup_local_scratch(
                    pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)
                return {"success": False, "seq_dir": vp, "stage": 1,
                        "error": f"Object grounding failed after {MAX_RETRIES} attempts"}

            # ── FPS resolution ─────────────────────────────────────────────
            effective_fps: Optional[int] = (
                dynamic_fps if dynamic_fps is not None else fps)
            if self.verbose:
                if dynamic_fps is not None:
                    print(f"  [FPS] Using VLM-recommended FPS={dynamic_fps} for Stage 2.")
                else:
                    print(f"  [FPS] VLM FPS unavailable; using CLI fps={fps} for Stage 2.")

            # ── STAGE 2 ────────────────────────────────────────────────────
            ok2 = False
            for attempt in range(MAX_RETRIES):
                try:
                    with suppress_output(self.verbose):
                        pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok2 = \
                            self.stage2_tracking(
                                vp, local_frames_dir, mask_path,
                                fps=effective_fps,
                                mask_frame_id=mask_frame_id,
                                method=method, hint_str=hint_str)
                    if ok2:
                        break
                except Exception as e:
                    ok2 = False
                    if self.verbose:
                        print(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 2 failed: {e}")
                    time.sleep(3)

            if not ok2:
                self._cleanup_local_scratch(
                    pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)
                return {"success": False, "seq_dir": vp, "stage": 2,
                        "error": f"Tracking failed after {MAX_RETRIES} attempts"}

            # ── TEMPORAL ALIGNMENT ─────────────────────────────────────────
            if keep_indices:
                self._remap_prediction_indices(src_pred_path, keep_indices)

            # ── STAGE 3 ────────────────────────────────────────────────────
            ok3 = False
            for attempt in range(MAX_RETRIES):
                try:
                    with suppress_output(self.verbose):
                        report, ok3 = self.stage3_anomaly_detection(
                            vp, pred_name, src_pred_path,
                            vqa_metadata=vqa_metadata,
                            frames_dir=local_frames_dir,
                            sample_interval=sample_interval,
                            vlm_model=vlm_model,
                            caption=caption,
                            normal_video_path=normal_video_path,
                            hint_str=hint_str,
                        )
                    if ok3:
                        break
                except Exception as e:
                    ok3 = False
                    if self.verbose:
                        print(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 3 failed: {e}")
                    time.sleep(3)

            if not ok3:
                self._cleanup_local_scratch(
                    pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)
                return {"success": False, "seq_dir": vp, "stage": 3,
                        "error": f"Anomaly detection failed after {MAX_RETRIES} attempts"}

            # ── IOU VALIDATION ─────────────────────────────────────────────
            anomaly_detected = report.get("anomaly_detected", False)
            label = vqa_metadata.get("answer", "Normal").lower()
            context          = vqa_metadata.get("context", [])
            answer_sheet     = [0] * len(context)
            if label == "normal":
                if not anomaly_detected:
                    iou = 1.0
                else:
                    iou = 0.0
            else:
                if anomaly_detected:
                    for anomaly in report.get("anomalies", []):
                        start = anomaly.get("start_frame", 0)
                        end   = anomaly.get("end_frame",   0)
                        for i in range(start, end + 1):
                            if 0 <= i < len(answer_sheet):
                                answer_sheet[i] = 1

                intersection = sum(c == 1 and a == 1 for c, a in zip(context, answer_sheet))
                union        = sum(c == 1 or  a == 1 for c, a in zip(context, answer_sheet))
                iou          = intersection / union if union > 0 else 0.0

                if self.verbose:
                    print(f"  [IOU] iteration={iteration+1}  IOU={iou:.4f}  "
                        f"(threshold=0.7)")

            # Keep track of the best attempt so far
            if iou > best_iou:
                best_iou          = iou
                best_report       = report
                best_pred_name    = pred_name
                best_src_pred_path = src_pred_path
                report["iou_with_ground_truth"] = iou

            # ── IOU passed: persist to NVMe and stop iterating ────────────
            if iou > 0.7:
                with suppress_output(self.verbose):
                    # Stage 2 → NVMe
                    dst_pred = _pred_nvme_dir(vp, vqa_metadata)
                    if osp.exists(src_pred_path):
                        dst_pred.parent.mkdir(parents=True, exist_ok=True)
                        if dst_pred.exists():
                            shutil.rmtree(str(dst_pred))
                        shutil.copytree(src_pred_path, str(dst_pred))
                    # Stage 3 → NVMe (written here for the first and only time)
                    rpt_path = _report_nvme_path(vp, vqa_metadata)
                    rpt_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(rpt_path, "w") as fh:
                        json.dump(report, fh, indent=4)

                self._cleanup_local_scratch(
                    pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)

                return {
                    "success":           True,
                    "seq_dir":           vp,
                    "prediction_name":   pred_name,
                    "report_path":       str(_report_nvme_path(vp, vqa_metadata)),
                    "anomaly_detected":  report.get("anomaly_detected", False),
                    "caption":           caption,
                    "normal_video_path": normal_video_path,
                    "dynamic_fps":       dynamic_fps,
                    "iou":               iou,
                }

            # ── IOU failed: clean up this iteration and retry ──────────────
            if self.verbose:
                print(f"  [IOU] IOU={iou:.4f} < 0.7 — cleaning up and retrying …")

            # On the last iteration do NOT clean up yet — the fallback NVMe
            # write below needs best_src_pred_path to still exist on disk.
            is_last_iteration = (iteration == MAX_CORRECTIVE_RETRIES - 1)
            if not is_last_iteration:
                self._cleanup_local_scratch(
                    pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)

        # ── All corrective retries exhausted without reaching IOU > 0.7 ───
        # Persist the best result we obtained so the sample is not lost.
        if self.verbose:
            print(f"  [IOU] Exhausted {MAX_CORRECTIVE_RETRIES} corrective retries. "
                  f"Best IOU={best_iou:.4f}. Persisting best result to NVMe.")

        if best_report:
            with suppress_output(self.verbose):
                # Stage 2 → NVMe
                dst_pred = _pred_nvme_dir(vp, vqa_metadata)
                if best_src_pred_path and osp.exists(best_src_pred_path):
                    dst_pred.parent.mkdir(parents=True, exist_ok=True)
                    if dst_pred.exists():
                        shutil.rmtree(str(dst_pred))
                    shutil.copytree(best_src_pred_path, str(dst_pred))
                # Stage 3 → NVMe
                rpt_path = _report_nvme_path(vp, vqa_metadata)
                rpt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(rpt_path, "w") as fh:
                    json.dump(best_report, fh, indent=4)

        # Clean up the last iteration's scratch (skipped above to allow the copy).
        self._cleanup_local_scratch(
            pred_name, local_frames_dir, jpeg_images_dir, pred_out_dir)

        return {
            "success":           True,   # pipeline completed even if IOU < 0.7
            "seq_dir":           vp,
            "prediction_name":   best_pred_name,
            "report_path":       str(_report_nvme_path(vp, vqa_metadata)),
            "anomaly_detected":  best_report.get("anomaly_detected", False),
            "caption":           caption,
            "normal_video_path": normal_video_path,
            "dynamic_fps":       dynamic_fps,
            "iou":               best_iou,
            "warning":           f"IOU never exceeded 0.7 (best={best_iou:.4f})",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "ST-VAD IPAD: Spatio-Temporal Anomaly Detection for the IPAD dataset.\n\n"
            "Example:\n"
            "  python pipe_IPAD.py analyze /path/to/IPAD_dataset/S02 \\\n"
            "      -c configs/default.yaml \\\n"
            "      --vqa_file /path/to/IPAD_VQA.jsonl --split test \\\n"
            "      --vlm openai --fps 3 --sample_interval 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command",   choices=["analyze"])
    p.add_argument("video",
                   help="Path to the IPAD subset directory to process, "
                        "e.g. /path/to/IPAD_dataset/S02.")
    p.add_argument("-c", "--config",     required=True,
                   help="TubeletGraph YAML config file.")
    p.add_argument("-o", "--output",     default="output",
                   help="Kept for CLI compat; actual outputs go to NVMe stage1/2/3.")
    p.add_argument("--vqa_file",         required=True,
                   help="Path to IPAD_VQA.jsonl.")
    p.add_argument("--split",            default="test",
                   help="Value of the 'split' field in IPAD_VQA.jsonl to process "
                        "(default: test).")
    p.add_argument("--vlm",              default="openai",
                   choices=["openai", "claude", "ollama"])
    p.add_argument("--fps",              type=int,   default=None,
                   help="Fallback FPS for Stage 2 when the VLM-recommended FPS is "
                        "unavailable.")
    p.add_argument("--sample_interval",  type=int,   default=10,
                   help="Frame sampling interval passed to prompt_vad.py.")
    p.add_argument("--method",           default="Ours")
    p.add_argument("--no-auto",          action="store_true",
                   help="Disable automatic object detection in Stage 1.")
    p.add_argument("-v", "--verbose",    action="store_true")
    p.add_argument("--hint",             action="store_true")
    return p


def main():
    args = get_parser().parse_args()

    framework = IPADFramework(
        config_path=args.config,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # ── Load VQA metadata ──────────────────────────────────────────────────
    if args.verbose:
        print(f"Reading VQA file: {args.vqa_file}")

    vqa_metadata: Dict = {}
    try:
        with open(args.vqa_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                vp = data.get("video_path", "")
                if not vp:
                    continue
                vqa_metadata[vp] = data
                vqa_metadata[str(Path(vp).resolve())] = data
    except Exception as e:
        print(f"Error reading VQA file: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(vqa_metadata)} VQA entries (×2 keys).")

    # ── Discover image-sequence directories under args.video ──────────────
    video_root = Path(args.video).resolve()
    if not video_root.is_dir():
        print(f"Error: video directory does not exist: {video_root}")
        sys.exit(1)

    seq_dirs: List[Path] = []
    seen: set = set()

    for candidate in sorted(video_root.rglob("*")):
        if not candidate.is_dir():
            continue
        if str(candidate) in seen:
            continue
        has_images = (any(candidate.glob("*.jpg")) or
                      any(candidate.glob("*.jpeg")) or
                      any(candidate.glob("*.png")))
        if not has_images:
            continue

        entry = (vqa_metadata.get(str(candidate))
                 or vqa_metadata.get(str(candidate.resolve())))
        if entry is None:
            continue

        if entry.get("org_split") != args.split:
            continue

        seq_dirs.append(candidate)
        seen.add(str(candidate))

    if args.verbose:
        print(f"Found {len(seq_dirs)} sequence(s) under '{video_root}' "
              f"with split='{args.split}'.")

    if not seq_dirs:
        print(f"No sequences found under '{args.video}' matching split='{args.split}'. "
              f"Check --vqa_file and --split.")
        sys.exit(0)

    framework.preload_sam3()

    failed: List[Path] = []

    for i in trange(len(seq_dirs), desc="Processing Sequences", dynamic_ncols=True):
        vp = seq_dirs[i]

        rpt = _report_nvme_path(str(vp), vqa_metadata)
        if rpt.is_file():
            print(f"  [SKIP] {rpt.name} already exists.")
            continue

        # Retrieve the per-sample VQA entry so process_sample can read
        # fields like "answer", "context", and "anomaly_status".
        sample_entry = (vqa_metadata.get(str(vp))
                        or vqa_metadata.get(str(vp.resolve()))
                        or {})

        try:
            result = framework.process_sample(
                seq_dir=str(vp),
                vqa_metadata=sample_entry,
                vlm_model=args.vlm,
                fps=args.fps,
                sample_interval=args.sample_interval,
                method=args.method,
                auto_mode=not args.no_auto,
                hint=args.hint,
            )
            if not result["success"]:
                failed.append(vp)
                print(f"  [ERROR] Stage {result.get('stage', '?')} – "
                      f"{vp.name}: {result.get('error', '')}")
            elif "warning" in result and args.verbose:
                print(f"  [WARN] {vp.name}: {result['warning']}")
        except Exception as e:
            failed.append(vp)
            print(f"  [CRITICAL] {vp.name}: {e}")
            if args.verbose:
                print(traceback.format_exc())

    if args.verbose:
        total = len(seq_dirs)
        print(f"\n{'='*60}")
        print(f"Inference done.  {total - len(failed)}/{total} succeeded.")

    if failed:
        if args.verbose:
            print(f"{len(failed)} failure(s):")
            for ff in failed:
                print(f"  {ff}")
        sys.exit(1)


if __name__ == "__main__":
    main()
