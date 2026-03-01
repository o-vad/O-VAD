#!/usr/bin/env python3
"""
ST-VAD Framework: IPAD Dataset — Spatio-Temporal Video Anomaly Detection
Three-stage pipeline:

  Stage 1 – Object Grounding   : VLM detects objects; shared SAM3 segments them.
             After object detection, a second VLM call produces:
               • A temporal caption describing object interactions / motion across frames.
               • A recommended FPS (int, 2–20) for Stage 2 graph construction.
  Stage 2 – Object Tracking    : quick_run.py / TubeletGraph builds tubelets + state graph.
             Uses the VLM-recommended FPS (falling back to --fps if unavailable).
  Stage 3 – Anomaly Detection  : prompt_vad.py reasons over state graph with VLM.
             Receives the Stage 1 caption via --caption so the VLM can cross-reference
             observed object motion with the graph-derived state changes.

IPAD input: each sample is a DIRECTORY of image frames (not a video file).
  e.g.  .../IPAD_dataset/S02/testing/frames/01/  →  0000000.jpg, 0000001.jpg, ...

NVMe permanent storage mirrors the VQA source path:
  Stage 1 → /work/nvme/bgiv/qilong/stage1/IPAD/IPAD_dataset/S02/testing/frames/01_mask/0000000.png
            (mask ONLY — frames stay in local /tmp, never written to NVMe)
  Stage 2 → /work/nvme/bgiv/qilong/stage2/IPAD/IPAD_dataset/S02/testing/frames/01/
            (TubeletGraph prediction files)
  Stage 3 → /work/nvme/bgiv/qilong/stage3/IPAD/IPAD_dataset/S02/testing/frames/01_report.json

Local TubeletGraph scratch (_custom_dataset, _interm_out, _pred_out) and local
tmp frames are cleaned per-sample immediately after NVMe storage is confirmed.

Naming convention for local scratch dirs (globally unique across all IPAD subsets):
  unique_stem = parts of seq_dir after "IPAD_dataset" joined by "_"
  e.g.  .../IPAD_dataset/S02/testing/frames/01  →  "S02_testing_frames_01"
  This prevents collisions when running S01 and S02 in parallel.

Metrics (Stage 4, printed after all inference is done):
  Video-level : Accuracy, Precision, Recall, F1, AUROC
                (report["anomaly_detected"] vs VQA["answer"])
  Frame-level : AUROC
                (report["anomalies"][*].start_frame/end_frame vs VQA["context"])

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
        # Parts AFTER "IPAD_dataset": (S02, testing, frames, 01)
        rel_parts = parts[idx + 1:]
        if rel_parts:
            return "_".join(rel_parts)
    # Fallback: use last two components (subset + seq_id)
    p = Path(seq_dir).resolve()
    return f"{p.parent.name}_{p.name}"


def _nvme_rel(seq_dir: str, vqa_metadata: Dict) -> Tuple[Path, str]:
    """
    Returns (parent_dir, stem) for NVMe paths — always a RELATIVE Path so that
      STAGE*_ROOT / parent_dir / stem
    is always correct.

    Anchors the relative path at the "IPAD" or "IPAD_dataset" directory so the
    NVMe layout is clean and readable:
      stage3/IPAD/IPAD_dataset/S01/testing/frames/01_report.json

    Python Path join bug: Path("/a") / Path("/b") == Path("/b") — absolute wins!
    We always extract a relative slice from the VQA video_path to prevent this.

    Priority:
      1. VQA "video_path" field — find "IPAD" or "IPAD_dataset" anchor inside it
      2. Marker-based filesystem fallback
    """
    for key in (str(seq_dir), str(Path(seq_dir).resolve())):
        entry = vqa_metadata.get(key)
        if entry:
            parts = Path(entry["video_path"]).parts
            # Find the highest-level IPAD marker to keep the path short and readable
            for marker in ("IPAD", "IPAD_dataset"):
                if marker in parts:
                    idx = parts.index(marker)
                    rel_parts = parts[idx:]      # e.g. (IPAD, IPAD_dataset, S01, testing, frames, 01)
                    rel = Path(*rel_parts)
                    return rel.parent, rel.name  # parent=IPAD/.../frames, stem=01
            # Marker not found — fall back to stripping the leading "/"
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
# OpenAI Responses API client  (used only for caption + FPS generation)
# ---------------------------------------------------------------------------

class _OpenAIResponsesClient:
    """
    Minimal OpenAI Responses-API client used for the Stage 1 captioning call.

    Uses the same API pattern as the project's OpenAIClient (Responses API).
    Raises ImportError when the 'openai' package is not installed so callers
    can fall back gracefully.
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
    Frames are already thinned/sampled (0000000.jpg, 0000001.jpg, …).
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
) -> Tuple[str, Optional[int]]:
    """
    Send a sample of frames from `frames_dir` to the OpenAI Responses API and ask for:

      (a) A temporal caption describing object interactions and motion across the frames.
          Each detected object must be explicitly stated as MOVING or STATIC at each
          temporal stage, e.g. "Object A continues to move forward, but Object B stops."

      (b) A recommended FPS integer (2–20) for Stage 2 TubeletGraph graph construction.
          Higher FPS for fast/dynamic scenes; lower for slow/static scenes.

    Parameters
    ----------
    frames_dir : str
        Directory containing the thinned/sampled frames (output of Stage 1 frame extraction).
        Frames are named 0000000.jpg, 0000001.jpg, … in chronological order.
    objects : list[str]
        Object names/descriptions detected by the VLM in Stage 1
        (e.g. ["person", "trolley"]).
    vlm_model : str
        VLM backend identifier.  Only "openai" is supported for this call.
        For other backends the function returns ("", None) immediately.
    verbose : bool
        Print progress and warnings to stdout when True.

    Returns
    -------
    caption : str
        Temporal description of the video focusing on object motion and interactions.
        Empty string when the call fails or vlm_model != "openai".
    dynamic_fps : int | None
        VLM-recommended FPS clamped to [2, 20].
        None when parsing fails or the call is skipped (caller uses --fps fallback).

    VLM response format (STRICTLY enforced in the prompt):
        <CAPTION>
        [free-form temporal caption]
        </CAPTION>
        <FPS>
        [integer 2–20]
        </FPS>

    Parsing
    -------
    Caption: extracted with  re.search(r'<CAPTION>(.*?)</CAPTION>', response, re.DOTALL)
    FPS    : extracted with  re.search(r'<FPS>\\s*(\\d+)\\s*</FPS>', response)
             then clamped:   max(2, min(20, fps_raw))
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
        "   - For EACH object, explicitly state at every temporal stage whether it is\n"
        "     MOVING or STATIC, e.g. 'Object A continues to move forward, but Object B\n"
        "     stops moving and remains still from this point onward.'\n"
        "   - Describe any deformation, contact, separation, acceleration, or other\n"
        "     physical state changes in chronological order.\n"
        "   - Reference approximate frame ranges where relevant.\n\n"
        "2. A recommended FPS (frames per second) integer for video graph construction. Default 3.\n"
        "   - Range: integer between 2 and 20 (inclusive).\n"
        "   - Use a HIGHER value (e.g. 10–20) for fast-moving or highly dynamic scenes.\n"
        "   - Use a LOWER value (e.g. 2–5) for slow, static, or gradually changing scenes.\n"
        "   - Choose the value that best captures the temporal density of meaningful\n"
        "     state changes without redundant frames.\n\n"
        "Respond using EXACTLY this format (no text outside the tags):\n\n"
        "<CAPTION>\n"
        "[Your detailed temporal caption here]\n"
        "</CAPTION>\n"
        "<FPS>\n"
        "[Integer between 2 and 20]\n"
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

    # ── Parse <FPS> and clamp to [2, 20] ──────────────────────────────────
    fps_match = re.search(r"<FPS>\s*(\d+)\s*</FPS>", raw_response)
    if fps_match:
        fps_raw     = int(fps_match.group(1))
        dynamic_fps: Optional[int] = max(2, min(20, fps_raw))
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

        self.vlm_mask_script   = osp.join(self.base_dir, "annotate", "vlm_mask_grounded.py")
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
            "annotate/vlm_mask_grounded.py":  self.vlm_mask_script,
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
        fps: Optional[int] = None,
    ) -> Tuple[str, str, bool, List[int], str, Optional[int]]:
        """
        seq_dir : path to image-sequence directory (one IPAD sample).
        Returns  : (local_frames_tmp_dir, nvme_mask_path, success, keep_indices,
                    caption, dynamic_fps)

        Frames → local /tmp only  (large, transient — NOT written to NVMe).
        Mask   → NVMe stage1      (small, permanent).
        local_frames_tmp_dir is kept alive until _cleanup_local_scratch() runs
        after stage 3 completes.

        After object detection, a second VLM call via generate_caption_and_fps()
        produces a temporal caption and a recommended FPS for Stage 2.
        On any failure, caption="" and dynamic_fps=None are returned so the
        pipeline continues unaffected.

        This call happens BEFORE the SAM3 segmentation so the VLM receives an
        unobstructed view of the original frames.
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 1 – Object Grounding: {Path(seq_dir).name}")

        mod       = self._load_stage1_module()
        segmenter = self._get_segmenter()

        # unique_stem is the local scratch key — globally unique across IPAD subsets
        stem = _unique_stem(seq_dir)

        # Frames live in /tmp/stvad_ipad_s1_XXXX/<unique_stem>/
        # The basename of local_frames_dir == unique_stem == pred_name.
        _tmp_parent      = Path(tempfile.mkdtemp(prefix="stvad_ipad_s1_"))
        local_frames_dir = _tmp_parent / stem
        local_frames_dir.mkdir(parents=True, exist_ok=True)

        nvme_mask_dir = _mask_nvme_dir(seq_dir, vqa_metadata)
        keep_indices: List[int] = []

        # Initialise new return values
        caption:     str           = ""
        dynamic_fps: Optional[int] = None

        try:
            import numpy as np

            frame_files = sorted([
                f for f in Path(seq_dir).iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ])
            if not frame_files:
                return str(local_frames_dir), "", False, [], "", None

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
                return str(local_frames_dir), "", False, [], "", None

            # ── Caption + FPS generation ──────────────────────────────────
            # Called immediately after object detection and BEFORE the SAM3
            # segmentation, so the VLM sees the original unmasked frames and
            # the full temporal sequence already available in local_frames_dir.
            caption, dynamic_fps = generate_caption_and_fps(
                frames_dir=str(local_frames_dir),
                objects=objects,
                vlm_model=vlm_model,
                verbose=self.verbose,
            )

            # ── SAM3 segmentation ─────────────────────────────────────────
            if scan_frames and input_proc.num_frames > 1:
                _, mask, _ = mod.scan_frames_for_objects(
                    input_proc, objects, segmenter,
                    num_frames_to_scan=num_scan_frames, threshold=threshold)
            else:
                mask, _ = segmenter.segment(
                    first_frame_path, objects, threshold=threshold,
                    retry_with_variations=True, min_threshold=0.1)

            if not np.any(mask > 0):
                # Segmentation failed but we still have valid caption/fps —
                # return them so the caller can decide what to do.
                return str(local_frames_dir), "", False, [], caption, dynamic_fps

            # Write mask to NVMe stage1 only
            nvme_mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = nvme_mask_dir / "0000000.png"
            mod.save_vos_mask(mask, str(mask_path))
            # save_vos_mask may emit a _vis.png sibling — remove it immediately
            vis = nvme_mask_dir / "0000000_vis.png"
            if vis.exists():
                vis.unlink(missing_ok=True)
            
            # ── Save detected obj to NVMe stage1 ──────────────────────────────────
            obj_path = nvme_mask_dir / "obj.json"
            with open(obj_path, "w") as f:
                json.dump(objects, f, indent=4)

            return str(local_frames_dir), str(mask_path), True, keep_indices, caption, dynamic_fps

        except Exception:
            if self.verbose:
                traceback.print_exc()
            return str(local_frames_dir), "", False, [], caption, dynamic_fps
        # NOTE: local_frames_dir intentionally kept alive here;
        # _cleanup_local_scratch() removes it after stage 3.

    # ------------------------------------------------------------------
    # Stage 2 – TubeletGraph Tracking
    # ------------------------------------------------------------------

    def stage2_tracking(
        self,
        seq_dir: str,
        frames_dir: str,
        mask_path: str,
        fps: Optional[int],
        method: str = "Ours",
    ) -> Tuple[str, str, str, str, bool]:
        """
        Returns (pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok).

        pred_name == Path(frames_dir).name == unique_stem.
        jpeg_images_dir and pred_out_dir are the EXACT local scratch paths
        recorded BEFORE running, so _cleanup_local_scratch can delete them
        precisely even if quick_run.py's internal rename fails.

        The `fps` parameter receives the already-resolved effective FPS from
        process_sample() — either the VLM-recommended dynamic_fps or the CLI
        --fps fallback — so no additional logic is needed here.
        """
        if self.verbose:
            fps_display = str(fps) if fps is not None else "default"
            print(f"\n{'='*60}\nSTAGE 2 – Tracking: {Path(seq_dir).name}  "
                  f"(fps={fps_display})")

        pred_name = Path(frames_dir).name   # == unique_stem, e.g. "S02_testing_frames_01"

        # Record exact paths before the subprocess runs
        jpeg_images_dir = str(self.project_root / "_custom_dataset" / "JPEGImages" / pred_name)
        pred_out_dir    = str(self.project_root / "_pred_out" / pred_name)
        src_pred        = self.project_root / "_pred_out" / pred_name

        cmd = [
            "python3", self.quick_run_script,
            "-c", self.config_path,
            "--input_dir", frames_dir,   # local /tmp frames
            "--input_mask", mask_path,   # NVMe stage1 mask
            "--method", method,
        ]
        if fps is not None:
            cmd += ["--fps", str(fps)]

        # cwd=project_root ensures TubeletGraph internal paths resolve correctly and
        # _custom_dataset/_interm_out/_pred_out are created inside the project dir.
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
    ) -> Tuple[Dict, bool]:
        """
        Runs prompt_vad.py for anomaly detection.

        When non-empty, the caption produced by generate_caption_and_fps() in
        Stage 1 is forwarded to prompt_vad.py via the --caption argument so the
        VLM can cross-reference observed object motion with the state graph.

        If the caption is empty (e.g. OpenAI call failed, or vlm_model != "openai"),
        the --caption argument is simply omitted and prompt_vad.py behaves as before.
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 3 – Anomaly Detection: {Path(seq_dir).name}")
            if caption:
                preview = caption[:100].replace("\n", " ")
                print(f"  Caption: '{preview}{'…' if len(caption) > 100 else ''}'")
            else:
                print("  Caption: (none)")

        rpt_path = _report_nvme_path(seq_dir, vqa_metadata)
        rpt_path.parent.mkdir(parents=True, exist_ok=True)

        _, stem = _nvme_rel(seq_dir, vqa_metadata)

        cmd = [
            "python3", self.prompt_vad_script,
            "-c", self.config_path,
            "-p", pred_name,
            "--sample_interval", str(sample_interval),
            "--video_path", frames_dir,            # local /tmp frames (actual images)
            "--vlm", vlm_model,
            "--output_dir", str(rpt_path.parent),  # write directly into NVMe stage3 dir
            "--detect_anomalies",
        ]

        if caption:
            cmd += ["--caption", caption]

        if self.verbose:
            cmd.append("-v")

        ok = self._run(cmd, "prompt_vad.py", check=False)

        # prompt_vad.py may write the report under pred_name or stem; normalise to canonical
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
    # Local scratch cleanup  (called only after NVMe data is confirmed)
    # ------------------------------------------------------------------

    def _cleanup_local_scratch(self, pred_name: str, local_frames_dir: str,
                                jpeg_images_dir: str = "", pred_out_dir: str = ""):
        """
        Removes all per-sample local scratch:
          - _custom_dataset/JPEGImages/<pred_name>/    (exact recorded path)
          - _custom_dataset/Annotations/<pred_name>/
          - _interm_out/*/<pred_name>/
          - _pred_out/<pred_name>/  +  any residual custom-<pred_name>-* folders
          - splits/custom_<pred_name>.txt
          - configs/*<pred_name>*.yaml
          - /tmp/stvad_ipad_s1_XXXX/<pred_name>/  (local frames, via parent dir)

        NVMe data is never touched.
        """
        root = self.project_root

        # _custom_dataset/JPEGImages
        if jpeg_images_dir and Path(jpeg_images_dir).exists():
            shutil.rmtree(jpeg_images_dir, ignore_errors=True)
        else:
            p = root / "_custom_dataset" / "JPEGImages" / pred_name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

        # _custom_dataset/Annotations
        p = root / "_custom_dataset" / "Annotations" / pred_name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

        # _interm_out
        interm = root / "_interm_out"
        if interm.exists():
            for cat in interm.iterdir():
                if not cat.is_dir():
                    continue
                t = cat / pred_name
                if t.exists():
                    shutil.rmtree(t, ignore_errors=True)

        # _pred_out — exact recorded path first, then scan for residuals
        if pred_out_dir and Path(pred_out_dir).exists():
            shutil.rmtree(pred_out_dir, ignore_errors=True)
        pred_out_root = root / "_pred_out"
        if pred_out_root.is_dir():
            for candidate in list(pred_out_root.iterdir()):
                if candidate.is_dir() and pred_name in candidate.name:
                    shutil.rmtree(candidate, ignore_errors=True)

        # splits
        splits_dir = root / "splits"
        if splits_dir.is_dir():
            sf = splits_dir / f"custom_{pred_name}.txt"
            if sf.exists():
                sf.unlink(missing_ok=True)

        # configs
        configs_dir = root / "configs"
        if configs_dir.is_dir():
            for cfg in list(configs_dir.iterdir()):
                if cfg.suffix == ".yaml" and cfg.name != "default.yaml" \
                        and pred_name in cfg.name:
                    cfg.unlink(missing_ok=True)

        # local tmp frames: local_frames_dir = _tmp_parent / unique_stem
        # Delete the _tmp_parent wrapper entirely to clean everything at once
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
    ) -> Dict:
        """
        Full three-stage pipeline for one IPAD sequence directory.

        FPS resolution for Stage 2 (priority order):
          1. dynamic_fps from the VLM caption call  (if not None, already clamped 2–20)
          2. fps argument  (the CLI --fps value, may be None)
          3. None  →  quick_run.py uses its own built-in default

        The caption is forwarded verbatim to Stage 3 via the --caption CLI argument
        so prompt_vad.py can present it to the VLM under the heading "caption".
        """
        vp = str(seq_dir)
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
        report:        Dict          = {}

        # ── STAGE 1 ────────────────────────────────────────────────
        ok1 = False
        for attempt in range(MAX_RETRIES):
            try:
                with suppress_output(self.verbose):
                    (local_frames_dir, mask_path, ok1,
                     keep_indices, caption, dynamic_fps) = \
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
            return {"success": False, "seq_dir": vp, "stage": 1,
                    "error": f"Object grounding failed after {MAX_RETRIES} attempts"}

        # ── FPS resolution ─────────────────────────────────────────
        # Prefer the VLM-recommended dynamic_fps over the CLI --fps argument.
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
                            method=method)
                if ok2:
                    break
            except Exception as e:
                ok2 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 2 failed: {e}")
                time.sleep(3)

        if not ok2:
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)
            return {"success": False, "seq_dir": vp, "stage": 2,
                    "error": f"Tracking failed after {MAX_RETRIES} attempts"}

        # ── TEMPORAL ALIGNMENT ─────────────────────────────────────
        if keep_indices:
            self._remap_prediction_indices(src_pred_path, keep_indices)

        # ── STAGE 3 ────────────────────────────────────────────────
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
                        caption=caption)
                if ok3:
                    break
            except Exception as e:
                ok3 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{MAX_RETRIES}] Stage 3 failed: {e}")
                time.sleep(3)

        if not ok3:
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)
            return {"success": False, "seq_dir": vp, "stage": 3,
                    "error": f"Anomaly detection failed after {MAX_RETRIES} attempts"}

        # ── POST-PROCESSING ────────────────────────────────────────
        # Copy Stage 2 graph to NVMe stage2, then clean all local scratch.
        with suppress_output(self.verbose):
            dst_pred = _pred_nvme_dir(vp, vqa_metadata)
            if osp.exists(src_pred_path):
                dst_pred.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_pred_path, str(dst_pred), dirs_exist_ok=True)

            # Cleanup ONLY after NVMe copy is confirmed
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)

        return {
            "success":          True,
            "seq_dir":          vp,
            "prediction_name":  pred_name,
            "report_path":      str(_report_nvme_path(vp, vqa_metadata)),
            "anomaly_detected": report.get("anomaly_detected", False),
            "caption":          caption,
            "dynamic_fps":      dynamic_fps,
        }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _get_frame_count(seq_dir: str) -> int:
    return len([
        f for f in Path(seq_dir).iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])


def _report_to_frame_scores(report: Dict, total_frames: int) -> List[float]:
    """
    Convert a report's anomaly list into a per-frame confidence score array.

    Each frame's score is the MAXIMUM confidence among all anomalies whose
    [start_frame, end_frame] interval covers that frame.  Frames covered by
    no anomaly get score 0.0.  This gives AUROC a continuous ranking signal
    rather than a degenerate binary one.

    Robust to start_frame / end_frame being stored as int, float, or a list
    (in which case the first element is used).
    """
    scores = [0.0] * total_frames
    for anomaly in report.get("anomalies", []):
        raw_start = anomaly.get("start_frame", anomaly.get("start", 0))
        raw_end   = anomaly.get("end_frame",   anomaly.get("end",   0))

        # Guard: if the field was accidentally serialised as a list, take first element
        if isinstance(raw_start, list):
            raw_start = raw_start[0] if raw_start else 0
        if isinstance(raw_end, list):
            raw_end = raw_end[-1] if raw_end else 0  # last element gives widest span

        start      = int(raw_start)
        end        = int(raw_end)
        confidence = float(anomaly.get("confidence", 0.0))

        for f in range(max(0, start), min(total_frames, end + 1)):
            if confidence > scores[f]:
                scores[f] = confidence
    return scores


def compute_auroc(y_true: List[float], y_score: List[float]) -> float:
    """
    Trapezoidal AUROC — pure Python, no sklearn required.
    Returns 0.5 for degenerate cases (all labels identical).
    """
    if not y_true:
        return 0.5
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5

    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tpr_pts, fpr_pts = [0.0], [0.0]
    tp = fp = 0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / pos)
        fpr_pts.append(fp / neg)

    auroc = 0.0
    for i in range(1, len(fpr_pts)):
        auroc += (fpr_pts[i] - fpr_pts[i - 1]) * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0
    return auroc


def print_metrics(seq_dirs: List[Path], vqa_metadata: Dict):
    """
    Collect predictions from saved NVMe reports and print all six metrics.

    Video-level (5 metrics):
      GT:    VQA entry["answer"]  in {"Normal", "Abnormal"}
      Pred:  report["anomaly_detected"]  bool  → 0 / 1
      Score: max confidence across report["anomalies"] (0.0 if list is empty)
      Metrics: Accuracy, Precision, Recall, F1, AUROC

    Frame-level (1 metric):
      GT:    VQA entry["context"]  — list of 0/1 labels, one per frame
      Score: per-frame max confidence from report["anomalies"] via
             start_frame / end_frame intervals (0.0 for uncovered frames)
      Metric: AUROC
    """
    vid_true:    List[int]   = []
    vid_pred:    List[int]   = []
    vid_score:   List[float] = []
    frame_true:  List[float] = []
    frame_score: List[float] = []

    evaluated = 0

    for vp in seq_dirs:
        rpt_file = _report_nvme_path(str(vp), vqa_metadata)
        if not rpt_file.is_file():
            continue

        try:
            with open(rpt_file) as f:
                report = json.load(f)
        except Exception:
            continue

        # VQA lookup: try both raw path and resolved abs path
        entry = (vqa_metadata.get(str(vp))
                 or vqa_metadata.get(str(vp.resolve())))
        if not entry:
            tqdm.write(f"  [WARN] No VQA entry for {vp}, skipping from metrics.")
            continue

        # ── Video-level ───────────────────────────────────────────
        gt_answer = entry.get("answer", "Normal").strip().lower()
        gt_vid    = 1 if gt_answer == "abnormal" else 0

        # Binary prediction from anomaly_detected
        pred_vid  = 1 if report.get("anomaly_detected", False) else 0

        # Continuous score: highest confidence among all detected anomalies.
        # Empty anomalies list → score 0.0 (model predicts no anomaly).
        anomalies = report.get("anomalies", [])
        if anomalies:
            vid_confidence = max(float(a.get("confidence", 0.0)) for a in anomalies)
        else:
            vid_confidence = 0.0

        vid_true.append(gt_vid)
        vid_pred.append(pred_vid)
        vid_score.append(vid_confidence)

        # ── Frame-level ───────────────────────────────────────────
        # GT: VQA "context" is a list of per-frame 0/1 labels.
        # If absent, fall back to replicating the video-level label across all frames.
        context = entry.get("context", [])
        if context:
            total_frames = len(context)
            gt_frames    = [float(v) for v in context]
        else:
            total_frames = _get_frame_count(str(vp))
            gt_frames    = [float(gt_vid)] * total_frames

        # Score: per-frame max confidence from anomaly intervals
        pred_frames = _report_to_frame_scores(report, total_frames)

        frame_true.extend(gt_frames)
        frame_score.extend(pred_frames)

        evaluated += 1

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 4 – Metrics")
    print(f"{'='*60}")
    print(f"  Evaluated : {evaluated} / {len(seq_dirs)} sequences")

    if evaluated == 0:
        print("  No reports found — cannot compute metrics.")
        return

    TP = sum(1 for g, p in zip(vid_true, vid_pred) if g == 1 and p == 1)
    TN = sum(1 for g, p in zip(vid_true, vid_pred) if g == 0 and p == 0)
    FP = sum(1 for g, p in zip(vid_true, vid_pred) if g == 0 and p == 1)
    FN = sum(1 for g, p in zip(vid_true, vid_pred) if g == 1 and p == 0)

    vid_acc   = (TP + TN) / evaluated
    vid_prec  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    vid_rec   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    vid_f1    = (2 * vid_prec * vid_rec / (vid_prec + vid_rec)
                 if (vid_prec + vid_rec) > 0 else 0.0)
    vid_auroc = compute_auroc([float(v) for v in vid_true], vid_score)
    frm_auroc = compute_auroc(frame_true, frame_score)

    print(f"\n  {'─'*42}")
    print(f"  Video-level metrics  ({evaluated} sequences)")
    print(f"  {'─'*42}")
    print(f"  Accuracy  : {vid_acc:.4f}   ({TP+TN}/{evaluated} correct)")
    print(f"  Precision : {vid_prec:.4f}   (TP={TP}, FP={FP})")
    print(f"  Recall    : {vid_rec:.4f}   (TP={TP}, FN={FN})")
    print(f"  F1-Score  : {vid_f1:.4f}")
    print(f"  AUROC     : {vid_auroc:.4f}   (ranked by max anomaly confidence)")
    print(f"  {'─'*42}")
    print(f"  Frame-level metrics  ({len(frame_true)} frames total)")
    print(f"  {'─'*42}")
    print(f"  AUROC     : {frm_auroc:.4f}   (ranked by per-frame max confidence)")
    print(f"  {'─'*42}\n")


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
                        "e.g. /path/to/IPAD_dataset/S02.  "
                        "Only image-sequence subdirs found here that also appear "
                        "in --vqa_file with the matching --split are processed.")
    p.add_argument("-c", "--config",     required=True,
                   help="TubeletGraph YAML config file.")
    p.add_argument("-o", "--output",     default="output",
                   help="Kept for CLI compat; actual outputs go to NVMe stage1/2/3.")
    p.add_argument("--vqa_file",         required=True,
                   help="Path to IPAD_VQA.jsonl.  Provides split labels, GT answers, "
                        "per-frame context arrays, and canonical NVMe path structure.")
    p.add_argument("--split",            default="test",
                   help="Value of the 'split' field in IPAD_VQA.jsonl to process "
                        "(default: test).  Must match exactly, e.g. 'test', 'train', 'mask'.")
    p.add_argument("--vlm",              default="openai",
                   choices=["openai", "claude", "ollama"])
    p.add_argument("--fps",              type=int,   default=None,
                   help="Fallback FPS for Stage 2 when the VLM-recommended FPS is "
                        "unavailable (e.g. OpenAI call failed or --vlm != openai).")
    p.add_argument("--sample_interval",  type=int,   default=10,
                   help="Frame sampling interval passed to prompt_vad.py.")
    p.add_argument("--method",           default="Ours")
    p.add_argument("--no-auto",          action="store_true",
                   help="Disable automatic object detection in Stage 1.")
    p.add_argument("-v", "--verbose",    action="store_true")
    return p


def main():
    args = get_parser().parse_args()

    framework = IPADFramework(
        config_path=args.config,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # ── Load VQA metadata ──────────────────────────────────────────────────
    # IPAD_VQA.jsonl uses ABSOLUTE paths in "video_path", e.g.:
    #   /u/qilong/.../IPAD_dataset/S06/training/frames/01
    # We index by the absolute path directly so lookups are trivial.
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
                # Primary key: exactly as written in the jsonl (may be absolute)
                vqa_metadata[vp] = data
                # Secondary key: resolved absolute path (guards against symlinks / ".." etc.)
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
            tqdm.write(f"  [SKIP] {rpt.name} already exists.")
            continue

        try:
            result = framework.process_sample(
                seq_dir=str(vp),
                vqa_metadata=vqa_metadata,
                vlm_model=args.vlm,
                fps=args.fps,
                sample_interval=args.sample_interval,
                method=args.method,
                auto_mode=not args.no_auto,
            )
            if not result["success"]:
                failed.append(vp)
                tqdm.write(f"  [ERROR] Stage {result.get('stage', '?')} – "
                           f"{vp.name}: {result.get('error', '')}")
        except Exception as e:
            failed.append(vp)
            tqdm.write(f"  [CRITICAL] {vp.name}: {e}")
            if args.verbose:
                tqdm.write(traceback.format_exc())

    if args.verbose:
        total = len(seq_dirs)
        print(f"\n{'='*60}")
        print(f"Inference done.  {total - len(failed)}/{total} succeeded.")

    # ── STAGE 4: metrics ───────────────────────────────────────────────────
    print_metrics(seq_dirs, vqa_metadata)

    if failed:
        if args.verbose:
            print(f"{len(failed)} failure(s):")
            for ff in failed:
                print(f"  {ff}")
        sys.exit(1)


if __name__ == "__main__":
    main()