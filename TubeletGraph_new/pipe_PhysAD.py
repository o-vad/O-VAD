#!/usr/bin/env python3
"""
ST-VAD Framework: Spatio-Temporal Video Anomaly Detection
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

NVMe permanent storage mirrors the VQA source path, anchored at "PhysAD":
  Stage 1 → /work/nvme/bgiv/qilong/stage1/PhysAD/ball/test/insufficient_gas/0008_mask/0000000.png
            (mask ONLY — frames are local tmp, never written to NVMe)
  Stage 2 → /work/nvme/bgiv/qilong/stage2/PhysAD/ball/test/insufficient_gas/0008/
            (graph / prediction files)
  Stage 3 → /work/nvme/bgiv/qilong/stage3/PhysAD/ball/test/insufficient_gas/0008_report.json

Note: the on-disk dataset directory is "Phys-AD" (with dash), but NVMe paths use
"PhysAD" (no dash) for a cleaner layout.

Naming convention for local scratch dirs (globally unique across all categories):
  unique_stem = parts after "Phys-AD" joined by "_", file stem without extension
  e.g. .../Phys-AD/ball/test/insufficient_gas/0008.mp4  →  "ball_test_insufficient_gas_0008"

Local TubeletGraph scratch (_custom_dataset, _interm_out, _pred_out) are cleaned
per-video immediately after NVMe data is safely stored.

Metrics (Stage 4, printed after all inference):
  Video-level (5) : Accuracy, Precision, Recall, F1, AUROC
                    (report["anomaly_detected"] vs VQA["answer"])
  BERT score      : cosine similarity between VQA["context"] and
                    report["anomalies"][*]["anomaly_type"] + ["anomaly_subtype"]
                    using google-bert/bert-base-uncased CLS-token embeddings.

Usage:
    python pipe.py analyze <video_dir> -c <config> [options]
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

# Max frames sent to the VLM for caption + FPS selection (limits cost / tokens)
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


def _unique_stem(video_path: str) -> str:
    """
    Globally-unique local scratch identifier for one PhysAD video.
    Anchored at the "Phys-AD" directory so the name is unique across ALL
    categories (ball, sticky_roller, …) even when processing concurrently.

    Examples:
      .../Phys-AD/ball/test/insufficient_gas/0001.mp4  →  "ball_test_insufficient_gas_0001"
      .../Phys-AD/sticky_roller/train/0000.mp4          →  "sticky_roller_train_0000"
    """
    parts = Path(video_path).resolve().parts
    if "Phys-AD" in parts:
        idx = parts.index("Phys-AD")
        rel_parts = parts[idx + 1:]
        if rel_parts:
            dirs = rel_parts[:-1]
            stem = Path(rel_parts[-1]).stem
            return "_".join(list(dirs) + [stem])
    return Path(video_path).stem


def _nvme_rel(video_path: str, vqa_metadata: Dict) -> Tuple[Path, str]:
    """
    Returns (parent_dir, stem) for NVMe paths — always RELATIVE.
    Anchors at the "Phys-AD" marker and substitutes "PhysAD" (no dash).
    """
    for key in (str(video_path), str(Path(video_path).resolve())):
        entry = vqa_metadata.get(key)
        if entry:
            parts = Path(entry["video_path"]).parts
            if "Phys-AD" in parts:
                idx = parts.index("Phys-AD")
                rel_parts = ("PhysAD",) + parts[idx + 1:]
                p = Path(*rel_parts)
                return p.parent, p.stem
            p = Path(entry["video_path"])
            if p.is_absolute():
                stripped = p.parts[1:]
                p = Path(*stripped) if stripped else Path(".")
            return p.parent, p.stem

    parts = Path(video_path).resolve().parts
    if "Phys-AD" in parts:
        idx = parts.index("Phys-AD")
        rel_parts = ("PhysAD",) + parts[idx + 1:]
        p = Path(*rel_parts)
        return p.parent, p.stem
    return Path(Path(video_path).parent.name), Path(video_path).stem


def _mask_nvme_dir(video_path: str, vqa_metadata: Dict) -> Path:
    parent, stem = _nvme_rel(video_path, vqa_metadata)
    return STAGE1_ROOT / parent / f"{stem}_mask"


def _pred_nvme_dir(video_path: str, vqa_metadata: Dict) -> Path:
    parent, stem = _nvme_rel(video_path, vqa_metadata)
    return STAGE2_ROOT / parent / stem


def _report_nvme_path(video_path: str, vqa_metadata: Dict) -> Path:
    parent, stem = _nvme_rel(video_path, vqa_metadata)
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

      (b) A recommended FPS integer (2–10) for Stage 2 TubeletGraph graph construction.
          Higher FPS for fast/dynamic scenes; lower for slow/static scenes.

    Parameters
    ----------
    frames_dir : str
        Directory containing the thinned/sampled frames (output of Stage 1 frame extraction).
        Frames are named 0000000.jpg, 0000001.jpg, … in chronological order.
    objects : list[str]
        Object names/descriptions detected by the VLM in Stage 1
        (e.g. ["basketball", "gripper"]).
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
        VLM-recommended FPS clamped to [2,10].
        None when parsing fails or the call is skipped (caller uses --fps fallback).

    VLM response format (STRICTLY enforced in the prompt):
        <CAPTION>
        [free-form temporal caption]
        </CAPTION>
        <FPS>
        [integer 2–10]
        </FPS>

    Parsing
    -------
    Caption: extracted with  re.search(r'<CAPTION>(.*?)</CAPTION>', response, re.DOTALL)
    FPS    : extracted with  re.search(r'<FPS>\\s*(\\d+)\\s*</FPS>', response)
             then clamped:   max(2, min(10, fps_raw))
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
        "   - Use description tune (firstly, then, after that, ...) and do NOT use detailed frame id."
        "   - Focus on the detected objects and how they interact or change state over time.\n"
        "   - For EACH object, explicitly state at every temporal stage whether it is\n"
        "     MOVING or STATIC, e.g. 'Object A continues to move forward, but Object B\n"
        "     stops moving and remains still from this point onward.'\n"
        "   - Describe any deformation, contact, separation, acceleration, or other\n"
        "     physical state changes in chronological order.\n"
        "   - Reference approximate frame ranges where relevant.\n\n"
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

    # ── Parse <CAPTION> ────────────────────────────────────────────────────
    caption_match = re.search(r"<CAPTION>(.*?)</CAPTION>", raw_response, re.DOTALL)
    caption       = caption_match.group(1).strip() if caption_match else ""

    # ── Parse <FPS> and clamp to [2, 10] ──────────────────────────────────
    fps_match   = re.search(r"<FPS>\s*(\d+)\s*</FPS>", raw_response)
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

class STVADFramework:
    def __init__(self, config_path: str, input_root: str,
                 output_dir: str = "output", verbose: bool = False):
        self.config_path = config_path
        self.input_root  = input_root
        self.output_dir  = output_dir
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
    ) -> Tuple[str, str, bool, List[int], str, Optional[int]]:
        """
        Returns:
          (local_frames_dir, nvme_mask_path, success, keep_indices,
           caption, dynamic_fps)

        Frames → local /tmp dir only (NOT NVMe; large and transient).
        Mask   → NVMe stage1 (small, permanent).

        All frames (full original fps) are copied into local_frames_dir and
        passed unchanged to both Stage 2 and Stage 3.  Stage 2 handles its
        own temporal subsampling internally via the --fps argument.

        NEW additions (caption + dynamic_fps)
        --------------------------------------
        After object detection, generate_caption_and_fps() is called to produce
        a temporal caption and a recommended FPS for Stage 2.
        On any failure caption="" and dynamic_fps=None are returned.
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 1 – Object Grounding: {Path(video_path).name}")

        mod       = self._load_stage1_module()
        segmenter = self._get_segmenter()

        stem             = _unique_stem(video_path)
        _tmp_parent      = Path(tempfile.mkdtemp(prefix="stvad_s1_"))
        local_frames_dir = _tmp_parent / stem
        local_frames_dir.mkdir(parents=True, exist_ok=True)

        nvme_mask_dir = _mask_nvme_dir(video_path, vqa_metadata)
        tmp_extract   = Path(tempfile.mkdtemp(prefix=f"stvad_extract_{stem}_"))
        is_dir        = Path(video_path).is_dir()
        keep_indices: List[int] = []

        # Initialise new return values
        caption:     str           = ""
        dynamic_fps: Optional[int] = None

        try:
            import numpy as np

            local_frames_dir.mkdir(parents=True, exist_ok=True)

            # ── Extract / hard-link frames into local_frames_dir ──────────
            if is_dir:
                frame_files = sorted([
                    f for f in Path(video_path).iterdir()
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                ])
                if not frame_files:
                    return str(local_frames_dir), "", False, [], "", None

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

            # ── Object detection ──────────────────────────────────────────
            input_proc_thinned = mod.InputProcessor(str(local_frames_dir), None, frame_index=0)
            first_frame_path, _ = input_proc_thinned.process()

            detector = mod.VLMObjectDetector(vlm_model)
            objects  = detector.detect_objects(first_frame_path)
            if not objects:
                return str(local_frames_dir), "", False, [], "", None

            # ── Caption + FPS generation ──────────────────────────────────
            # Called immediately after object detection and BEFORE the SAM3
            # segmentation, so the VLM sees the original unmasked frames and
            # the full temporal sequence already available in local_frames_dir.
            #
            # generate_caption_and_fps() is NOT wrapped in suppress_output so
            # that its verbose prints (query status, FPS choice, caption preview)
            # remain visible regardless of the outer suppress_output context.
            caption, dynamic_fps = generate_caption_and_fps(
                frames_dir=str(local_frames_dir),
                objects=objects,
                vlm_model=vlm_model,
                verbose=self.verbose,
            )

            # ── SAM3 segmentation ─────────────────────────────────────────
            if scan_frames and input_proc_thinned.num_frames > 1:
                _, mask, _ = mod.scan_frames_for_objects(
                    input_proc_thinned, objects, segmenter,
                    num_frames_to_scan=num_scan_frames, threshold=threshold)
            else:
                mask, _ = segmenter.segment(
                    first_frame_path, objects, threshold=threshold,
                    retry_with_variations=True, min_threshold=0.1)

            if not np.any(mask > 0):
                # Segmentation failed but we still have valid caption/fps —
                # return them so the caller can decide what to do.
                return str(local_frames_dir), "", False, [], caption, dynamic_fps

            # ── Save mask to NVMe stage1 ──────────────────────────────────
            nvme_mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = nvme_mask_dir / "0000000.png"
            mod.save_vos_mask(mask, str(mask_path))
            vis_path = nvme_mask_dir / "0000000_vis.png"
            if vis_path.exists():
                vis_path.unlink(missing_ok=True)

            # ── Save detected obj to NVMe stage1 ──────────────────────────────────
            obj_path = nvme_mask_dir / "obj.json"
            with open(obj_path, "w") as f:
                json.dump(objects, f, indent=4)

            return str(local_frames_dir), str(mask_path), True, keep_indices, caption, dynamic_fps

        except Exception:
            if self.verbose:
                traceback.print_exc()
            return str(local_frames_dir), "", False, [], caption, dynamic_fps
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
        method: str = "Ours",
    ) -> Tuple[str, str, str, str, bool]:
        """
        Returns (pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok).

        pred_name == Path(frames_dir).name == unique_stem.
        After quick_run.py finishes, _pred_out/ is scanned for the actual output
        directory whose name contains pred_name (quick_run.py names it
        custom-{stem}-{method}_{vlm_model}). This is used as src_pred_path so
        the rest of the pipeline finds the correct JSON files.
        """
        if self.verbose:
            fps_display = str(fps) if fps is not None else "default"
            print(f"\n{'='*60}\nSTAGE 2 – Tracking: {Path(video_path).name}  "
                  f"(fps={fps_display})")

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
        """
        Runs prompt_vad.py for anomaly detection.

        NEW parameter: caption (str)
        ----------------------------
        When non-empty, the caption produced by generate_caption_and_fps() in
        Stage 1 is forwarded to prompt_vad.py via the --caption argument.

        prompt_vad.py is expected to include the caption in its VLM prompt under
        the heading "caption", giving the model additional context about the
        observed object motion before reasoning over the state graph.

        If the caption is empty (e.g. OpenAI call failed, or vlm_model != "openai"),
        the --caption argument is simply omitted and prompt_vad.py behaves as before.
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 3 – Anomaly Detection: {Path(video_path).name}")
            if caption:
                preview = caption[:100].replace("\n", " ")
                print(f"  Caption: '{preview}{'…' if len(caption) > 100 else ''}'")
            else:
                print("  Caption: (none)")

        rpt_path = _report_nvme_path(video_path, vqa_metadata)
        rpt_path.parent.mkdir(parents=True, exist_ok=True)

        _, stem = _nvme_rel(video_path, vqa_metadata)

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

        # ── Forward caption to prompt_vad.py (NEW) ────────────────────────
        # prompt_vad.py receives the caption as a CLI argument so it can
        # incorporate it into the VLM prompt under the heading "caption".
        # When caption is empty we skip the argument entirely.
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
    # Local scratch cleanup
    # ------------------------------------------------------------------

    def _cleanup_local_scratch(self, pred_name: str, local_frames_dir: str,
                               jpeg_images_dir: str = "", pred_out_dir: str = ""):
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
        """
        Full three-stage pipeline for one video.

        Stage 1 now returns two additional values:
          caption     (str)       – temporal caption from generate_caption_and_fps()
          dynamic_fps (int|None)  – VLM-recommended FPS for Stage 2

        FPS resolution for Stage 2 (priority order):
          1. dynamic_fps from the VLM caption call  (if not None, already clamped 2-10)
          2. fps argument  (the CLI --fps value, may be None)
          3. None  →  quick_run.py uses its own built-in default

        The caption is forwarded verbatim to Stage 3 via the --caption CLI argument
        so prompt_vad.py can present it to the VLM under the heading "caption".
        """
        vp = str(video_path)
        MAX_RETRIES = 100

        local_frames_dir = ""
        mask_path        = ""
        pred_name        = ""
        src_pred_path    = ""
        jpeg_images_dir  = ""
        pred_out_dir     = ""
        keep_indices:  List[int]    = []
        caption:       str          = ""
        dynamic_fps:   Optional[int] = None
        report:        Dict         = {}

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
            return {"success": False, "video_path": vp, "stage": 1,
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
                            fps=effective_fps, method=method)
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
                    report, ok3 = self.stage3_anomaly_detection(
                        vp, pred_name, src_pred_path,
                        vqa_metadata=vqa_metadata,
                        frames_dir=local_frames_dir,
                        sample_interval=sample_interval,
                        vlm_model=vlm_model,
                        caption=caption)           # ← forwarded from Stage 1
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
            return {"success": False, "video_path": vp, "stage": 3,
                    "error": f"Anomaly detection failed after {MAX_RETRIES} attempts"}

        # ── POST-PROCESSING ────────────────────────────────────────
        # Copy Stage 2 graph to NVMe stage2, then clean all local scratch.
        with suppress_output(self.verbose):
            dst_pred = _pred_nvme_dir(vp, vqa_metadata)
            if osp.exists(src_pred_path):
                dst_pred.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_pred_path, str(dst_pred), dirs_exist_ok=True)
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)

        return {
            "success":          True,
            "video_path":       vp,
            "prediction_name":  pred_name,
            "report_path":      str(_report_nvme_path(vp, vqa_metadata)),
            "anomaly_detected": report.get("anomaly_detected", False),
            "caption":          caption,
            "dynamic_fps":      dynamic_fps,
        }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_auroc(y_true: List[float], y_score: List[float]) -> float:
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
        if label: tp += 1
        else:     fp += 1
        tpr_pts.append(tp / pos)
        fpr_pts.append(fp / neg)
    auroc = 0.0
    for i in range(1, len(fpr_pts)):
        auroc += (fpr_pts[i] - fpr_pts[i - 1]) * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0
    return auroc


def _load_bert():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        model_name = "google-bert/bert-base-uncased"
        print(f"  Loading BERT model ({model_name}) …", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"loaded on {device}.")
        return tokenizer, model, device
    except ImportError:
        print("  WARNING: 'transformers' or 'torch' not installed — BERT score skipped.")
        return None, None, None
    except Exception as e:
        print(f"  WARNING: Failed to load BERT model: {e} — BERT score skipped.")
        return None, None, None


def _bert_embed(texts: List[str], tokenizer, model, device):
    import torch
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()


def _cosine_sim(a, b) -> float:
    a = a / (a.norm() + 1e-9)
    b = b / (b.norm() + 1e-9)
    import torch
    return float(torch.dot(a, b).item())


def compute_bert_score(video_files: List[Path], vqa_metadata: Dict) -> Optional[float]:
    qualifying: List[Tuple[str, List[str]]] = []
    for vp in video_files:
        rpt_file = _report_nvme_path(str(vp), vqa_metadata)
        if not rpt_file.is_file():
            continue
        try:
            with open(rpt_file) as f:
                report = json.load(f)
        except Exception:
            continue
        entry = vqa_metadata.get(str(vp)) or vqa_metadata.get(str(vp.resolve()))
        if not entry:
            continue
        if entry.get("answer", "Normal").strip().lower() != "abnormal":
            continue
        ctx = entry.get("context", "").strip()
        if not ctx or ctx.lower().endswith("_normal") or ctx.lower() == "normal":
            continue
        anomalies = report.get("anomalies", [])
        if not anomalies:
            continue
        candidate_texts: List[str] = []
        for a in anomalies:
            atype    = str(a.get("anomaly_type",    "")).replace("_", " ").strip()
            asubtype = str(a.get("anomaly_subtype", "")).replace("_", " ").strip()
            text     = f"{atype} {asubtype}".strip() if asubtype else atype
            if text:
                candidate_texts.append(text)
        ref_text = ctx.replace("_", " ").strip()
        if candidate_texts and ref_text:
            qualifying.append((ref_text, candidate_texts))

    if not qualifying:
        return None
    tokenizer, model, device = _load_bert()
    if tokenizer is None:
        return None

    all_texts = []
    for ref, cands in qualifying:
        all_texts.append(ref); all_texts.extend(cands)
    unique_texts = list(dict.fromkeys(all_texts))
    print(f"  BERT: embedding {len(unique_texts)} unique text(s) …", end=" ", flush=True)
    import torch
    parts = []
    for i in range(0, len(unique_texts), 64):
        parts.append(_bert_embed(unique_texts[i:i+64], tokenizer, model, device))
    all_embeds = torch.cat(parts, dim=0)
    text2emb   = {t: all_embeds[i] for i, t in enumerate(unique_texts)}
    print("done.")

    scores: List[float] = []
    for ref, cands in qualifying:
        best = max(_cosine_sim(text2emb[ref], text2emb[c]) for c in cands)
        scores.append(best)
    return float(sum(scores) / len(scores))


def print_metrics(video_files: List[Path], vqa_metadata: Dict):
    vid_true:  List[int]   = []
    vid_pred:  List[int]   = []
    vid_score: List[float] = []
    evaluated = 0

    for vp in video_files:
        rpt_file = _report_nvme_path(str(vp), vqa_metadata)
        if not rpt_file.is_file():
            continue
        try:
            with open(rpt_file) as f:
                data = json.load(f)
        except Exception:
            continue

        entry = vqa_metadata.get(str(vp)) or vqa_metadata.get(str(vp.resolve()))
        if entry:
            gt = 1 if entry.get("answer", "Normal").strip().lower() == "abnormal" else 0
        else:
            gt = 0 if "normal" in str(vp).lower() else 1

        # Binary prediction from anomaly_detected
        pred = 1 if data.get("anomaly_detected", False) else 0

        # AUROC score: highest confidence among all detected anomalies.
        # If the anomalies list is empty the model predicts no anomaly → score 0.0.
        anomalies = data.get("anomalies", [])
        if anomalies:
            confidence = max(
                float(a.get("confidence", 0.0)) for a in anomalies
            )
        else:
            confidence = 0.0

        vid_true.append(gt)
        vid_pred.append(pred)
        vid_score.append(confidence)
        evaluated += 1

    print(f"\n{'='*60}")
    print("STAGE 4 – Metrics")
    print(f"{'='*60}")
    print(f"  Evaluated : {evaluated} / {len(video_files)} videos")
    if evaluated == 0:
        print("  No reports found — cannot compute metrics.")
        return

    TP = sum(1 for g, p in zip(vid_true, vid_pred) if g == 1 and p == 1)
    TN = sum(1 for g, p in zip(vid_true, vid_pred) if g == 0 and p == 0)
    FP = sum(1 for g, p in zip(vid_true, vid_pred) if g == 0 and p == 1)
    FN = sum(1 for g, p in zip(vid_true, vid_pred) if g == 1 and p == 0)

    acc   = (TP + TN) / evaluated
    prec  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auroc = compute_auroc([float(v) for v in vid_true], vid_score)

    print(f"\n  {'─'*42}")
    print(f"  Video-level metrics  ({evaluated} videos)")
    print(f"  {'─'*42}")
    print(f"  Accuracy  : {acc:.4f}   ({TP+TN}/{evaluated} correct)")
    print(f"  Precision : {prec:.4f}   (TP={TP}, FP={FP})")
    print(f"  Recall    : {rec:.4f}   (TP={TP}, FN={FN})")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUROC     : {auroc:.4f}   (ranked by max anomaly confidence)")
    print(f"  {'─'*42}")
    print(f"\n  Computing BERT anomaly-type score …")
    bert = compute_bert_score(video_files, vqa_metadata)
    print(f"  {'─'*42}")
    print(f"  BERT anomaly-type score")
    print(f"  (VQA context ↔ predicted anomaly_type + subtype)")
    print(f"  {'─'*42}")
    if bert is not None:
        print(f"  BERT Score : {bert:.4f}")
    else:
        print(f"  BERT Score : N/A  (no qualifying samples, or BERT unavailable)")
    print(f"  {'─'*42}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ST-VAD: Spatio-Temporal Video Anomaly Detection (3-stage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", choices=["analyze"])
    p.add_argument("video",
                   help="Root directory; all .mp4 files found recursively are processed "
                        "(filtered by --split and --vqa_file).")
    p.add_argument("-c", "--config",     required=True)
    p.add_argument("-o", "--output",     default="output",
                   help="Kept for CLI compat; actual outputs go to NVMe stage1/2/3.")
    p.add_argument("--vlm",              default="openai",
                   choices=["openai", "claude", "ollama"])
    p.add_argument("--fps",              type=int,   default=None,
                   help="Fallback FPS for Stage 2 when the VLM-recommended FPS is "
                        "unavailable (e.g. OpenAI call failed or --vlm != openai).")
    p.add_argument("--sample_interval",  type=int,   default=10)
    p.add_argument("--method",           default="Ours")
    p.add_argument("--split",            type=str,   default="test")
    p.add_argument("--vqa_file",         type=str,   default=None)
    p.add_argument("--no-auto",          action="store_true")
    p.add_argument("-v", "--verbose",    action="store_true")
    return p


def main():
    args = get_parser().parse_args()

    framework = STVADFramework(
        config_path=args.config,
        input_root=args.video,
        output_dir=args.output,
        verbose=args.verbose,
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
                    vqa_metadata[str(Path(vp).resolve())] = data
        except Exception as e:
            print(f"Error reading VQA file: {e}")
            sys.exit(1)
        if args.verbose:
            print(f"Loaded {len(vqa_metadata)//2} VQA entries.")

    video_root = Path(args.video).resolve()
    if not video_root.is_dir():
        print(f"Error: video directory does not exist: {video_root}")
        sys.exit(1)

    video_files: List[Path] = []
    seen: set = set()

    for candidate in sorted(video_root.rglob("*.mp4")):
        if str(candidate) in seen:
            continue
        if vqa_metadata:
            entry = (vqa_metadata.get(str(candidate))
                     or vqa_metadata.get(str(candidate.resolve())))
            if entry is None:
                continue
            org_split = entry.get("org_split", "").lower()
            in_path   = args.split.lower() in [part.lower() for part in candidate.parts]
            if org_split != args.split.lower() and not in_path:
                continue
        else:
            if args.split and args.split.lower() not in \
                    [part.lower() for part in candidate.parts]:
                continue
        video_files.append(candidate)
        seen.add(str(candidate))

    if args.verbose:
        print(f"Found {len(video_files)} video(s) matching split='{args.split}'.")

    if not video_files:
        print(f"No videos found under '{args.video}' matching split='{args.split}'.")
        sys.exit(0)

    framework.preload_sam3()
    failed: List[Path] = []

    for i in trange(len(video_files), desc="Processing Videos", dynamic_ncols=True):
        vp = video_files[i]

        rpt = _report_nvme_path(str(vp), vqa_metadata)
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

    if args.verbose:
        total = len(video_files)
        print(f"\n{'='*60}")
        print(f"Done. {total - len(failed)}/{total} succeeded.")

    print_metrics(video_files, vqa_metadata)

    if failed:
        if args.verbose:
            print(f"{len(failed)} failure(s):")
            for ff in failed:
                print(f"  {ff}")
        sys.exit(1)


if __name__ == "__main__":
    main()