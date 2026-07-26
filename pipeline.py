#!/usr/bin/env python3
"""
O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning.

Unified three-stage pipeline driver for all supported benchmarks.  One driver
serves every dataset; select it with --dataset.

  Stage 1 – Object Grounding   : VLM detects objects; shared SAM3 segments them.
             After object detection, a second VLM call produces:
               • A temporal caption describing object interactions / motion.
               • A recommended FPS (int, 2–10) for Stage 2 graph construction.
  Stage 2 – Object Tracking    : quick_run.py / TubeletGraph builds tubelets and
             the object state graph, using the VLM-recommended FPS (falling back
             to --fps when unavailable).
  Stage 3 – Anomaly Detection  : prompt_vad.py reasons over the state graph.
             Receives the Stage 1 caption via --caption so the VLM can
             cross-reference observed motion against graph-derived state changes.

Everything dataset-specific lives in a DatasetAdapter subclass: on-disk layout,
NVMe path anchoring, sample discovery, grounding prompts and Stage 3 extras.
The three stages themselves are shared.

Supported datasets
------------------
  physad   Phys-AD.  Samples are .mp4 files.
           NVMe layout anchors at "Phys-AD" on disk and "PhysAD" under STAGE*_ROOT:
             stage1/PhysAD/ball/test/insufficient_gas/0008_mask/0000000.png
             stage2/PhysAD/ball/test/insufficient_gas/0008/
             stage3/PhysAD/ball/test/insufficient_gas/0008_report.json

  ipad     IPAD.  Samples are DIRECTORIES of frames, not video files:
             .../IPAD_dataset/S02/testing/frames/01/  →  0000000.jpg, ...
           A known-normal reference sequence is resolved automatically from the
           dataset layout (.../IPAD_dataset/<subset>/training/frames/01) and
           forwarded to Stage 3 via --normal_video_path.
           NVMe layout mirrors the source path from the "IPAD" component down.

  autolab  AutoLab / LiquidAD.  Samples are .mp4 files and VQA video_path values
           are RELATIVE (e.g. "train/abnormal_16/000.mp4"), so --dataset_root is
           used to resolve them.  Grounding uses fixed pipette object prompts and
           bbox-cropped multi-instance segmentation.
           NVMe layout: stage{1,2,3}/AutoLab/<split>/<subset>/<stem>...

Storage roots (STAGE1_ROOT / STAGE2_ROOT / STAGE3_ROOT) must be edited below to
point at your own fast storage.  Local TubeletGraph scratch (_custom_dataset,
_interm_out, _pred_out) is cleaned per sample once stage data is safely stored;
pass --keep-scratch to retain it for debugging.

Samples whose report already exists are skipped, so an interrupted run resumes
where it stopped.  To spread a run across jobs or GPUs, give each job a disjoint
--sample_id_s / --sample_id_e range over the same storage roots.

Parameter settings used for the reported results
------------------------------------------------
  --fps 3               Fallback FPS for Stage 2. The Stage 1 caption call
                        recommends an FPS in [2, 10] and that takes precedence;
                        this value is used only when that call is unavailable.
  --sample_interval     Frame stride for state-change detection:
                          60  Phys-AD and AutoLab
                          25  IPAD
                        Lower means more comparisons per clip: finer temporal
                        resolution, proportionally more VLM calls.
  --split test          Matched against 'org_split' in the VQA file.
  --method Ours         Method key from the config. Ours_abl_prox / _sem /
                        _both disable the proximity and semantic constraints.

Run `python pipeline.py --help` for the full flag reference.

Usage:
    python pipeline.py analyze <ROOT> --dataset <NAME> -c <CONFIG> [options]

    python pipeline.py analyze /data/Phys-AD/ball --dataset physad \\
        -c configs/default.yaml --vqa_file /data/vqa/PhysAD_VQA.jsonl \\
        --vlm openai --fps 3 --sample_interval 60 --split test

    python pipeline.py analyze /data/IPAD/IPAD_dataset/S04 --dataset ipad \\
        -c configs/default.yaml --vqa_file /data/vqa/IPAD_VQA.jsonl \\
        --vlm openai --fps 3 --sample_interval 25 --split test

    python pipeline.py analyze /data/AutoLab/test/abnormal_1 --dataset autolab \\
        -c configs/default.yaml --vqa_file /data/AutoLab/autolab.jsonl \\
        --dataset_root /data/AutoLab \\
        --vlm openai --fps 3 --sample_interval 60 --split test
"""

import argparse
import base64
import importlib.util
import inspect
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm, trange

# ---------------------------------------------------------------------------
# Global storage roots — update to match your own fast/NVMe mount point
# ---------------------------------------------------------------------------
STAGE1_ROOT = Path("/work/nvme/bgiv/username/stage1")
STAGE2_ROOT = Path("/work/nvme/bgiv/username/stage2")
STAGE3_ROOT = Path("/work/nvme/bgiv/username/stage3")

# Max frames sent to the VLM for caption + FPS selection (limits cost / tokens)
CAPTION_MAX_FRAMES = 8

# Model used for the Stage 1 caption call when talking to the OpenAI API.
# When a base_url is configured the served model is used instead.
CAPTION_MODEL_DEFAULT = "gpt-4o"

# Default retries per stage before a sample is declared failed (see --max-retries).
# Each failed attempt sleeps 3s, so a persistently broken environment takes
# ~5 minutes per sample to give up at the default setting.
MAX_RETRIES = 100

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


def _supported_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop None values, then keep only the keys `fn` actually accepts.

    The annotate/ grounding helpers have evolved through several signatures
    (e.g. scan_frames_for_objects gained a required `detector` argument and the
    bbox-crop options).  Filtering here lets one call site drive whichever
    version of the module is installed, and lets an adapter opt out of an
    argument simply by leaving it None.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


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


# ---------------------------------------------------------------------------
# OpenAI Responses API client  (used only for caption + FPS generation)
# ---------------------------------------------------------------------------

class _OpenAIResponsesClient:
    """
    OpenAI-protocol client for the Stage 1 caption + FPS call.

    base_url is None -> the OpenAI API via the Responses endpoint (the path that
                        produced all reported results; unchanged).
    base_url is set  -> an OpenAI-compatible server such as a local vLLM, via
                        chat/completions, which is the endpoint those implement.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o",
                 base_url: Optional[str] = None):
        if not HAS_OPENAI:
            raise ImportError(
                "'openai' package not installed — cannot run caption generation. "
                "Install it with: pip install openai"
            )
        self.base_url = base_url or None
        self.api_key  = api_key or os.environ.get("OPENAI_API_KEY")
        if self.base_url and not self.api_key:
            # local servers ignore the token, but the SDK requires a non-empty one
            self.api_key = "EMPTY"
        self.model = model
        kwargs: Dict[str, Any] = {"api_key": self.api_key,
                                  "timeout": 120.0, "max_retries": 2}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = _openai_mod.OpenAI(**kwargs)

    def _data_uri(self, image_path: str) -> str:
        ext        = Path(image_path).suffix.lower().lstrip(".")
        media_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
        }.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{media_type};base64,{data}"

    def _encode_image(self, image_path: str) -> Dict:
        """Encode a local image file as a base64 input_image block."""
        return {"type": "input_image", "image_url": self._data_uri(image_path)}

    def query(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.base_url:
            content: List[Dict] = []
            if image_paths:
                for p in image_paths:
                    content.append({"type": "image_url",
                                    "image_url": {"url": self._data_uri(p)}})
            content.append({"type": "text", "text": prompt})

            messages: List[Dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            response = _create_with_param_fallback(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        content = []
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

        response = _create_with_param_fallback(self.client.responses.create, **kwargs)
        return response.output_text


# ---------------------------------------------------------------------------
# Caption + FPS generation  (called inside Stage 1, after object detection)
# ---------------------------------------------------------------------------

CAPTION_STYLE_GENERIC = "generic"   # Phys-AD / IPAD
CAPTION_STYLE_LAB     = "lab"       # AutoLab lab-automation phrasing


def _select_caption_frames(frames_dir: str, max_frames: int = CAPTION_MAX_FRAMES) -> List[str]:
    """Return up to `max_frames` evenly-spaced frame file paths from `frames_dir`."""
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


def _caption_prompts(n_frames: int, objects_str: str, style: str,
                     hint_str: str = "") -> Tuple[str, str]:
    """Build the (system, user) caption prompt pair for the requested style."""
    if style == CAPTION_STYLE_LAB:
        system_prompt = (
            "You are an expert video analyst specialising in physical anomaly detection "
            "in lab automation procedures. "
            "You receive a sequence of frames sampled uniformly from a video and a list "
            "of detected objects. Your job is to describe the video content with temporal "
            "precision and to recommend an appropriate processing frame rate."
        )
        user_prompt = (
            f"I am providing {n_frames} evenly-spaced frames from a lab automation video.\n"
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
    else:
        system_prompt = (
            "You are an expert video analyst specialising in physical anomaly detection. "
            "You receive a sequence of frames sampled uniformly from a video and a list "
            "of detected objects. Your job is to describe the video content with temporal "
            "precision and to recommend an appropriate processing frame rate."
        )
        user_prompt = (
            f"I am providing {n_frames} evenly-spaced frames from a video sequence.\n"
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

    if hint_str:
        user_prompt += f"\n\nAdditional hint:\n{hint_str}"

    return system_prompt, user_prompt


def generate_caption_and_fps(
    frames_dir: str,
    objects: List[str],
    vlm_model: str = "openai",
    verbose: bool = False,
    hint_str: str = "",
    style: str = CAPTION_STYLE_GENERIC,
    endpoint: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> Tuple[str, Optional[int]]:
    """
    Send a sample of frames from `frames_dir` to a VLM and ask for:

      (a) A temporal caption describing object interactions and motion.
      (b) A recommended FPS integer (2–10) for Stage 2 graph construction.

    Both backbones speak the OpenAI protocol, so this is one call either way —
    `endpoint` carries (base_url, model) and decides where it lands.

    Returns (caption, dynamic_fps); on failure returns ("", None) so the caller
    falls back to --fps.
    """
    if vlm_model not in ("openai", "qwen"):
        if verbose:
            print(f"  [Caption] Skipped — vlm_model='{vlm_model}' (supported: 'openai', 'qwen').")
        return "", None

    frame_paths = _select_caption_frames(frames_dir, max_frames=CAPTION_MAX_FRAMES)
    if not frame_paths:
        if verbose:
            print("  [Caption] No frames found in frames_dir — skipping.")
        return "", None

    objects_str = ", ".join(str(o) for o in objects) if objects else "the objects in the scene"
    system_prompt, user_prompt = _caption_prompts(
        len(frame_paths), objects_str, style, hint_str)

    base_url, model = (endpoint or (None, None))
    raw_response = ""
    try:
        client = _OpenAIResponsesClient(
            model=model or CAPTION_MODEL_DEFAULT, base_url=base_url)
        if verbose:
            where = base_url or "the OpenAI API"
            print(f"  [Caption] Querying {client.model} via {where} with "
                  f"{len(frame_paths)} frame(s) for caption + FPS …")
        raw_response = client.query(
            prompt=user_prompt,
            image_paths=frame_paths,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception as e:
        # Surface via tqdm.write: this runs inside suppress_output(), which dup2s
        # stdout/stderr to /dev/null, so a plain print would vanish. A silent
        # failure here yields a caption-less Stage 3 whose report is still
        # written, and the resume check then skips that sample forever.
        tqdm.write(f"  [Caption] FAILED ({vlm_model}): {e}")
        return "", None

    # ── Parse <CAPTION> ────────────────────────────────────────────────────
    caption_match = re.search(r"<CAPTION>(.*?)</CAPTION>", raw_response, re.DOTALL)
    caption = caption_match.group(1).strip() if caption_match else ""

    # ── Parse <FPS> and clamp to [2, 10] ──────────────────────────────────
    fps_match = re.search(r"<FPS>\s*(\d+)\s*</FPS>", raw_response)
    if fps_match:
        dynamic_fps: Optional[int] = max(2, min(10, int(fps_match.group(1))))
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
# Dataset adapters
# ---------------------------------------------------------------------------

class DatasetAdapter:
    """
    Per-dataset behaviour: on-disk layout, storage-path anchoring, sample
    discovery, grounding prompts and Stage 3 extras.

    Subclasses must set `name` and `kind` and implement `unique_stem` and
    `nvme_rel`; everything else has a sensible default.
    """

    name: str = ""
    kind: str = "video"            # "video" (.mp4 samples) | "sequence" (frame dirs)

    stage1_module: str = "vlm_mask_grounded.py"
    tmp_prefix:    str = "ovad_s1_"
    caption_style: str = CAPTION_STYLE_GENERIC

    # Stage 1 grounding knobs.  None means "do not pass — use the module default".
    fixed_objects:         Optional[List[str]] = None   # skip VLM object detection
    fixed_caption_objects: Optional[List[str]] = None   # object list shown to the caption VLM
    bbox_object_desc:      Optional[str]       = None   # enables bbox-cropped segmentation
    max_instances:         Optional[int]       = None
    bbox_padding:          Optional[float]     = None

    # Save the mask at the best scanned frame (True) or always at frame 0 (False).
    use_best_frame: bool = False

    # "inprocess" shares the loaded modules/singletons; "subprocess" isolates them.
    default_runner: str = "subprocess"

    def __init__(self, dataset_root: Optional[str] = None):
        self.dataset_root = dataset_root

    # -- storage paths ------------------------------------------------------

    def unique_stem(self, path: str) -> str:
        """Globally-unique local scratch identifier for one sample."""
        raise NotImplementedError

    def nvme_rel(self, path: str, vqa_metadata: Dict) -> Tuple[Path, str]:
        """Return (parent_dir, stem) for storage paths — always RELATIVE."""
        raise NotImplementedError

    def mask_dir(self, path: str, vqa_metadata: Dict) -> Path:
        parent, stem = self.nvme_rel(path, vqa_metadata)
        return STAGE1_ROOT / parent / f"{stem}_mask"

    def pred_dir(self, path: str, vqa_metadata: Dict) -> Path:
        parent, stem = self.nvme_rel(path, vqa_metadata)
        return STAGE2_ROOT / parent / stem

    def report_path(self, path: str, vqa_metadata: Dict) -> Path:
        parent, stem = self.nvme_rel(path, vqa_metadata)
        return STAGE3_ROOT / parent / f"{stem}_report.json"

    # -- VQA + discovery ----------------------------------------------------

    def load_vqa(self, vqa_file: str) -> Dict:
        """Index VQA entries by both their raw and resolved video_path."""
        vqa_metadata: Dict = {}
        with open(vqa_file, "r") as f:
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
        return vqa_metadata

    def lookup(self, path: Path, vqa_metadata: Dict) -> Optional[Dict]:
        return (vqa_metadata.get(str(path))
                or vqa_metadata.get(str(path.resolve())))

    def discover(self, root: Path, vqa_metadata: Dict, split: str) -> List[Path]:
        """Find samples under `root` that have a VQA entry in the requested split."""
        samples: List[Path] = []
        seen: set = set()
        for candidate in sorted(root.rglob("*.mp4")):
            if str(candidate) in seen:
                continue
            entry = self.lookup(candidate, vqa_metadata)
            if entry is None:
                continue
            if entry.get("org_split", "").lower() != split.lower():
                continue
            samples.append(candidate)
            seen.add(str(candidate))
        return samples

    # -- stage hooks --------------------------------------------------------

    def resolve_normal_reference(self, path: str) -> Optional[str]:
        """Path to a known-normal reference clip for Stage 3, if the dataset has one."""
        return None

    def grounding_objects(self, detector, first_frame_path: str) -> List[str]:
        """Object prompts for SAM3.  Datasets with a fixed inventory override the VLM."""
        if self.fixed_objects is not None:
            return list(self.fixed_objects)
        return detector.detect_objects(first_frame_path)

    def caption_objects(self, objects: List[str]) -> List[str]:
        """Object list shown to the captioning VLM (may differ from the SAM3 prompts)."""
        if self.fixed_caption_objects is not None:
            return list(self.fixed_caption_objects)
        return objects


class PhysADAdapter(DatasetAdapter):
    """
    Phys-AD: .mp4 samples laid out as <category>/<split>/<anomaly_type>/<id>.mp4.

    Storage paths anchor at the "Phys-AD" path component and substitute
    "PhysAD" (no dash) for a cleaner layout under STAGE*_ROOT.  Local scratch
    names join every component after the anchor, e.g.
      .../Phys-AD/ball/test/insufficient_gas/0008.mp4
        →  stem "ball_test_insufficient_gas_0008"
    which stays unique when several categories run concurrently.
    """

    name   = "physad"
    kind   = "video"
    anchor = "Phys-AD"
    nvme_anchor = "PhysAD"
    tmp_prefix     = "ovad_physad_s1_"
    default_runner = "inprocess"

    def unique_stem(self, path: str) -> str:
        parts = Path(path).resolve().parts
        if self.anchor in parts:
            rel = parts[parts.index(self.anchor) + 1:]
            if rel:
                return "_".join(list(rel[:-1]) + [Path(rel[-1]).stem])
        return Path(path).stem

    def _anchored(self, raw: str) -> Optional[Tuple[Path, str]]:
        parts = Path(raw).parts
        if self.anchor in parts:
            p = Path(*((self.nvme_anchor,) + parts[parts.index(self.anchor) + 1:]))
            return p.parent, p.stem
        return None

    def nvme_rel(self, path: str, vqa_metadata: Dict) -> Tuple[Path, str]:
        for key in (str(path), str(Path(path).resolve())):
            entry = vqa_metadata.get(key)
            if entry:
                hit = self._anchored(entry["video_path"])
                if hit:
                    return hit
                p = Path(entry["video_path"])
                if p.is_absolute():
                    stripped = p.parts[1:]
                    p = Path(*stripped) if stripped else Path(".")
                return p.parent, p.stem

        hit = self._anchored(str(Path(path).resolve()))
        if hit:
            return hit
        return Path(Path(path).parent.name), Path(path).stem


class IPADAdapter(DatasetAdapter):
    """
    IPAD: each sample is a DIRECTORY of frames, e.g.
      .../IPAD_dataset/S02/testing/frames/01/  →  0000000.jpg, 0000001.jpg, ...

    Storage paths mirror the source path from the "IPAD" (or "IPAD_dataset")
    component down, keeping the marker itself.  Stage 3 additionally receives a
    known-normal reference sequence resolved from the dataset layout.
    """

    name    = "ipad"
    kind    = "sequence"
    markers = ("IPAD", "IPAD_dataset")
    tmp_prefix = "ovad_ipad_s1_"

    def unique_stem(self, path: str) -> str:
        parts = Path(path).resolve().parts
        if "IPAD_dataset" in parts:
            rel = parts[parts.index("IPAD_dataset") + 1:]
            if rel:
                return "_".join(rel)
        p = Path(path).resolve()
        return f"{p.parent.name}_{p.name}"

    def _anchored(self, raw: str) -> Optional[Tuple[Path, str]]:
        parts = Path(raw).parts
        for marker in self.markers:
            if marker in parts:
                rel = Path(*parts[parts.index(marker):])
                return rel.parent, rel.name
        return None

    def nvme_rel(self, path: str, vqa_metadata: Dict) -> Tuple[Path, str]:
        for key in (str(path), str(Path(path).resolve())):
            entry = vqa_metadata.get(key)
            if entry:
                hit = self._anchored(entry["video_path"])
                if hit:
                    return hit
                p = Path(entry["video_path"])
                if p.is_absolute():
                    stripped = p.parts[1:]
                    p = Path(*stripped) if stripped else Path(".")
                return p.parent, p.name

        hit = self._anchored(str(Path(path).resolve()))
        if hit:
            return hit
        return Path(Path(path).parent.name), Path(path).name

    def discover(self, root: Path, vqa_metadata: Dict, split: str) -> List[Path]:
        """IPAD samples are directories, so look for frame dirs rather than files."""
        samples: List[Path] = []
        seen: set = set()
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_dir() or str(candidate) in seen:
                continue
            has_images = (any(candidate.glob("*.jpg")) or
                          any(candidate.glob("*.jpeg")) or
                          any(candidate.glob("*.png")))
            if not has_images:
                continue
            entry = self.lookup(candidate, vqa_metadata)
            if entry is None:
                continue
            if entry.get("org_split", "").lower() != split.lower():
                continue
            samples.append(candidate)
            seen.add(str(candidate))
        return samples

    def resolve_normal_reference(self, path: str) -> Optional[str]:
        """
        For any sequence under .../IPAD_dataset/<subset>/... the known-normal
        reference is .../IPAD_dataset/<subset>/training/frames/01.
        Returns None when that path does not exist on disk.
        """
        parts = Path(path).resolve().parts
        if "IPAD_dataset" not in parts:
            return None
        idx = parts.index("IPAD_dataset")
        if len(parts) <= idx + 1:
            return None
        normal = Path(*parts[: idx + 1]) / parts[idx + 1] / "training" / "frames" / "01"
        return str(normal) if normal.is_dir() else None


class AutoLabAdapter(DatasetAdapter):
    """
    AutoLab / LiquidAD: .mp4 samples whose VQA video_path values are RELATIVE
    (e.g. "train/abnormal_16/000.mp4"), resolved against --dataset_root.

    Grounding uses a fixed pipette object inventory rather than VLM detection,
    and segmentation runs bbox-cropped with multiple instances per prompt.
    Storage layout: stage{1,2,3}/AutoLab/<split>/<subset>/<stem>...
    """

    name   = "autolab"
    kind   = "video"
    anchor = "AutoLab"
    tmp_prefix    = "ovad_autolab_s1_"
    caption_style = CAPTION_STYLE_LAB

    fixed_objects         = ["single-channel pipette", "liquid"]
    fixed_caption_objects = ["multichannel pipette"]
    bbox_object_desc      = "multi-channel pipette"
    max_instances         = 8
    bbox_padding          = 0.1
    use_best_frame        = True

    def unique_stem(self, path: str) -> str:
        p = Path(path).resolve()
        if self.dataset_root:
            try:
                rel = p.relative_to(Path(self.dataset_root).resolve())
                return "_".join(list(rel.parent.parts) + [rel.stem])
            except ValueError:
                pass
        parts = p.parts
        if self.anchor in parts:
            rel = parts[parts.index(self.anchor) + 1:]
            if rel:
                return "_".join(list(rel[:-1]) + [Path(rel[-1]).stem])
        return p.stem

    def nvme_rel(self, path: str, vqa_metadata: Dict) -> Tuple[Path, str]:
        for key in (str(path), str(Path(path).resolve())):
            entry = vqa_metadata.get(key)
            if entry:
                p     = Path(entry["video_path"])
                parts = p.parts
                if parts and parts[0] == self.anchor:
                    p = Path(*parts[1:])
                return Path(self.anchor) / p.parent, p.stem

        parts = Path(path).resolve().parts
        if self.anchor in parts:
            p = Path(*((self.anchor,) + parts[parts.index(self.anchor) + 1:]))
            return p.parent, p.stem
        return Path(Path(path).parent.name), Path(path).stem

    def load_vqa(self, vqa_file: str) -> Dict:
        """
        Index by the raw (relative) video_path, then add absolute aliases built
        against dataset_root so lookups work with either form.
        """
        vqa_metadata: Dict = {}
        with open(vqa_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                vp = data.get("video_path", "")
                if not vp:
                    continue
                vqa_metadata[vp] = data

        if self.dataset_root:
            root  = Path(self.dataset_root).resolve()
            extra: Dict = {}
            for entry in list(vqa_metadata.values()):
                vp = entry.get("video_path", "")
                if not vp:
                    continue
                abs_path = str((root / vp).resolve())
                if abs_path not in vqa_metadata:
                    extra[abs_path] = entry
            vqa_metadata.update(extra)
        return vqa_metadata

    def lookup(self, path: Path, vqa_metadata: Dict) -> Optional[Dict]:
        entry = super().lookup(path, vqa_metadata)
        if entry is not None:
            return entry

        if self.dataset_root:
            try:
                rel = str(path.relative_to(Path(self.dataset_root).resolve()))
                entry = vqa_metadata.get(rel)
                if entry is not None:
                    return entry
            except ValueError:
                pass

        abs_str = str(path)
        for k, v in vqa_metadata.items():
            if not Path(k).is_absolute() and abs_str.endswith(k.lstrip("./")):
                return v
        return None

    def discover(self, root: Path, vqa_metadata: Dict, split: str) -> List[Path]:
        samples: List[Path] = []
        seen: set = set()
        for candidate in sorted(root.rglob("*.mp4")):
            if str(candidate) in seen:
                continue
            entry = self.lookup(candidate, vqa_metadata)
            if entry is None:
                continue
            org_split = entry.get("org_split", entry.get("split", "")).lower()
            if org_split != split.lower():
                continue
            samples.append(candidate)
            seen.add(str(candidate))
        return samples


ADAPTERS = {
    "physad":  PhysADAdapter,
    "ipad":    IPADAdapter,
    "autolab": AutoLabAdapter,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OVADPipeline:
    """Three-stage O-VAD driver, parameterised by a DatasetAdapter."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        config_path: str,
        input_root: str,
        output_dir: str = "output",
        verbose: bool = False,
        runner: Optional[str] = None,
        keep_scratch: bool = False,
        mask_frame_mode: str = "auto",
        max_retries: int = MAX_RETRIES,
        endpoint: Tuple[Optional[str], Optional[str]] = (None, None),
    ):
        self.adapter      = adapter
        self.config_path  = config_path
        self.input_root   = input_root
        self.output_dir   = output_dir      # kept for CLI compat only
        self.verbose      = verbose
        self.runner       = runner or adapter.default_runner
        self.keep_scratch = keep_scratch
        self.max_retries  = max_retries
        self.endpoint     = endpoint   # (base_url, model); (None, None) = OpenAI API
        self.base_dir     = osp.dirname(osp.abspath(__file__))

        if mask_frame_mode == "auto":
            self.use_best_frame = adapter.use_best_frame
        else:
            self.use_best_frame = (mask_frame_mode == "best")

        self.vlm_mask_script   = osp.join(self.base_dir, "annotate", adapter.stage1_module)
        self.quick_run_script  = osp.join(self.base_dir, "quick_run.py")
        self.prompt_vad_script = osp.join(self.base_dir, "TubeletGraph", "vlm", "prompt_vad.py")

        self.project_root = Path(self.quick_run_script).parent

        self._sam3_segmenter = None
        self._stage1_mod: Optional[object] = None
        self._module_cache: Dict[str, object] = {}

        self._verify_scripts()
        STAGE1_ROOT.mkdir(parents=True, exist_ok=True)
        STAGE2_ROOT.mkdir(parents=True, exist_ok=True)
        STAGE3_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Script / module plumbing
    # ------------------------------------------------------------------

    def _verify_scripts(self):
        required = {
            f"annotate/{self.adapter.stage1_module}": self.vlm_mask_script,
            "quick_run.py":                           self.quick_run_script,
            "TubeletGraph/vlm/prompt_vad.py":         self.prompt_vad_script,
        }
        missing = [f"{n}  (expected: {p})" for n, p in required.items() if not osp.isfile(p)]
        if missing:
            print("Missing required scripts:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)

    def _run_subprocess(self, script_path: str, args_list: List[str],
                        description: str, cwd: Optional[str] = None) -> bool:
        """Run a stage script as an isolated subprocess."""
        cmd = [sys.executable, script_path] + args_list
        if self.verbose:
            print(f"[CMD] {' '.join(cmd)}")
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
        result = subprocess.run(cmd, env={**os.environ}, check=False,
                                stdout=stdout, stderr=stderr, cwd=cwd)
        if result.returncode != 0:
            if self.verbose:
                print(f"[FAIL] {description} (exit {result.returncode})")
            return False
        return True

    def _run_in_process(self, script_path: str, args_list: List[str],
                        description: str, cwd: Optional[str] = None) -> bool:
        """
        Run a stage script inside this process by importing it and calling main().

        Keeps heavyweight singletons (Qwen weights, SAM3) shared across stages
        instead of paying a fresh import per sample.  sys.argv and the working
        directory are patched so the target still behaves as if run standalone.
        """
        script_dir = osp.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        mod = self._module_cache.get(script_path)
        if mod is None:
            module_name = osp.splitext(osp.basename(script_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            mod  = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                if self.verbose:
                    print(f"[FAIL] Failed to load module {script_path}: {e}")
                    traceback.print_exc()
                return False
            self._module_cache[script_path] = mod

        if not hasattr(mod, "main"):
            if self.verbose:
                print(f"[FAIL] {script_path} has no main() function to call.")
            return False

        original_argv = sys.argv[:]
        original_cwd  = os.getcwd()
        sys.argv = [script_path] + args_list
        if cwd:
            os.chdir(cwd)

        if self.verbose:
            print(f"[IN-PROCESS CMD] python3 {script_path} {' '.join(args_list)}")

        ok = True
        try:
            mod.main()
        except SystemExit as e:
            # sys.exit(0) is a normal success; anything else is a failure.
            if e.code not in (0, None):
                ok = False
                if self.verbose:
                    print(f"[FAIL] {description} exited with code {e.code}")
        except Exception as e:
            ok = False
            if self.verbose:
                print(f"[FAIL] Exception in {description}: {e}")
                traceback.print_exc()
        finally:
            sys.argv = original_argv
            if cwd:
                os.chdir(original_cwd)

        return ok

    def _run_stage(self, script_path: str, args_list: List[str],
                   description: str, cwd: Optional[str] = None) -> bool:
        if self.runner == "inprocess":
            return self._run_in_process(script_path, args_list, description, cwd)
        return self._run_subprocess(script_path, args_list, description, cwd)

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
            print("Pre-loading SAM3 model (once for all samples)...")
        with suppress_output(self.verbose):
            self._get_segmenter()
        if self.verbose:
            print("SAM3 ready.\n")

    # ------------------------------------------------------------------
    # Stage 1 – Object Grounding & Segmentation  +  Caption/FPS generation
    # ------------------------------------------------------------------

    def _materialise_frames(self, mod, sample_path: str, local_frames_dir: Path,
                            tmp_extract: Path) -> List[int]:
        """
        Copy the sample's frames into local scratch under contiguous 7-digit names
        and return the original frame indices they map back to.

        Directory samples are hard-linked where the filesystem allows it; video
        samples are decoded into a temporary directory first.
        """
        if Path(sample_path).is_dir():
            frame_files = sorted([
                f for f in Path(sample_path).iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ])
            if not frame_files:
                return []
            keep_indices = list(range(len(frame_files)))
            for new_idx, old_idx in enumerate(keep_indices):
                old_f = frame_files[old_idx]
                new_f = local_frames_dir / f"{new_idx:07d}{old_f.suffix}"
                try:
                    os.link(old_f, new_f)
                except OSError:
                    shutil.copy2(old_f, new_f)
            return keep_indices

        input_proc = mod.InputProcessor(sample_path, str(tmp_extract), frame_index=0)
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
        return keep_indices

    def _save_bbox_visualisation(self, input_proc, frame_idx: int,
                                 crop_bboxes: Dict, out_dir: Path):
        """Write a debug PNG of the VLM crop boxes over the chosen frame."""
        try:
            from PIL import Image, ImageDraw
            frame_path = input_proc.get_frame_path(frame_idx)
            vis_img = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(vis_img)
            for obj_name, box in crop_bboxes.items():
                if not box:
                    continue
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 15)), obj_name, fill="red")
            vis_img.save(str(out_dir / "bbox_visualize.png"))
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to save bbox visualization: {e}")

    def stage1_object_grounding(
        self,
        sample_path: str,
        vqa_metadata: Dict,
        vlm_model: str = "openai",
        auto_mode: bool = True,
        scan_frames: bool = True,
        threshold: float = 0.1,
        num_scan_frames: int = 5,
    ) -> Tuple[str, str, bool, List[int], str, Optional[int], int]:
        """
        Ground the objects in one sample and write its first-frame mask.

        Returns (local_frames_dir, mask_path, success, keep_indices,
                 caption, dynamic_fps, mask_frame_id).
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 1 – Object Grounding: {Path(sample_path).name}")

        mod       = self._load_stage1_module()
        segmenter = self._get_segmenter()

        stem             = self.adapter.unique_stem(sample_path)
        _tmp_parent      = Path(tempfile.mkdtemp(prefix=self.adapter.tmp_prefix))
        local_frames_dir = _tmp_parent / stem
        local_frames_dir.mkdir(parents=True, exist_ok=True)

        nvme_mask_dir = self.adapter.mask_dir(sample_path, vqa_metadata)
        tmp_extract   = Path(tempfile.mkdtemp(prefix=f"ovad_extract_{stem}_"))

        keep_indices: List[int]    = []
        caption:      str          = ""
        dynamic_fps:  Optional[int] = None
        mask_frame_id: int          = 0
        _ok = False   # drives scratch cleanup in the finally block

        try:
            import numpy as np

            keep_indices = self._materialise_frames(
                mod, sample_path, local_frames_dir, tmp_extract)
            if not keep_indices:
                return str(local_frames_dir), "", False, [], "", None, mask_frame_id

            input_proc = mod.InputProcessor(str(local_frames_dir), None, frame_index=0)
            first_frame_path, _ = input_proc.process()

            _base_url, _model = self.endpoint
            detector = mod.VLMObjectDetector(
                vlm_model, base_url=_base_url, model=_model)
            objects  = self.adapter.grounding_objects(detector, first_frame_path)
            if not objects:
                return str(local_frames_dir), "", False, [], "", None, mask_frame_id

            caption, dynamic_fps = generate_caption_and_fps(
                frames_dir=str(local_frames_dir),
                objects=self.adapter.caption_objects(objects),
                vlm_model=vlm_model,
                verbose=self.verbose,
                style=self.adapter.caption_style,
                endpoint=self.endpoint,
            )

            # ── SAM3 segmentation ─────────────────────────────────────────
            crop_bboxes: Dict[str, Any] = {}

            if scan_frames and input_proc.num_frames > 1:
                scan_kwargs = _supported_kwargs(mod.scan_frames_for_objects, {
                    "detector":           detector,
                    "num_frames_to_scan": num_scan_frames,
                    "threshold":          threshold,
                    "max_instances":      self.adapter.max_instances,
                    "bbox_padding":       self.adapter.bbox_padding,
                    "bbox_object_desc":   self.adapter.bbox_object_desc,
                })
                best_frame_idx, mask, metadata = mod.scan_frames_for_objects(
                    input_proc, objects, segmenter, **scan_kwargs)

                if self.use_best_frame:
                    mask_frame_id = best_frame_idx
                crop_bbox = (metadata or {}).get("crop_bbox")
                if crop_bbox:
                    crop_bboxes[objects[0]] = crop_bbox
            else:
                bbox = None
                if self.adapter.bbox_object_desc:
                    bbox = detector.locate_group_bbox(
                        first_frame_path,
                        object_desc=self.adapter.bbox_object_desc,
                        padding=self.adapter.bbox_padding or 0.1,
                    )
                    if bbox:
                        crop_bboxes[objects[0]] = bbox
                seg_kwargs = _supported_kwargs(segmenter.segment, {
                    "crop_bbox":     bbox,
                    "max_instances": self.adapter.max_instances,
                })
                mask, _metadata = segmenter.segment(
                    first_frame_path, objects,
                    threshold=threshold,
                    retry_with_variations=True,
                    min_threshold=0.1,
                    **seg_kwargs)

            if not np.any(mask > 0):
                return (str(local_frames_dir), "", False, [],
                        caption, dynamic_fps, mask_frame_id)

            # ── Persist mask + detected objects ───────────────────────────
            nvme_mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = nvme_mask_dir / f"{mask_frame_id:07d}.png"
            mod.save_vos_mask(mask, str(mask_path))

            # save_vos_mask may emit a _vis.png sibling — remove it immediately
            vis_path = nvme_mask_dir / f"{mask_frame_id:07d}_vis.png"
            if vis_path.exists():
                vis_path.unlink(missing_ok=True)

            if crop_bboxes:
                self._save_bbox_visualisation(
                    input_proc, mask_frame_id, crop_bboxes, nvme_mask_dir)

            with open(nvme_mask_dir / "obj.json", "w") as f:
                json.dump(objects, f, indent=4)

            _ok = True
            return (str(local_frames_dir), str(mask_path), True, keep_indices,
                    caption, dynamic_fps, mask_frame_id)

        except Exception:
            if self.verbose:
                traceback.print_exc()
            return (str(local_frames_dir), "", False, [],
                    caption, dynamic_fps, mask_frame_id)
        finally:
            shutil.rmtree(tmp_extract, ignore_errors=True)
            if not _ok:
                # Every retry re-decodes the video into a fresh mkdtemp; without
                # this a sample that fails all --max-retries attempts leaves that
                # many full frame copies behind, since the caller's give-up path
                # returns before _cleanup_local_scratch runs.
                shutil.rmtree(_tmp_parent, ignore_errors=True)

    # ------------------------------------------------------------------
    # Stage 2 – TubeletGraph Tracking
    # ------------------------------------------------------------------

    def stage2_tracking(
        self,
        sample_path: str,
        frames_dir: str,
        mask_path: str,
        fps: Optional[int],
        mask_frame_id: int = 0,
        method: str = "Ours",
    ) -> Tuple[str, str, str, str, bool]:
        """Returns (pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok)."""
        if self.verbose:
            fps_display = str(fps) if fps is not None else "default"
            print(f"\n{'='*60}\nSTAGE 2 – Tracking: {Path(sample_path).name}  "
                  f"(fps={fps_display}, mask_frame_id={mask_frame_id})")

        pred_name       = Path(frames_dir).name
        jpeg_images_dir = str(self.project_root / "_custom_dataset" / "JPEGImages" / pred_name)
        pred_out_dir    = str(self.project_root / "_pred_out" / pred_name)
        src_pred        = self.project_root / "_pred_out" / pred_name

        args = [
            "-c", self.config_path,
            "--input_dir", frames_dir,
            "--input_mask", mask_path,
            "--method", method,
            "--mask_frame_id", str(mask_frame_id),
        ]
        if fps is not None:
            args += ["--fps", str(fps)]

        ok = self._run_stage(self.quick_run_script, args,
                             "quick_run.py (TubeletGraph)",
                             cwd=str(self.project_root))

        return pred_name, str(src_pred), jpeg_images_dir, pred_out_dir, ok

    # ------------------------------------------------------------------
    # Pre-Stage 3: Temporal Alignment
    # ------------------------------------------------------------------

    def _remap_prediction_indices(self, pred_dir: str, keep_indices: List[int]):
        """Remap contiguous TubeletGraph frame IDs back to original frame indices."""
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
        sample_path: str,
        pred_name: str,
        vqa_metadata: Dict,
        frames_dir: str,
        sample_interval: int = 10,
        vlm_model: str = "openai",
        caption: str = "",
        normal_video_path: Optional[str] = None,
    ) -> Tuple[Dict, bool]:
        """
        Run prompt_vad.py over the Stage 2 state graph.

        The Stage 1 caption is forwarded via --caption.  Datasets that define a
        known-normal reference clip also pass --normal_video_path so the VLM can
        compare against a concrete baseline; no caption is generated for it.
        """
        if self.verbose:
            print(f"\n{'='*60}\nSTAGE 3 – Anomaly Detection: {Path(sample_path).name}")
            if caption:
                preview = caption[:100].replace("\n", " ")
                print(f"  Caption: '{preview}{'…' if len(caption) > 100 else ''}'")
            else:
                print("  Caption: (none)")
            if normal_video_path:
                print(f"  Normal reference: {normal_video_path}")

        rpt_path = self.adapter.report_path(sample_path, vqa_metadata)
        rpt_path.parent.mkdir(parents=True, exist_ok=True)
        _, stem = self.adapter.nvme_rel(sample_path, vqa_metadata)

        args = [
            "-c", self.config_path,
            "-p", pred_name,
            "--sample_interval", str(sample_interval),
            "--video_path", frames_dir,
            "--vlm", vlm_model,
            "--output_dir", str(rpt_path.parent),
            "--detect_anomalies",
        ]
        if caption:
            args += ["--caption", caption]
        if normal_video_path:
            args += ["--normal_video_path", normal_video_path]
        if self.verbose:
            args.append("-v")

        ok = self._run_stage(self.prompt_vad_script, args, "prompt_vad.py")

        # Normalise the report path: prompt_vad.py may name the file after any
        # of several identifiers depending on how the prediction was registered.
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
        """Remove all per-sample local scratch (never touches the storage roots)."""
        if self.keep_scratch:
            return

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
    # Per-sample pipeline
    # ------------------------------------------------------------------

    def process_sample(
        self,
        sample_path: str,
        vqa_metadata: Dict,
        vlm_model: str = "openai",
        fps: Optional[int] = None,
        sample_interval: int = 10,
        method: str = "Ours",
        auto_mode: bool = True,
    ) -> Dict:
        """
        Full three-stage pipeline for one sample.

        FPS resolution for Stage 2 (priority order):
          1. dynamic_fps from the Stage 1 caption call (already clamped to 2–10)
          2. the --fps argument
          3. None → quick_run.py falls back to its own default

        The Stage 1 caption is forwarded verbatim to Stage 3 so prompt_vad.py can
        present it to the VLM alongside the graph-derived state changes.
        """
        vp = str(sample_path)

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

        normal_video_path = self.adapter.resolve_normal_reference(vp)
        if self.verbose:
            if normal_video_path:
                print(f"  [Normal] Reference sequence: {normal_video_path}")
            elif self.adapter.name == "ipad":
                print(f"  [Normal] No reference sequence found for: {vp}")

        # ── STAGE 1 ────────────────────────────────────────────────
        ok1 = False
        for attempt in range(self.max_retries):
            try:
                with suppress_output(self.verbose):
                    (local_frames_dir, mask_path, ok1, keep_indices,
                     caption, dynamic_fps, mask_frame_id) = \
                        self.stage1_object_grounding(
                            vp, vqa_metadata=vqa_metadata,
                            vlm_model=vlm_model, auto_mode=auto_mode)
                if ok1:
                    break
            except Exception as e:
                ok1 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{self.max_retries}] Stage 1 failed: {e}")
                time.sleep(3)

        if not ok1:
            return {"success": False, "sample_path": vp, "stage": 1,
                    "error": f"Object grounding failed after {self.max_retries} attempts"}

        # ── FPS resolution ─────────────────────────────────────────
        effective_fps: Optional[int] = dynamic_fps if dynamic_fps is not None else fps
        if self.verbose:
            if dynamic_fps is not None:
                tqdm.write(f"  [FPS] Using VLM-recommended FPS={dynamic_fps} for Stage 2.")
            else:
                tqdm.write(f"  [FPS] VLM FPS unavailable; using CLI fps={fps} for Stage 2.")

        # ── STAGE 2 ────────────────────────────────────────────────
        ok2 = False
        for attempt in range(self.max_retries):
            try:
                with suppress_output(self.verbose):
                    pred_name, src_pred_path, jpeg_images_dir, pred_out_dir, ok2 = \
                        self.stage2_tracking(
                            vp, local_frames_dir, mask_path,
                            fps=effective_fps,
                            mask_frame_id=mask_frame_id,
                            method=method)
                if ok2:
                    break
            except Exception as e:
                ok2 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{self.max_retries}] Stage 2 failed: {e}")
                time.sleep(3)

        if not ok2:
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)
            return {"success": False, "sample_path": vp, "stage": 2,
                    "error": f"Tracking failed after {self.max_retries} attempts"}

        # ── TEMPORAL ALIGNMENT ─────────────────────────────────────
        if keep_indices:
            self._remap_prediction_indices(src_pred_path, keep_indices)

        # ── STAGE 3 ────────────────────────────────────────────────
        ok3 = False
        for attempt in range(self.max_retries):
            try:
                with suppress_output(self.verbose):
                    report, ok3 = self.stage3_anomaly_detection(
                        vp, pred_name,
                        vqa_metadata=vqa_metadata,
                        frames_dir=local_frames_dir,
                        sample_interval=sample_interval,
                        vlm_model=vlm_model,
                        caption=caption,
                        normal_video_path=normal_video_path)
                if ok3:
                    break
            except Exception as e:
                ok3 = False
                if self.verbose:
                    tqdm.write(f"  [Retry {attempt+1}/{self.max_retries}] Stage 3 failed: {e}")
                time.sleep(3)

        if not ok3:
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)
            return {"success": False, "sample_path": vp, "stage": 3,
                    "error": f"Anomaly detection failed after {self.max_retries} attempts"}

        # ── POST-PROCESSING ────────────────────────────────────────
        # Copy the Stage 2 graph to permanent storage, then clear local scratch.
        with suppress_output(self.verbose):
            dst_pred = self.adapter.pred_dir(vp, vqa_metadata)
            if osp.exists(src_pred_path):
                dst_pred.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_pred_path, str(dst_pred), dirs_exist_ok=True)
            self._cleanup_local_scratch(pred_name, local_frames_dir,
                                        jpeg_images_dir, pred_out_dir)

        return {
            "success":           True,
            "sample_path":       vp,
            "prediction_name":   pred_name,
            "report_path":       str(self.adapter.report_path(vp, vqa_metadata)),
            "anomaly_detected":  report.get("anomaly_detected", False),
            "caption":           caption,
            "dynamic_fps":       dynamic_fps,
            "normal_video_path": normal_video_path,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "O-VAD: object-centric video anomaly detection (3-stage pipeline).\n\n"
            "Examples:\n"
            "  python pipeline.py analyze /data/Phys-AD/ball --dataset physad \\\n"
            "      -c configs/default.yaml --vqa_file /data/vqa/PhysAD_VQA.jsonl \\\n"
            "      --vlm openai --fps 3 --sample_interval 60 --split test\n\n"
            "  python pipeline.py analyze /data/IPAD/IPAD_dataset/S04 --dataset ipad \\\n"
            "      -c configs/default.yaml --vqa_file /data/vqa/IPAD_VQA.jsonl \\\n"
            "      --vlm openai --fps 3 --sample_interval 25 --split test\n\n"
            "  python pipeline.py analyze /data/AutoLab/test/abnormal_1 --dataset autolab \\\n"
            "      -c configs/default.yaml --vqa_file /data/AutoLab/autolab.jsonl \\\n"
            "      --dataset_root /data/AutoLab --vlm openai --fps 3 \\\n"
            "      --sample_interval 60 --split test"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", choices=["analyze"])
    p.add_argument("video",
                   help="Root directory searched recursively for samples "
                        "(.mp4 files, or frame directories for --dataset ipad), "
                        "filtered by --split and --vqa_file.")
    p.add_argument("--dataset", required=True, choices=sorted(ADAPTERS),
                   help="Benchmark layout to use ('autolab' is the LiquidAD data).")
    p.add_argument("-c", "--config", required=True,
                   help="TubeletGraph YAML config file.")
    p.add_argument("-o", "--output", default="output",
                   help="Kept for CLI compat; actual outputs go to STAGE1/2/3_ROOT.")
    p.add_argument("--vqa_file", required=True,
                   help="Ground-truth JSONL; also acts as the sample allow-list.")
    p.add_argument("--split", type=str, default="test",
                   help="Value of 'org_split' in the VQA file to process.")
    p.add_argument("--vlm", default="openai", choices=["openai", "qwen"],
                   help="VLM backbone for all three stages. 'openai' uses the "
                        "OpenAI API ($OPENAI_API_KEY); 'qwen' uses an "
                        "OpenAI-compatible server such as a local vLLM, set via "
                        "vlm.base_url in the config or $OVAD_VLM_BASE_URL.")
    p.add_argument("--fps", type=int, default=None,
                   help="Fallback FPS for Stage 2 when the VLM-recommended FPS is "
                        "unavailable (e.g. the caption call failed).")
    p.add_argument("--sample_interval", type=int, default=10,
                   help="Frame sampling interval passed to prompt_vad.py.")
    p.add_argument("--method", default="Ours",
                   help="Method key from the config (e.g. Ours, Ours_abl_prox).")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Dataset root used to resolve relative VQA video_path "
                        "values (autolab). Defaults to the AutoLab/ component "
                        "found in the positional video argument.")
    p.add_argument("--runner", choices=["inprocess", "subprocess"], default=None,
                   help="How Stages 2/3 are launched. 'inprocess' shares loaded "
                        "models across stages; 'subprocess' isolates them. "
                        "Defaults per dataset (physad: inprocess, others: subprocess).")
    p.add_argument("--mask-frame-mode", dest="mask_frame_mode",
                   choices=["auto", "first", "best"], default="auto",
                   help="Which frame the Stage 1 mask is anchored to. 'first' "
                        "always uses frame 0; 'best' uses the highest-coverage "
                        "scanned frame. Default per dataset (autolab: best, "
                        "others: first).")
    p.add_argument("--sample_id_s", type=int, default=None,
                   help="Shard start index (inclusive) into the discovered sample "
                        "list, for splitting a run across jobs or GPUs.")
    p.add_argument("--sample_id_e", type=int, default=None,
                   help="Shard end index (exclusive). Launch several jobs with "
                        "disjoint [--sample_id_s, --sample_id_e) ranges pointing "
                        "at the same STAGE*_ROOT; their outputs merge on disk.")
    p.add_argument("--keep-scratch", dest="keep_scratch", action="store_true",
                   help="Retain local TubeletGraph scratch dirs for debugging.")
    p.add_argument("--max-retries", dest="max_retries", type=int, default=MAX_RETRIES,
                   help=f"Attempts per stage before a sample is marked failed "
                        f"(default: {MAX_RETRIES}). Each failed attempt sleeps 3s, "
                        f"so lower this when debugging a broken environment.")
    p.add_argument("--no-auto", action="store_true",
                   help="Disable automatic object detection in Stage 1.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _resolve_endpoint(vlm: str, config_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve (base_url, model) for the chosen backbone and publish it to the
    environment.

    --vlm decides, and all three stages follow it. Stages 2 and 3 run as
    separate processes; they read the two variables exported here rather than
    re-reading the config, so they cannot drift onto a different backbone.

      openai -> (None, vlm.openai_model)
      qwen   -> (vlm.base_url, <model the server reports>)

    The qwen model name comes from the server's own /v1/models rather than the
    config, so there is no second value to keep in sync — and an unreachable or
    misconfigured server is reported here, before any frames are processed.
    """
    cfg_vlm = {}
    try:
        from omegaconf import OmegaConf
        cfg_vlm = OmegaConf.load(config_path).get("vlm", {}) or {}
    except Exception:
        pass

    def _cfg(key):
        try:
            value = cfg_vlm.get(key, None)
        except Exception:
            value = None
        return str(value).strip() if value is not None else None

    if vlm == "openai":
        base_url, model = None, (_cfg("openai_model") or CAPTION_MODEL_DEFAULT)
    else:
        base_url = os.environ.get("OVAD_VLM_BASE_URL") or _cfg("base_url")
        if not base_url:
            print(
                "Error: --vlm qwen needs an OpenAI-compatible server to talk to.\n\n"
                "Start one, for example:\n"
                "  vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000 \\\n"
                "      --limit-mm-per-prompt '{\"image\": 16}'\n\n"
                f"then set vlm.base_url in {config_path}, or export OVAD_VLM_BASE_URL."
            )
            sys.exit(1)
        model = os.environ.get("OVAD_VLM_MODEL") or _cfg("qwen_model")
        if not model:
            try:
                import openai as _o
                served = _o.OpenAI(base_url=base_url, api_key="EMPTY",
                                   timeout=30.0).models.list().data
                model = served[0].id
            except Exception as e:
                print(f"Error: cannot reach the VLM server at {base_url}\n  {e}\n\n"
                      f"Start it, or set vlm.base_url to the right address.")
                sys.exit(1)

    # Publish for Stages 2 and 3. Clearing is as important as setting: a stale
    # base_url would pull them onto a local server while Stage 1 used OpenAI.
    os.environ["OVAD_VLM_BACKBONE"] = vlm
    if base_url:
        os.environ["OVAD_VLM_BASE_URL"] = base_url
    else:
        os.environ.pop("OVAD_VLM_BASE_URL", None)
    os.environ["OVAD_VLM_MODEL"] = model

    print(f"[VLM] backbone={vlm}  model={model}  "
          f"endpoint={base_url or 'the OpenAI API'}  (all three stages)")
    return base_url, model


def _derive_dataset_root(dataset: str, video_root: Path,
                         explicit: Optional[str]) -> Optional[str]:
    """Resolve --dataset_root, falling back to the dataset anchor in --video."""
    if explicit:
        return explicit
    if dataset != "autolab":
        return None
    parts = video_root.parts
    if AutoLabAdapter.anchor in parts:
        idx = parts.index(AutoLabAdapter.anchor)
        return str(Path(*parts[:idx + 1]))
    return str(video_root)


def main():
    args = get_parser().parse_args()

    video_root = Path(args.video).resolve()
    if not video_root.is_dir():
        print(f"Error: input directory does not exist: {video_root}")
        sys.exit(1)

    endpoint = _resolve_endpoint(args.vlm, args.config)

    dataset_root = _derive_dataset_root(args.dataset, video_root, args.dataset_root)
    adapter      = ADAPTERS[args.dataset](dataset_root=dataset_root)

    pipeline = OVADPipeline(
        adapter=adapter,
        config_path=args.config,
        input_root=str(video_root),
        output_dir=args.output,
        verbose=args.verbose,
        runner=args.runner,
        keep_scratch=args.keep_scratch,
        mask_frame_mode=args.mask_frame_mode,
        max_retries=args.max_retries,
        endpoint=endpoint,
    )

    # ── Load VQA metadata ──────────────────────────────────────────────────
    if args.verbose:
        print(f"Reading VQA file: {args.vqa_file}")
    try:
        vqa_metadata = adapter.load_vqa(args.vqa_file)
    except Exception as e:
        print(f"Error reading VQA file: {e}")
        sys.exit(1)

    if args.verbose:
        n_entries = len({id(v) for v in vqa_metadata.values()})
        print(f"Loaded {n_entries} VQA entries ({len(vqa_metadata)} lookup keys).")

    # ── Discover samples ───────────────────────────────────────────────────
    samples = adapter.discover(video_root, vqa_metadata, args.split)

    if args.verbose:
        print(f"Found {len(samples)} sample(s) under '{video_root}' "
              f"matching split='{args.split}'.")

    if not samples:
        print(f"No samples found under '{args.video}' matching split='{args.split}'. "
              f"Check --vqa_file, --dataset and --split.")
        sys.exit(0)

    # ── Shard ──────────────────────────────────────────────────────────────
    # The slice applies to the discovered (VQA- and split-filtered) list, so
    # equal-width ranges yield equal-sized shards.
    if args.sample_id_s is not None or args.sample_id_e is not None:
        n_total = len(samples)
        samples = samples[args.sample_id_s:args.sample_id_e]
        print(f"Shard [{args.sample_id_s}:{args.sample_id_e}] → "
              f"{len(samples)} of {n_total} sample(s).")
        if not samples:
            print("Shard range selects no samples — nothing to do.")
            sys.exit(0)

    pipeline.preload_sam3()
    failed: List[Path] = []

    label = "Sequences" if adapter.kind == "sequence" else "Videos"
    for i in trange(len(samples), desc=f"Processing {label}", dynamic_ncols=True):
        vp = samples[i]

        rpt = adapter.report_path(str(vp), vqa_metadata)
        if rpt.is_file():
            tqdm.write(f"  [SKIP] {rpt.name}")
            continue

        try:
            result = pipeline.process_sample(
                sample_path=str(vp),
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

    total = len(samples)
    print(f"\nDone. {total - len(failed)}/{total} processed successfully.")

    if failed:
        print(f"{len(failed)} failure(s):")
        for ff in failed:
            print(f"  {ff}")
        sys.exit(1)


if __name__ == "__main__":
    main()
