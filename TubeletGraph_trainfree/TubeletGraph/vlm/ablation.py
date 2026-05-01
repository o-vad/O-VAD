#!/usr/bin/env python3
"""
ablation_study.py — ST-VAD Ablation Study Runner

Re-runs Stage 3 (VLM anomaly reasoning + visual verifier) using pre-computed
full_report.json files as inputs, with systematic component removal.

Ablation IDs:
  0: full              — all inputs intact
  1: w/o caption       — caption replaced with ""
  2: w/o state changes — state_changes replaced with []
  3: w/o CoT           — CoT reasoning chain replaced by direct single-shot prompt
  4: w/o verifier      — skip verify_anomalies_visually step

Dataset detection (auto from dataset_path):
  "Phys-AD" in path → PhysAD mode: video .mp4 files, fps-sampled frames, BERT metric
  "IPAD"    in path → IPAD mode:   image-sequence dirs,  fps-sampled frames, frame-AUROC metric

Full-report directory structure (mirrors dataset layout):
  full_report_dir/
    <category>/           e.g. sticky_roller
      <split>/            e.g. test
        [<subset>/]       e.g. detach/
          0000_report.json

Output structure mirrors full_report_dir exactly, under ablation sub-folder:
  output_root/
    <ablation_id>/
      <dataset_name>/
        <category>/
          <split>/
            [<subset>/]
              0000_report.json

Usage (PhysAD):
  python ablation_study.py \\
    --dataset_path    /u/qilong/anomaly_detect/datasets/Phys-AD \\
    --vqa_file        /u/qilong/anomaly_detect/datasets/vqa_data/PhysAD_VQA.jsonl \\
    --full_report_dir /work/nvme/bgiv/qilong/stage3/PhysAD \\
    --dataset_name    PhysAD \\
    --fps 5 \\
    --ablation 0 1 2 3 4 \\
    --vlm openai \\
    --output_root /work/nvme/bgiv/qilong/ablations

Usage (IPAD):
  python ablation_study.py \\
    --dataset_path    /u/qilong/anomaly_detect/datasets/IPAD/S02 \\
    --vqa_file        /u/qilong/anomaly_detect/datasets/vqa_data/IPAD_VQA.jsonl \\
    --full_report_dir /work/nvme/bgiv/qilong/stage3/IPAD/IPAD_dataset \\
    --dataset_name    IPAD_dataset \\
    --fps 5 \\
    --ablation 0 1 2 3 4 \\
    --vlm openai \\
    --output_root /work/nvme/bgiv/qilong/ablations
"""

import os
import sys
import json
import re
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

# ── reuse VLM clients from prompt_vad.py ─────────────────────────────────────
from prompt_vad import (
    VLMClient, create_vlm_client,
    PROMPT_VISUAL_VERIFICATION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET MODE DETECTION
# =============================================================================

def detect_dataset_mode(dataset_path: str) -> str:
    p = str(dataset_path).lower()
    if "phys-ad" in p or "physad" in p:
        return "physad"
    if "ipad" in p:
        return "ipad"
    logger.warning(
        f"Cannot auto-detect dataset mode from path '{dataset_path}'. "
        "Defaulting to 'physad'. Use --dataset_mode to override."
    )
    return "physad"


# =============================================================================
# FRAME LOADING  (fps-aware)
# =============================================================================

def load_frames_from_video(video_path: str, target_fps: float) -> List[Tuple[int, Any]]:
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not available — cannot extract frames from video.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if target_fps <= 0 or target_fps >= native_fps:
        step = 1
    else:
        step = max(1, round(native_fps / target_fps))

    frames: List[Tuple[int, Any]] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            import cv2 as _cv2
            frames.append((idx, _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    logger.info(
        f"Video {Path(video_path).name}: native={native_fps:.1f}fps "
        f"→ sampled {len(frames)} frames at ~{target_fps}fps (step={step})"
    )
    return frames


def load_frames_from_image_dir(image_dir: str, target_fps: float,
                                native_fps: float = 30.0) -> List[Tuple[int, Any]]:
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not available — cannot load images.")
        return []

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    if not files:
        return []

    if target_fps <= 0 or target_fps >= native_fps:
        step = 1
    else:
        step = max(1, round(native_fps / target_fps))

    frames: List[Tuple[int, Any]] = []
    for i, fp in enumerate(files):
        if i % step == 0:
            img = cv2.imread(str(fp))
            if img is not None:
                frames.append((i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    logger.info(
        f"Dir {Path(image_dir).name}: {len(files)} images "
        f"→ sampled {len(frames)} frames at ~{target_fps}fps (step={step})"
    )
    return frames


def load_sample_frames(
    video_or_dir: Optional[str],
    dataset_mode: str,
    target_fps: float,
    native_fps: float = 30.0,
) -> List[Tuple[int, Any]]:
    if not video_or_dir:
        return []

    vp = str(video_or_dir)
    if dataset_mode == "ipad":
        if os.path.isdir(vp):
            return load_frames_from_image_dir(vp, target_fps, native_fps)
        if os.path.isfile(vp):
            return load_frames_from_video(vp, target_fps)
    else:  # physad
        if os.path.isfile(vp):
            return load_frames_from_video(vp, target_fps)
        if os.path.isdir(vp):
            return load_frames_from_image_dir(vp, target_fps, native_fps)

    logger.warning(f"Cannot load frames from: {vp}")
    return []


# =============================================================================
# CONSTANTS / PROMPTS
# =============================================================================

ABLATION_NAMES = {
    0: "full",
    1: "wo_caption",
    2: "wo_state_changes",
    3: "wo_CoT",
    4: "wo_verifier",
}

SYSTEM_PROMPT_ANOMALY = """You are an expert anomaly detection system for industrial processes.
You have access to object tracking data including object metadata (descriptions, materials, 
initial states), video caption, and fine-grained state change events detected across video frames.

Your role is to:
1. Carefully study the tracked object metadata and state change events provided
2. Observe object states and changes grounded in this tracking data
3. Compare observations against expected normal behaviors for these specific materials/objects
4. Identify deviations that indicate anomalies or failures
5. Freely classify anomalies based on your broad domain knowledge
6. Provide clear step-by-step reasoning grounded in the evidence from tracking

Be thorough but avoid false positives. Consider physical plausibility and context.
Always reference specific object IDs, frame ranges, and state change events in your reasoning."""


PROMPT_ANOMALY_REASONING = """You are analyzing video of a manipulation task for anomalies.
Anomalies should cause or reflect real flaw in object, real flaw in assembly line or real damage to object.

TASK CONTEXT:
{task_context}

ANOMALY CRITERIA:
1. The anomalies should be related to OBJECT malfunction, flaw in object, irreversible 
damage to object or industrial assembly line, others like object changes in location, 
slight changes in shape, normal changes without damage, should NOT be included.
2. Should NOT about the inconsistency of caption and frames.
3. Should NOT about forecasting. For example, if anomaly is about damage but 
damage is not directly shown, do NOT output damage "may happen".
4. Should NOT about unrealistic. All cases are real world or simulated data, anomalies 
like "cardbox is leaking" should be regarded as analysis noise instead of anomaly.
5. Should NOT about operation. Repetition or recurrence of same operation is normal for
testing the object's functionality, do NOT regard it as abnormal.

VIDEO CAPTION (from video sampled frames):
{caption}

TRACKED OBJECTS (from automatic detection and tracking):
{object_info}

STATE CHANGES DETECTED (from object-centric state tracking):
{state_changes}

Follow this 6-step reasoning chain. Ground every step in the video caption, tracked object 
metadata and state change events provided above.

STEP 1 - OBSERVATION:
What specific changes and events occurred? Reference the tracked objects by 
their IDs and descriptions. Cite the frame ranges and state change types from 
the tracking data. Note each object's material, initial state, and how it evolved.

STEP 2 - EXPECTATION:
Given the task context and the objects' materials/properties, what should have 
happened? What constitutes normal behavior for these specific objects and this 
process? Consider physical plausibility given the materials involved. For example,
bearings should constantly rotate without stopping, conveyor belt should constantly 
move in even speed without stopping.

STEP 3 - COMPARISON:
How do the observed state changes differ from expectations? Be specific:
which object, which frames, what change was unexpected and why?

STEP 4 - CAUSATION:
What could cause these deviations? Reason freely — consider equipment issues, 
material defects, process errors, environmental factors, or any other plausible 
cause. Do NOT limit yourself to predefined categories.

STEP 5 - CLASSIFICATION:
Classify any anomalies you found using your own judgment. The anomalies should meet 
ANOMALY CRITERIA. You are free to name the anomaly type and subtype based on what 
you observe — there is no fixed taxonomy. For reference, here are some EXAMPLES of 
anomaly types seen in  industrial settings, but you should create your own labels 
if none of these fit:
  - manipulation_failure (e.g., grip slip, misalignment, get stuck, function failure)
  - material_anomaly (e.g., unexpected leakage, contamination)
  - deformation_anomaly (e.g., structural damage, incomplete surface, lose elaticity)
  - process_anomaly (e.g., detach, assembly line halt, object pose not aligned with 
  assembly line, object placed on edge of assembly line)
These are only examples. Use whatever classification best describes your findings.

STEP 6 - SEVERITY ASSESSMENT:
Rate severity (none/low/medium/high/critical) based on:
- If no anomaly detected, severity should be none.
- Impact on task completion
- Safety implications
- Quality implications
- Reversibility of the issue

If no anomalies found, anomalies should be empty [].

Output your analysis as JSON:
{{
  "reasoning": {{
    "step1_observation": "detailed observations referencing object IDs, frames, state changes...",
    "step2_expectation": "expected normal behavior given materials and task...",
    "step3_comparison": "specific deviations with object IDs and frame ranges...",
    "step4_causation": "possible causes...",
    "step5_classification": "your anomaly classifications (free-form, not restricted to examples)...",
    "step6_severity": "severity assessment with justification..."
  }},
  "anomalies": [
    {{
      "anomaly_type": "your chosen type (free-form)",
      "anomaly_subtype": "your chosen subtype (free-form)",
      "severity": "none/low/medium/high/critical",
      "description": "detailed description referencing tracked objects and state changes",
      "affected_objects": ["obj_ids from tracking data"],
      "evidence_frames": [frame_numbers from state change events],
      "confidence": 0.0-1.0
    }}
  ],
  "is_anomalous": true/false,
  "overall_severity": "none/low/medium/high/critical",
  "overall_confidence": 0.0-1.0,
  "summary": "one paragraph summary grounded in the tracking data"
}}"""

PROMPT_DIRECT_ANOMALY = """You are analyzing video of a manipulation task for anomalies.
Anomalies should cause or reflect real flaw in object, real flaw in assembly line or real damage to object.

TASK CONTEXT:
{task_context}

ANOMALY CRITERIA:
1. The anomalies should be related to OBJECT malfunction, flaw in object, irreversible 
damage to object or industrial assembly line, others like object changes in location, 
slight changes in shape, normal changes without damage, should NOT be included.
2. Should NOT about the inconsistency of caption and frames.
3. Should NOT about forecasting. For example, if anomaly is about damage but 
damage is not directly shown, do NOT output damage "may happen".
4. Should NOT about unrealistic. All cases are real world or simulated data, anomalies 
like "cardbox is leaking" should be regarded as analysis noise instead of anomaly.
5. Should NOT about operation. Repetition or recurrence of same operation is normal for
testing the object's functionality, do NOT regard it as abnormal.

VIDEO CAPTION (from video sampled frames):
{caption}

TRACKED OBJECTS (from automatic detection and tracking):
{object_info}

STATE CHANGES DETECTED (from object-centric state tracking):
{state_changes}

Output your analysis as JSON:
{{
  "anomalies": [
    {{
      "anomaly_type": "your chosen type (free-form)",
      "anomaly_subtype": "your chosen subtype (free-form)",
      "severity": "none/low/medium/high/critical",
      "description": "detailed description referencing tracked objects and state changes",
      "affected_objects": ["obj_ids from tracking data"],
      "evidence_frames": [frame_numbers from state change events],
      "confidence": 0.0-1.0
    }}
  ],
  "is_anomalous": true/false,
  "overall_severity": "none/low/medium/high/critical",
  "overall_confidence": 0.0-1.0,
  "summary": "one paragraph summary grounded in the tracking data"
}}"""

DEFAULT_TASK_CONTEXT = (
    "Task: Industrial manipulation operation\n"
    "Expected behavior: Controlled object manipulation without damage\n"
    "Normal patterns: Smooth state transitions, expected material behaviors\n"
    "Failure conditions: Unexpected deformation, material leakage, grip failures"
)


# =============================================================================
# VQA MATCHING
# =============================================================================

# Anchor strings that mark the dataset root in a video_path
_DATASET_ANCHORS = ("Phys-AD/", "IPAD_dataset/")


def _vqa_path_to_key(video_path: str) -> str:
    """Extract the identifying segment from a VQA video_path.

    Strips everything up to and including the dataset anchor
    ("Phys-AD/" or "IPAD_dataset/"), removes file extension, and
    replaces remaining "/" with "_".

    Examples
    --------
    /u/.../Phys-AD/sticky_roller/test/detach/0000.mp4
        → "sticky_roller_test_detach_0000"

    /u/.../IPAD_dataset/R01/testing/frames/02
        → "R01_testing_frames_02"
    """
    norm = video_path.replace("\\", "/")
    for anchor in _DATASET_ANCHORS:
        idx = norm.find(anchor)
        if idx != -1:
            segment = norm[idx + len(anchor):]
            break
    else:
        segment = norm.lstrip("/")

    # Strip file extension from the last component
    last = segment.rsplit("/", 1)[-1]
    if "." in last:
        segment = segment.rsplit(".", 1)[0]

    return segment.replace("/", "_")


def load_vqa_index(vqa_file: str) -> Dict[str, dict]:
    """Load VQA JSONL and index every entry by its normalised key.

    Key format (examples):
      "sticky_roller_test_detach_0000"
      "R01_testing_frames_02"
    """
    index: Dict[str, dict] = {}
    with open(vqa_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"VQA line {lineno} parse error: {e}")
                continue
            vp = entry.get("video_path", "")
            if not vp:
                continue
            key = _vqa_path_to_key(vp)
            index[key] = entry
    return index


def match_vqa_entry(video_name: str, vqa_index: Dict[str, dict]) -> Optional[dict]:
    """Exact-match a report's video_name against the VQA index.

    report video_name  e.g. "sticky_roller_test_detach_0000"
    vqa key            e.g. "sticky_roller_test_detach_0000"
    → direct dict lookup.
    """
    return vqa_index.get(video_name)


# =============================================================================
# OUTPUT PATH
# =============================================================================

def output_report_path(
    report_path: Path,
    full_report_dir: Path,
    output_root: Path,
    ablation_id: int,
    dataset_name: str,
) -> Path:
    """Compute the output path by finding the dataset anchor in report_path.

    Scans report_path's parts for a component that matches dataset_name
    (e.g. "PhysAD" or "IPAD_dataset") and takes everything after it as the
    relative sub-path.  This is independent of what was passed as
    full_report_dir, so it always preserves the full structure under the
    dataset root.

    Example
    -------
    report_path  = /work/.../stage3/PhysAD/sticky_roller/test/detach/0000_report.json
    dataset_name = PhysAD
    ablation_id  = 1
    → rel        = sticky_roller/test/detach/0000_report.json
    → output     = <output_root>/1/PhysAD/sticky_roller/test/detach/0000_report.json
    """
    parts = report_path.parts  # tuple of all path components

    # Find the index of the dataset_name component in the path
    anchor_idx = None
    for i, part in enumerate(parts):
        if part == dataset_name:
            anchor_idx = i
            break

    if anchor_idx is not None and anchor_idx + 1 < len(parts):
        # Take everything after the dataset_name component
        rel = Path(*parts[anchor_idx + 1:])
    else:
        # Fallback: use relative_to(full_report_dir) as before
        try:
            rel = report_path.relative_to(full_report_dir)
        except ValueError:
            rel = Path(report_path.name)

    return output_root / str(ablation_id) / dataset_name / rel

# =============================================================================
# SAMPLE DISCOVERY
# =============================================================================

def discover_samples(
    dataset_path: Path,
    full_report_dir: Path,
    vqa_index: Dict[str, dict],
    dataset_mode: str,
) -> List[Dict]:
    """Walk full_report_dir for all *_report.json files and pair each with its
    VQA entry and media source path."""
    samples: List[Dict] = []
    seen: set = set()

    for report_path in sorted(full_report_dir.rglob("*_report.json")):
        try:
            with open(report_path) as f:
                rdata = json.load(f)
        except Exception as e:
            logger.warning(f"Cannot read {report_path}: {e}")
            continue

        video_name = (rdata.get("video_name") or rdata.get("prediction_name") or "").strip()
        if not video_name:
            logger.warning(f"No video_name in {report_path}, skipping.")
            continue
        if video_name in seen:
            continue

        # Match to VQA entry via exact key lookup
        vqa_entry = match_vqa_entry(video_name, vqa_index)
        if vqa_entry is None:
            logger.debug(f"No VQA entry for '{video_name}', skipping.")
            continue

        # Locate media source
        media_path: Optional[Path] = None
        vp_str = vqa_entry.get("video_path", "")

        if dataset_mode == "physad":
            if vp_str and Path(vp_str).is_file():
                media_path = Path(vp_str)
            if media_path is None:
                idx = report_path.stem.replace("_report", "")
                rel_parts = report_path.relative_to(full_report_dir).parts
                if len(rel_parts) >= 2:
                    cat, spl = rel_parts[0], rel_parts[1]
                    hits = list((dataset_path / cat / spl).rglob(f"{idx}.mp4"))
                    if not hits:
                        hits = list(dataset_path.rglob(f"{idx}.mp4"))
                else:
                    hits = list(dataset_path.rglob(f"{idx}.mp4"))
                if hits:
                    media_path = hits[0]
        else:  # ipad
            if vp_str and Path(vp_str).is_dir():
                media_path = Path(vp_str)
            if media_path is None:
                hits_d = [p for p in dataset_path.rglob(video_name) if p.is_dir()]
                if hits_d:
                    media_path = hits_d[0]

        if media_path is None:
            logger.warning(f"Media source not found for '{video_name}' — frames unavailable.")

        samples.append({
            "video_name":  video_name,
            "media_path":  media_path,
            "report_path": report_path,
            "vqa_entry":   vqa_entry,
        })
        seen.add(video_name)

    logger.info(f"Discovered {len(samples)} samples.")
    return samples


# =============================================================================
# JSON PARSING
# =============================================================================

def _parse_json_response(text: str) -> Dict:
    for fn in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', t).group(1)),
        lambda t: json.loads(re.search(r'\{[\s\S]*\}', t).group(0)),
    ]:
        try:
            return fn(text)
        except Exception:
            pass
    logger.warning("Could not parse JSON from VLM response.")
    return {}


# =============================================================================
# ABLATION ENGINE
# =============================================================================

class AblationEngine:
    def __init__(
        self,
        vlm_client: VLMClient,
        ablation_id: int,
        dataset_mode: str,
        target_fps: float,
        verbose: bool = False,
    ):
        self.vlm_client   = vlm_client
        self.ablation_id  = ablation_id
        self.dataset_mode = dataset_mode
        self.target_fps   = target_fps
        self.verbose      = verbose

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def _build_prompt(self, caption, object_info_str, state_changes_str, task_context) -> str:
        if self.ablation_id == 3:
            return PROMPT_DIRECT_ANOMALY.format(
                task_context=task_context, caption=caption,
                object_info=object_info_str, state_changes=state_changes_str,
            )
        return PROMPT_ANOMALY_REASONING.format(
            task_context=task_context, caption=caption,
            object_info=object_info_str, state_changes=state_changes_str,
        )

    def run_sample(
        self,
        full_report: Dict,
        media_path: Optional[str],
        task_context: Optional[str] = None,
    ) -> Dict:
        t0 = time.time()

        # ── 1. Extract inputs from full_report ───────────────────────────────
        caption: str        = str(full_report.get("caption", "") or "")
        state_changes: List = list(full_report.get("state_changes", []) or [])

        obj_info: Dict[str, Dict] = {}
        for sc in state_changes:
            oid = str(sc.get("obj_id", ""))
            if oid and oid not in obj_info:
                obj_info[oid] = {"desc": sc.get("obj_name", "object")}

        # ── 2. Apply ablation suppression ────────────────────────────────────
        if self.ablation_id == 1:
            caption = ""
        elif self.ablation_id == 2:
            state_changes = []
            obj_info      = {}

        # ── 3. Serialise for prompt ───────────────────────────────────────────
        obj_list          = [{"obj_id": k, **v} for k, v in obj_info.items()]
        object_info_str   = json.dumps(obj_list,      indent=2)
        state_changes_str = json.dumps(state_changes, indent=2)
        task_context      = task_context or DEFAULT_TASK_CONTEXT

        # ── 4. Load frames at target_fps ─────────────────────────────────────
        frames = load_sample_frames(media_path, self.dataset_mode, self.target_fps)

        images = None
        if frames:
            idxs = [0]
            if len(frames) > 2:
                idxs.append(len(frames) // 2)
            if len(frames) > 1:
                idxs.append(len(frames) - 1)
            images = [frames[i][1] for i in idxs if i < len(frames)]
            self._log(f"Using {len(images)} key frames for VLM.")

        # ── 5. Call VLM ───────────────────────────────────────────────────────
        prompt   = self._build_prompt(caption, object_info_str, state_changes_str, task_context)
        response = self.vlm_client.query(
            prompt=prompt,
            images=images,
            system_prompt=SYSTEM_PROMPT_ANOMALY,
            temperature=0.0,
            max_tokens=4096,
        )
        result = _parse_json_response(response)

        # ── 6. Normalise anomaly frame fields ─────────────────────────────────
        raw_anomalies: List[Dict] = list(result.get("anomalies", []))
        for a in raw_anomalies:
            evf      = a.get("evidence_frames", [])
            safe_evf = [int(f) for f in evf if isinstance(f, (int, float))]
            a.setdefault("start_frame", min(safe_evf) if safe_evf else 0)
            a.setdefault("end_frame",   max(safe_evf) if safe_evf else 0)

        # ── 7. Optional visual verifier ───────────────────────────────────────
        if self.ablation_id != 4 and frames and raw_anomalies:
            self._log("Running visual verifier …")
            raw_anomalies = self._verify_anomalies(raw_anomalies, frames, caption)

        # ── 8. Compose output ─────────────────────────────────────────────────
        for i, a in enumerate(raw_anomalies):
            a["anomaly_id"] = f"anomaly_{i:04d}"

        is_anomalous       = len(raw_anomalies) > 0
        overall_severity   = result.get("overall_severity", "none")
        overall_confidence = result.get("overall_confidence", "0.0")
        if is_anomalous and overall_severity == "none":
            sev_order      = ["none", "low", "medium", "high", "critical"]
            overall_severity = sev_order[
                max(sev_order.index(a.get("severity", "none")) for a in raw_anomalies)
            ]

        reasoning_out = {} if self.ablation_id == 3 else result.get("reasoning", {})

        return {
            "video_name":         full_report.get("video_name", ""),
            "prediction_name":    full_report.get("prediction_name", ""),
            "ablation_id":        self.ablation_id,
            "ablation_name":      ABLATION_NAMES[self.ablation_id],
            "caption":            caption,
            "anomaly_detected":   is_anomalous,
            "num_anomalies":      len(raw_anomalies),
            "overall_confidence": overall_confidence,
            "overall_severity":   overall_severity,
            "anomalies":          raw_anomalies,
            "reasoning":          reasoning_out,
            "summary":            result.get("summary", ""),
            "timestamp":          datetime.now().isoformat(),
            "processing_time":    round(time.time() - t0, 2),
        }

    def _verify_anomalies(
        self,
        anomalies: List[Dict],
        frames: List[Tuple[int, Any]],
        caption: str,
    ) -> List[Dict]:
        verified = []
        for anomaly in anomalies:
            conf = float(anomaly.get("confidence", 0.8))
            if conf >= 0.8:
                verified.append(anomaly)
                continue
            if conf <= 0.2:
                continue

            frame_images = []
            for ev_frame in anomaly.get("evidence_frames", [])[:3]:
                for _, frame in frames:
                    frame_images.append(frame)
                    break
            if not frame_images and frames:
                frame_images = [frames[len(frames) // 2][1]]
            if not frame_images:
                verified.append(anomaly)
                continue

            prompt = PROMPT_VISUAL_VERIFICATION.format(
                anomaly_type=anomaly.get("anomaly_type", "unknown"),
                anomaly_description=anomaly.get("description", ""),
                affected_objects=", ".join(str(o) for o in anomaly.get("affected_objects", [])),
                caption=caption,
            )
            try:
                resp    = self.vlm_client.query(
                    prompt=prompt, images=frame_images,
                    system_prompt=SYSTEM_PROMPT_ANOMALY,
                    temperature=0.0, max_tokens=1024,
                )
                vresult = _parse_json_response(resp)
                vconf   = float(vresult.get("confidence", 0.8))

                if vresult.get("verified", True):
                    anomaly["verified"]   = True
                    anomaly["confidence"] = conf * vconf
                else:
                    anomaly["verified"]   = False
                    anomaly["confidence"] = conf * (1.0 - vconf)

                if anomaly["confidence"] >= 0.3:
                    verified.append(anomaly)
                else:
                    self._log(f"Verifier discarded anomaly (conf={anomaly['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"Verifier failed: {e} — keeping anomaly.")
                verified.append(anomaly)
        return verified


# =============================================================================
# CLI
# =============================================================================

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ST-VAD Ablation Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation IDs:
  0  full              all inputs intact
  1  w/o caption       caption -> ""
  2  w/o state changes state_changes -> []
  3  w/o CoT           direct single-shot prompt (no 6-step chain)
  4  w/o verifier      skip visual verification step

Dataset mode is auto-detected from dataset_path:
  "Phys-AD" -> physad  (mp4 videos, BERT metric)
  "IPAD"    -> ipad    (image dirs, frame-AUROC metric)
Use --dataset_mode to override.
""",
    )
    p.add_argument("--dataset_path",    required=True)
    p.add_argument("--vqa_file",        required=True)
    p.add_argument("--full_report_dir", required=True,
                   help="Root dir containing *_report.json files mirroring dataset layout.")
    p.add_argument("--dataset_name",    default="PhysAD",
                   help="Sub-folder name in output (default: PhysAD).")
    p.add_argument("--dataset_mode",    default=None, choices=["physad", "ipad"],
                   help="Override auto-detected dataset mode.")
    p.add_argument("--ablation",        type=int, nargs="+", default=[1, 2, 3, 4],
                   choices=[0, 1, 2, 3, 4])
    p.add_argument("--vlm",             default="openai", choices=["openai", "claude", "ollama"])
    p.add_argument("--fps",             type=float, default=5.0)
    p.add_argument("--native_fps",      type=float, default=30.0,
                   help="Native FPS for IPAD image sequences (ignored for PhysAD).")
    p.add_argument("--output_root",     default="/work/nvme/bgiv/qilong/ablations")
    p.add_argument("-v", "--verbose",   action="store_true")
    return p


def main():
    args = get_parser().parse_args()

    dataset_path    = Path(args.dataset_path).resolve()
    full_report_dir = Path(args.full_report_dir).resolve()
    output_root     = Path(args.output_root)

    if not dataset_path.is_dir():
        print(f"ERROR: dataset_path does not exist: {dataset_path}")
        sys.exit(1)
    if not full_report_dir.is_dir():
        print(f"ERROR: full_report_dir does not exist: {full_report_dir}")
        sys.exit(1)

    dataset_mode = args.dataset_mode or detect_dataset_mode(str(dataset_path))
    print(f"Dataset mode : {dataset_mode.upper()}")
    print(f"Target FPS   : {args.fps}")

    # Load VQA index (key = normalised path segment, e.g. "sticky_roller_test_detach_0000")
    logger.info(f"Loading VQA index from {args.vqa_file} …")
    vqa_index = load_vqa_index(args.vqa_file)
    logger.info(f"  → {len(vqa_index)} entries indexed.")

    # Discover samples
    samples = discover_samples(dataset_path, full_report_dir, vqa_index, dataset_mode)
    if not samples:
        print("No samples found. Check --dataset_path, --full_report_dir, --vqa_file.")
        sys.exit(0)

    # Shared VLM client
    vlm_client = create_vlm_client(args.vlm)

    # Run each ablation
    for ablation_id in args.ablation:
        tag = ABLATION_NAMES[ablation_id]
        print(f"\nAblation {ablation_id}: {tag}  ({len(samples)} samples)")

        engine = AblationEngine(
            vlm_client, ablation_id, dataset_mode, args.fps, verbose=args.verbose
        )
        failed: List[str] = []

        with tqdm(samples, desc=f"Ablation {ablation_id} [{tag}]", unit="sample") as pbar:
            for sample in pbar:
                vname       = sample["video_name"]
                media_path  = sample["media_path"]
                report_path = sample["report_path"]

                # Output path mirrors full_report_dir structure
                out_file = output_report_path(
                    report_path, full_report_dir, output_root, ablation_id, args.dataset_name
                )

                if out_file.is_file():
                    pbar.set_postfix_str(f"skip {vname}")
                    continue

                pbar.set_postfix_str(vname)
                try:
                    full_report = json.loads(report_path.read_text())
                    out_report  = engine.run_sample(
                        full_report=full_report,
                        media_path=str(media_path) if media_path else None,
                    )
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(json.dumps(out_report, indent=2))
                except Exception as e:
                    import traceback as tb
                    logger.error(f"FAILED {vname}: {e}")
                    if args.verbose:
                        logger.error(tb.format_exc())
                    failed.append(vname)

        if failed:
            print(f"  {len(failed)} failure(s): {failed}")

    print("\nDone.")


if __name__ == "__main__":
    main()