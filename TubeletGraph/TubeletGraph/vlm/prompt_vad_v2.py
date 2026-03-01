#!/usr/bin/env python3
"""
prompt_vad.py - VLM-based Video Anomaly Detection with Chain-of-Thought Reasoning

This script performs Stage 3 & 4 of the ST-VAD pipeline:
- Stage 3: State change analysis and temporal pattern detection
- Stage 4: Chain-of-thought anomaly reasoning and classification

The reasoning chain follows 7 steps (Step 0–6):
0. Process Understanding: What is the process doing? Which changes are expected?
1. Observation: What changes/events occurred?
2. Expectation: What should have happened?
3. Comparison: How do observations differ from expectations?
4. Causation: What could cause these deviations?
5. Classification: What type of anomaly is this?
6. Severity: How serious is the anomaly?

Key design changes (v2):
- Step 0 separates process-induced changes (normal) from object/process failures (anomalies)
- CHANGE_TYPE and CHANGE_CAUSE from Stage 2 are free-form — no fixed taxonomy
- Noise pre-filtering before the reasoning chain (keyword-based on free-form cause labels)
- Process-aware system prompt, task context, and visual verification

Usage:
    python prompt_vad.py -c configs/default.yaml -p custom-0000-Ours \\
        --video_path video.mp4 --detect_anomalies --sample_interval 10

Author: ST-VAD Framework
"""

import os
import os.path as osp
import sys
import json
import yaml
import argparse
import base64
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

# Try to import cv2 for video processing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[Warning] OpenCV not available. Video frame extraction disabled.")

# Try to import PIL for image processing
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[Warning] PIL not available. Some image features disabled.")

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import VLM clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class AnomalySeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StateChange:
    """Represents a detected state change in an object."""
    obj_id: str
    obj_name: str
    start_frame: int
    end_frame: int
    change_type: str  # Free-form from Stage 2 (e.g., deformation, material_release, rotational_resistance, ...)
    description: str
    severity: str = "slight"  # none, slight, moderate, severe
    confidence: float = 0.8
    change_cause: str = "uncertain"  # Free-form from Stage 2 (e.g., process_action, object_failure, ...)


@dataclass
class DetectedAnomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    anomaly_type: str
    anomaly_subtype: str
    severity: str
    description: str
    affected_objects: List[str]
    evidence_frames: List[int]
    start_frame: int
    end_frame: int
    confidence: float
    reasoning_trace: Dict[str, str]


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_name: str
    step_number: int
    input_context: str
    output: str
    confidence: float = 1.0


@dataclass
class AnomalyReport:
    """Complete anomaly detection report."""
    video_name: str
    prediction_name: str
    anomaly_detected: bool
    num_anomalies: int
    overall_severity: str
    anomalies: List[DetectedAnomaly]
    reasoning_trace: List[ReasoningStep]
    state_changes: List[StateChange]
    identified_events: List[str]
    anomalous_transitions: List[Dict]
    summary: str
    timestamp: str
    processing_time: float
    caption: str


# =============================================================================
# VLM CLIENT IMPLEMENTATIONS
# =============================================================================

class VLMClient:
    """Base VLM client interface."""
    
    def query(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        raise NotImplementedError


class ClaudeClient(VLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _encode_image(self, image: Any) -> Dict:
        """Encode image for Claude API."""
        if isinstance(image, str):
            if osp.isfile(image):
                with open(image, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext = Path(image).suffix.lower()
                media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext[1:], "image/jpeg")
            else:
                data = image
                media_type = "image/jpeg"
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            if success:
                data = base64.b64encode(buffer).decode("utf-8")
                media_type = "image/jpeg"
            else:
                raise ValueError("Failed to encode image")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data
            }
        }
    
    def query(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        content = []
        
        if images:
            for img in images:
                content.append(self._encode_image(img))
        
        content.append({"type": "text", "text": prompt})
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "You are an expert industrial vision system.",
            messages=[{"role": "user", "content": content}],
            temperature=temperature
        )
        
        return message.content[0].text


class OpenAIClient(VLMClient):
    """OpenAI GPT-5 client using the new Responses API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _encode_image(self, image: Any) -> Dict:
        """Encode image for the OpenAI Responses API."""
        if isinstance(image, str):
            if osp.isfile(image):
                with open(image, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext = Path(image).suffix.lower()
                media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext[1:], "image/jpeg")
            else:
                data = image
                media_type = "image/jpeg"
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            if success:
                data = base64.b64encode(buffer).decode("utf-8")
                media_type = "image/jpeg"
            else:
                raise ValueError("Failed to encode image")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # New Responses API image schema
        return {
            "type": "input_image",
            "image_url": f"data:{media_type};base64,{data}"
        }
    
    def query(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        content = []
        
        # Append images if present
        if images:
            for img in images:
                content.append(self._encode_image(img))
        
        # New Responses API text schema
        content.append({"type": "input_text", "text": prompt})
        
        # Structure the payload using 'input' instead of 'messages'
        input_data = [{"role": "user", "content": content}]
        
        # Build keyword arguments for the Responses API
        kwargs = {
            "model": self.model,
            "input": input_data,
            "max_output_tokens": max_tokens,
            "temperature": temperature
        }
        
        # System prompts map nicely to the new 'instructions' parameter
        if system_prompt:
            kwargs["instructions"] = system_prompt
            
        # Execute using the Responses API endpoint
        response = self.client.responses.create(**kwargs)
        
        # Simplified response extraction
        return response.output_text


class OllamaClient(VLMClient):
    """Ollama local LLM client (LLaVA, etc.)."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llava"):
        if not HAS_REQUESTS:
            raise ImportError("requests package not installed")
        self.host = host
        self.model = model
    
    def _encode_image(self, image: Any) -> str:
        """Encode image for Ollama API."""
        if isinstance(image, str):
            if osp.isfile(image):
                with open(image, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            return image
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            if success:
                return base64.b64encode(buffer).decode("utf-8")
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def query(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if images:
            data["images"] = [self._encode_image(img) for img in images]
        
        response = requests.post(f"{self.host}/api/generate", json=data)
        response.raise_for_status()
        return response.json()["response"]


def create_vlm_client(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> VLMClient:
    """Factory function to create VLM client."""
    provider = provider.lower()
    
    if provider in ["claude", "anthropic"]:
        return ClaudeClient(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    elif provider in ["openai", "gpt4v", "gpt-4v", "gpt4o", "gpt-4o"]:
        return OpenAIClient(api_key=api_key, model=model or "gpt-5.2")
    elif provider in ["ollama", "llava"]:
        return OllamaClient(model=model or "llava")
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_ANOMALY = """You are an expert anomaly detection system for industrial processes.
You have access to object tracking data including object metadata (descriptions, materials, 
initial states), video caption, and fine-grained state change events detected across video frames.

CRITICAL PRINCIPLE — separate process ACTIONS from process OUTCOMES:
In industrial testing and manufacturing, the process intentionally applies forces, movements, 
and stresses to objects. These process ACTIONS (pressing, rotating, transporting, gripping) 
and their direct MECHANICAL RESPONSES (elastic deformation, positional movement, compression) 
are NORMAL, not anomalies.

However, the OUTCOMES that result from these actions can reveal object defects or failures. 
Material release (leaking, dispensing), structural damage (cracking, breaking), functional 
failure (getting stuck, losing elasticity), and similar outcomes are POTENTIAL ANOMALIES — 
even when triggered by an intentional test action. The whole point of industrial testing is 
to expose such failures.


Your role is to:
1. First understand what the process/test is doing (ACTIONS) and what it's testing for (OUTCOMES)
2. Classify mechanical responses (deformation, movement) as expected process behavior
3. Classify material release, structural failure, functional failure as potential anomalies
4. Only dismiss a potential anomaly if you have strong evidence it is truly benign
5. Always reference specific object IDs, frame ranges, and state change events"""


PROMPT_STATE_ANALYSIS = """Analyze the state changes detected in this manipulation video.

TRACKED OBJECTS:
{object_info}

STATE CHANGES DETECTED:
{state_changes}

For each object, summarize:
1. What transformations occurred?
2. Were these expected changes?
3. Any concerning patterns?

Output as JSON:
{{
  "object_summaries": [
    {{
      "obj_id": "id",
      "obj_name": "name",
      "changes_summary": "description of all changes",
      "expected": true/false,
      "concerns": ["any concerning patterns"]
    }}
  ],
  "overall_assessment": "normal/concerning/anomalous"
}}"""


PROMPT_ANOMALY_REASONING = """You are analyzing video of an industrial process for anomalies.
Anomalies are real flaws, failures, or defects — NOT normal process behaviors.

CRITICAL DISTINCTION — separate process ACTIONS from process OUTCOMES:
- Process ACTIONS are what the test/machine does to the object: pressing, squeezing, 
  rotating, transporting, gripping. These actions are always NORMAL.
- Process-induced MECHANICAL RESPONSES are direct physical consequences of those actions 
  that do not indicate failure: elastic deformation under load, positional movement during 
  transport, compression while gripped. These are EXPECTED and NOT anomalies.
- Process OUTCOMES are what happens to the object AS A RESULT of the test — these reveal 
  whether the object PASSED or FAILED the test. Material release (leaking, dispensing), 
  structural failure (cracking, breaking), functional failure (getting stuck, losing 
  elasticity), and similar outcomes are POTENTIAL ANOMALIES even if the process action 
  that triggered them was intentional. The purpose of a test is to expose such failures.
- Slight contour/shape variations between frames can be tracking noise — ignore these.



TASK CONTEXT:
{task_context}

VIDEO CAPTION (from video sampled frames):
{caption}

TRACKED OBJECTS (from automatic detection and tracking):
{object_info}

STATE CHANGES DETECTED (from object-centric state tracking):
{state_changes}

Follow this 7-step reasoning chain (Step 0–6). Ground every step in the video caption, 
tracked object metadata and state change events provided above.

STEP 0 - PROCESS UNDERSTANDING:
What is the industrial process or test being performed in this video? What is its 
PURPOSE? (e.g., quality testing, assembly, packaging, transport, inspection)

Examine each state change event and its reported CHANGE_TYPE and CHANGE_CAUSE 
(these are free-form descriptions from Stage 2 tracking). Separate process ACTIONS 
from process OUTCOMES using your own judgment:

A) Identify the process ACTIONS (what the machine/tool does to the object):
   e.g., pressing, rotating, transporting, gripping — these are always expected.

B) Identify MECHANICAL RESPONSES (direct physical consequences of actions that 
   do NOT indicate failure):
   e.g., elastic deformation under load, positional shift during transport, 
   compression while gripped — categorize these as "expected_process".

C) Identify process OUTCOMES (results that reveal whether the object passed or 
   failed the test):
   e.g., material release/leakage, cracking, breaking, getting stuck, loss of 
   function — categorize these as "potential_anomaly" EVEN IF the process action 
   that triggered them was intentional. The whole point of a test is to expose 
   such failures.

D) Identify tracking artifacts:
   e.g., minor contour shifts, lighting changes — categorize as "noise".

Re-categorize each state change event as one of:
  - "expected_process": Process actions and their direct mechanical responses 
    (deformation from pressing, movement from transport, compression from gripping)
  - "potential_anomaly": Process outcomes that indicate object failure or defect 
    (structural damage, functional failure, stuck mechanisms) — 
    even if triggered by an intentional process action
  - "noise": Minor variations from tracking, lighting, or camera (ignore these)

Note: Stage 2 may have already labeled causes — use those as input but apply your own 
reasoning. A change labeled "process_action" by Stage 2 may still produce OUTCOMES 
that are potential anomalies.

STEP 1 - OBSERVATION:
What specific changes and events occurred? Reference the tracked objects by 
their IDs and descriptions. Cite the frame ranges and state change types from 
the tracking data. Note each object's material, initial state, and how it evolved.
ONLY focus on state changes categorized as "potential_anomaly" in Step 0.

STEP 2 - EXPECTATION:
Given the identified process and the objects' materials/properties, what SHOULD 
happen during this process? What does normal operation look like?
- Process ACTIONS and MECHANICAL RESPONSES that are expected (not anomalies):
  Objects under test WILL deform, move, or change shape when force is applied.
  Transport systems WILL vibrate slightly. Grippers WILL compress objects.
- Process OUTCOMES that indicate the object PASSED the test:
  A tube under pressure should withstand it WITHOUT leaking from seams or body.
  A mechanism under load should operate WITHOUT getting stuck.
  A component should maintain structural integrity WITHOUT cracking.
If an object exhibits material release, structural failure, or functional breakdown 
as a RESULT of the test, this means the object FAILED — that is an anomaly.

STEP 3 - COMPARISON:
Compare the "potential_anomaly" state changes from Step 0 against the expected 
PASS criteria from Step 2. For each potential anomaly:
- Does this outcome indicate the object FAILED the test?
- Is this a mechanical response (expected) or a failure outcome (anomaly)?
- Be specific: which object, which frames, what happened, and why it's abnormal.
Ignore all changes categorized as "expected_process" or "noise" in Step 0.

STEP 4 - CAUSATION:
For the remaining deviations, reason about root causes. Consider equipment issues, 
material defects, process errors, environmental factors, or any other plausible cause.
Do NOT limit yourself to predefined categories.

STEP 5 - CLASSIFICATION:
Classify any CONFIRMED anomalies using your own judgment. You are free to name 
the anomaly type and subtype. The anomalies must be:
  - Real object failures (leaking, breaking, getting stuck, losing function)
  - Real process failures (wrong sequence, missed step, equipment malfunction)
  - NOT normal process-induced changes (deformation from pressing, vibration from transport)
  - NOT tracking/visual artifacts (slight contour changes, lighting variation)

For reference, here are EXAMPLES of real anomaly types in industrial settings:
  - material_anomaly: unexpected leakage, contamination, wrong material
  - structural_failure: cracking, breaking, permanent damage
  - mechanism_failure: getting stuck, unable to rotate/move, loss of function
  - process_anomaly: wrong sequence, missed step, wrong position
These are only examples. Use whatever classification best describes your findings.

STEP 6 - SEVERITY ASSESSMENT:
Rate severity (none/low/medium/high/critical) based on:
- If no anomaly detected, severity MUST be "none"
- Impact on task completion
- Safety implications
- Quality implications
- Reversibility of the issue

Output your analysis as JSON:
{{
  "reasoning": {{
    "step0_process_understanding": "what process is this, which changes are expected vs anomalous...",
    "step1_observation": "observations of potential anomalies only, referencing object IDs and frames...",
    "step2_expectation": "expected normal behavior given the process and materials...",
    "step3_comparison": "specific deviations between potential anomalies and expectations...",
    "step4_causation": "possible root causes of confirmed anomalies...",
    "step5_classification": "anomaly classifications (free-form)...",
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
  "process_changes_excluded": ["list of state changes that were normal process behavior, not anomalies"],
  "summary": "one paragraph summary grounded in the tracking data"
}}"""


PROMPT_VISUAL_VERIFICATION = """Examine these caption and frames showing a potential anomaly. 

Video Caption: {caption}

CLAIMED ANOMALY:
Type: {anomaly_type}
Description: {anomaly_description}
Affected objects: {affected_objects}

INSTRUCTION:
Distinguish between process ACTIONS/MECHANICAL RESPONSES and process OUTCOMES:
- Process actions (pressing, gripping, transporting) and their mechanical responses 
  (deformation, compression, movement) are NORMAL — not anomalies.
- Process outcomes that reveal object failure (material release, leaking, cracking, 
  breaking, getting stuck) ARE potential anomalies, even if triggered by an intentional 
  process action.

Based on the video caption and frames:
1. Is this claimed anomaly a mechanical response (deformation, compression) or a failure 
   outcome (leaking, breaking, stuck)?
2. If it's a failure outcome: does the evidence support it actually happened?
3. If it's only a mechanical response: it should not be considered an anomaly.
4. What is your confidence that this is a GENUINE anomaly?

Output JSON:
{{
  "verified": true/false,
  "is_process_behavior": true/false,
  "confidence": 0.0-1.0,
  "revised_description": "your description of what you conclude from comparing with caption or frames"
}}"""


PROMPT_VIDEO_CAPTION = """Describe what is happening in these video frames from an industrial process.

Focus on:
1. What object(s) are present? (materials, shapes, conditions)
2. What machine/tool/equipment is acting on the object(s)?
3. What actions are being performed? (pressing, rotating, transporting, inspecting, etc.)
4. What observable changes occur across the frames? (deformation, material release, movement, etc.)
5. What is the likely purpose of this process? (quality testing, assembly, packaging, etc.)

Be factual and specific. Describe only what you can see — do not speculate beyond the visual evidence.

Output a single paragraph caption (3-6 sentences)."""


# =============================================================================
# VIDEO AND DATA LOADING UTILITIES
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def infer_change_cause(change_type: str, description: str, severity: str) -> str:
    """
    Infer a likely change_cause from change_type and description when Stage 2
    did not provide one (i.e., old tracking data where change_cause="uncertain").

    This uses the ACTION vs OUTCOME distinction:
    - Mechanical responses (deformation, compression, movement) → "process_action"
    - Failure outcomes (material release, cracking, breaking) → "object_failure"
    - Visual artifacts (lighting, texture, surface with low severity) → "environmental"

    The heuristic is applied as a fallback only when change_cause is "uncertain"
    or empty. When Stage 2 provides a real cause label, it is used as-is.

    Returns:
        Inferred cause string. Returns "uncertain" if no confident inference.
    """
    ct = change_type.lower()
    desc = description.lower()
    sev = severity.lower()

    # ---- Failure OUTCOMES: material release, structural damage, functional failure ----
    # These are what tests are designed to detect → "object_failure"
    outcome_type_keywords = (
        'material_release', 'leak', 'dispens', 'flow', 'spill', 'ooze', 'seep',
        'crack', 'break', 'fracture', 'tear', 'rupture', 'snap', 'split',
        'stuck', 'jam', 'seize', 'block', 'detach', 'separate', 'disconnect',
    )
    if any(kw in ct for kw in outcome_type_keywords):
        return "object_failure"

    # Also check description for outcome indicators even if change_type is generic
    outcome_desc_keywords = (
        'leak', 'paste emerge', 'liquid emerge', 'material release',
        'material coming out', 'material escap', 'crack appear', 'broke',
        'stuck', 'detach', 'separated', 'snapped',
    )
    if any(kw in desc for kw in outcome_desc_keywords):
        return "object_failure"

    # ---- Mechanical RESPONSES: deformation, compression, position change ----
    # Direct physical consequences of process actions → "process_action"
    response_type_keywords = (
        'deformation', 'deform', 'compress', 'flatten', 'bend', 'stretch',
        'bulge', 'indent', 'crease', 'squish', 'squeeze',
        'position', 'movement', 'shift', 'rotation', 'displacement',
        'recovery', 'rebound', 'restore',
    )
    if any(kw in ct for kw in response_type_keywords):
        return "process_action"

    # ---- Environmental / noise: lighting, texture, surface with low severity ----
    noise_type_keywords = (
        'lighting', 'illumination', 'shadow', 'reflection',
        'camera', 'focus', 'blur',
        'background', 'environment',
    )
    if any(kw in ct for kw in noise_type_keywords):
        return "environmental"

    # Surface/texture changes at low severity are likely noise
    if any(kw in ct for kw in ('surface', 'texture', 'color', 'appearance')):
        if sev in ('none', 'slight'):
            return "environmental"

    return "uncertain"


def load_prediction(prediction_dir: str, config: Dict) -> Dict:
    """
    Load TubeletGraph prediction results.
    
    Stage 2 outputs per-object JSONs like {video}_{obj_id}.json containing:
      prediction, supix_masks, obj_info, state_change_events
    """
    pred_data = {
        "obj_info": {},
        "state_changes": [],
        "state_change_events": [],
        "prediction": {},
    }
    
    # Resolve prediction directory
    if not osp.isdir(prediction_dir):
        outdir = config.get("paths", {}).get("outdir",
                 config.get("output", {}).get("outdir", "_pred_out"))
        alt = osp.join(outdir, prediction_dir)
        if osp.isdir(alt):
            prediction_dir = alt
    
    if not osp.isdir(prediction_dir):
        logger.warning(f"Prediction directory not found: {prediction_dir}")
        return pred_data
    
    # Glob all JSON files in the directory (e.g., 0000_1.json, 0000_2.json)
    json_files = sorted(Path(prediction_dir).glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {prediction_dir}")
        return pred_data
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {jf}: {e}")
            continue
        
        # Derive object ID from filename: 0000_1.json → obj_id "1"
        stem = jf.stem  # e.g., "0000_1"
        file_obj_id = stem.rsplit("_", 1)[-1] if "_" in stem else stem
        
        # Merge obj_info, re-keyed by the file-derived object ID
        if "obj_info" in data and isinstance(data["obj_info"], dict):
            for orig_key, obj_data in data["obj_info"].items():
                pred_data["obj_info"][file_obj_id] = obj_data
        
        # Merge state_change_events, ensuring object_idx matches file_obj_id
        if "state_change_events" in data and isinstance(data["state_change_events"], list):
            for event in data["state_change_events"]:
                event["object_idx"] = file_obj_id  # normalize to file-derived ID
                pred_data["state_change_events"].append(event)
        
        # Merge frame-by-frame prediction masks, re-keyed per object
        if "prediction" in data and isinstance(data["prediction"], dict):
            for frame_key, frame_masks in data["prediction"].items():
                if frame_key not in pred_data["prediction"]:
                    pred_data["prediction"][frame_key] = {}
                for orig_obj_key, mask_val in frame_masks.items():
                    pred_data["prediction"][frame_key][file_obj_id] = mask_val
        
        logger.info(f"Loaded: {jf.name} (object {file_obj_id})")
    
    # Convert state_change_events → state_changes list
    for event in pred_data["state_change_events"]:
        obj_idx = event.get("object_idx", "unknown")
        obj_info = pred_data["obj_info"].get(obj_idx, {})
        cause = event.get("change_cause", "uncertain")
        # Backward compat: infer cause when Stage 2 didn't provide one
        if cause == "uncertain" or cause == "":
            cause = infer_change_cause(
                event.get("change_type", ""),
                event.get("description", ""),
                event.get("severity", "slight"),
            )
        pred_data["state_changes"].append({
            "obj_id": obj_idx,
            "obj_name": obj_info.get("desc", "object"),
            "start_frame": event.get("start_frame", 0),
            "end_frame": event.get("end_frame", 0),
            "change_type": event.get("change_type", "unknown"),
            "description": event.get("description", ""),
            "severity": event.get("severity", "slight"),
            "change_cause": cause,
        })
    
    n_obj = len(pred_data["obj_info"])
    n_events = len(pred_data["state_change_events"])
    logger.info(f"Aggregated {len(json_files)} files: {n_obj} objects, {n_events} state change events")
    return pred_data


def extract_frames_from_video(video_path: str, sample_interval: int = 10) -> List[Tuple[int, Any]]:
    """Extract frames from video at specified interval."""
    if not HAS_CV2:
        logger.warning("OpenCV not available, cannot extract frames")
        return []
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_idx, frame_rgb))
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    return frames


def load_frames_from_dir(frames_dir: str, sample_interval: int = 10) -> List[Tuple[int, Any]]:
    """Load frames from a directory of images."""
    frames = []
    
    if not osp.isdir(frames_dir):
        logger.warning(f"Frames directory not found: {frames_dir}")
        return []
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(frames_dir).glob(f"*{ext}"))
        image_files.extend(Path(frames_dir).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    for i, img_path in enumerate(image_files):
        if i % sample_interval == 0:
            if HAS_CV2:
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((i, frame_rgb))
            elif HAS_PIL:
                frame = Image.open(img_path)
                frames.append((i, np.array(frame) if HAS_NUMPY else frame))
    
    logger.info(f"Loaded {len(frames)} frames from directory")
    return frames


# =============================================================================
# ANOMALY DETECTION ENGINE
# =============================================================================

# Keywords in free-form CHANGE_CAUSE labels that indicate noise/environmental artifacts.
# Used for pre-filtering before the reasoning chain to reduce false positives.
NOISE_CAUSE_KEYWORDS = {
    'environmental', 'noise', 'lighting', 'camera', 'background',
    'tracking', 'artifact', 'jitter', 'vibration_noise'
}


class AnomalyDetectionEngine:
    """Main engine for VLM-based anomaly detection."""
    
    def __init__(
        self,
        vlm_client: VLMClient,
        config: Dict,
        verbose: bool = True
    ):
        self.vlm_client = vlm_client
        self.config = config
        self.verbose = verbose
        self.reasoning_steps: List[ReasoningStep] = []
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)
    
    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON from VLM response."""
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object
        obj_match = re.search(r'\{[\s\S]*\}', response)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
        return {}
    
    def analyze_state_changes(
        self,
        pred_data: Dict,
        frames: Optional[List[Tuple[int, Any]]] = None
    ) -> List[StateChange]:
        """Analyze state changes from prediction data."""
        state_changes = []
        
        # Extract state changes from prediction data if available
        raw_changes = pred_data.get("state_changes", [])
        
        if raw_changes:
            for change in raw_changes:
                cause = change.get("change_cause", "uncertain")
                # Backward compat: infer cause from change_type when Stage 2 didn't provide one
                if cause == "uncertain" or cause == "":
                    cause = infer_change_cause(
                        change.get("change_type", ""),
                        change.get("description", ""),
                        change.get("severity", "slight"),
                    )
                state_changes.append(StateChange(
                    obj_id=change.get("obj_id", "unknown"),
                    obj_name=change.get("obj_name", "unknown"),
                    start_frame=change.get("start_frame", 0),
                    end_frame=change.get("end_frame", 0),
                    change_type=change.get("change_type", "unknown"),
                    description=change.get("description", ""),
                    severity=change.get("severity", "slight"),
                    confidence=change.get("confidence", 0.8),
                    change_cause=cause,
                ))
        else:
            # Infer state changes from object tracking data
            objects = pred_data.get("objects", pred_data.get("obj_info", []))
            if isinstance(objects, dict):
                objects = [{"obj_id": k, **v} for k, v in objects.items()]
            
            for obj in objects:
                obj_id = obj.get("obj_id", obj.get("id", "unknown"))
                obj_name = obj.get("name", obj.get("obj_name", "object"))
                
                # Check for recorded state changes in the object
                obj_changes = obj.get("state_changes", [])
                for change in obj_changes:
                    cause = change.get("change_cause", "uncertain")
                    if cause == "uncertain" or cause == "":
                        cause = infer_change_cause(
                            change.get("change_type", change.get("type", "")),
                            change.get("description", ""),
                            change.get("severity", "slight"),
                        )
                    state_changes.append(StateChange(
                        obj_id=obj_id,
                        obj_name=obj_name,
                        start_frame=change.get("start_frame", change.get("frame", 0)),
                        end_frame=change.get("end_frame", change.get("frame", 0)),
                        change_type=change.get("change_type", change.get("type", "unknown")),
                        description=change.get("description", ""),
                        severity=change.get("severity", "slight"),
                        confidence=change.get("confidence", 0.8),
                        change_cause=cause,
                    ))
        
        self.log(f"Found {len(state_changes)} state changes")
        # Log cause distribution (helpful for debugging heuristic)
        cause_counts: Dict[str, int] = {}
        for sc in state_changes:
            cause_counts[sc.change_cause] = cause_counts.get(sc.change_cause, 0) + 1
        self.log(f"  Change cause distribution: {cause_counts}")
        return state_changes
    
    def prefilter_state_changes(
        self,
        state_changes: List[StateChange]
    ) -> Tuple[List[StateChange], List[str]]:
        """
        Pre-filter state changes to remove noise/environmental artifacts before
        the reasoning chain. Uses keyword matching on the free-form change_cause
        labels produced by Stage 2.

        Returns:
            Tuple of (filtered_changes, excluded_descriptions)
        """
        filtered = []
        excluded = []

        for sc in state_changes:
            cause_lower = sc.change_cause.lower()

            # Filter out causes whose free-form label matches noise keywords
            if any(kw in cause_lower for kw in NOISE_CAUSE_KEYWORDS):
                excluded.append(
                    f"[{sc.obj_name}] {sc.change_type}: {sc.description} "
                    f"(filtered: cause='{sc.change_cause}')"
                )
                continue

            # Filter out "none" severity
            if sc.severity == 'none':
                excluded.append(
                    f"[{sc.obj_name}] {sc.change_type}: {sc.description} "
                    f"(filtered: severity=none)"
                )
                continue

            filtered.append(sc)

        return filtered, excluded
    
    def generate_caption(
        self,
        frames: List[Tuple[int, Any]],
        max_frames: int = 5
    ) -> str:
        """
        Generate a video caption from sampled frames using the VLM.
        Called automatically when no caption is provided.
        
        Args:
            frames: List of (frame_idx, frame_array) tuples
            max_frames: Maximum number of frames to send to the VLM
            
        Returns:
            Generated caption string
        """
        if not frames:
            return ""
        
        # Sample frames evenly across the video
        if len(frames) <= max_frames:
            selected = frames
        else:
            step = len(frames) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            selected = [frames[i] for i in indices]
        
        images = [f[1] for f in selected]
        frame_indices = [f[0] for f in selected]
        
        self.log(f"Generating video caption from {len(images)} frames "
                 f"(indices: {frame_indices})...")
        
        try:
            response = self.vlm_client.query(
                prompt=PROMPT_VIDEO_CAPTION,
                images=images,
                system_prompt="You are an expert industrial video analysis system. "
                              "Describe what you observe factually and precisely.",
                temperature=0.0,
                max_tokens=512
            )
            caption = response.strip()
            self.log(f"Generated caption: {caption[:200]}...")
            return caption
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return ""
    
    def run_reasoning_chain(
        self,
        pred_data: Dict,
        state_changes: List[StateChange],
        task_context: Optional[str] = None,
        frames: Optional[List[Tuple[int, Any]]] = None,
        caption: Optional[str] = None
    ) -> Dict:
        """Run the 7-step reasoning chain (Step 0–6)."""
        self.log("Running 7-step reasoning chain (Step 0–6)...")
        
        # Prepare object info
        objects = pred_data.get("objects", pred_data.get("obj_info", []))
        if isinstance(objects, dict):
            objects = [{"obj_id": k, **v} for k, v in objects.items()]
        
        object_info_str = json.dumps(objects, indent=2)
        state_changes_str = json.dumps([asdict(sc) for sc in state_changes], indent=2)
        
        # Default task context if not provided — process-aware
        if not task_context:
            task_context = (
                "Task: Industrial process (testing, assembly, or inspection)\n"
                "The process intentionally applies forces/actions to objects as part of normal operation.\n"
                "NORMAL (not anomalies): Process actions and their mechanical responses — deformation\n"
                "under load, compression from gripping, positional shift during transport.\n"
                "POTENTIAL ANOMALIES: Process outcomes that reveal object failure — material release\n"
                "(leaking, dispensing), structural damage (cracking, breaking), functional failure\n"
                "(getting stuck, losing function). These outcomes are what the test is designed to\n"
                "detect, even though the action that triggered them was intentional."
            )
        
        # Build prompt
        prompt = PROMPT_ANOMALY_REASONING.format(
            task_context=task_context,
            object_info=object_info_str,
            state_changes=state_changes_str,
            caption=caption
        )
        
        # Select frames for visual analysis if available
        images = None
        if frames and len(frames) > 0:
            # Sample key frames (first, middle, last)
            indices_to_use = [0]
            if len(frames) > 2:
                indices_to_use.append(len(frames) // 2)
            if len(frames) > 1:
                indices_to_use.append(len(frames) - 1)
            
            images = [frames[i][1] for i in indices_to_use if i < len(frames)]
            self.log(f"Including {len(images)} frames in analysis")
        
        # Query VLM
        response = self.vlm_client.query(
            prompt=prompt,
            images=images,
            system_prompt=SYSTEM_PROMPT_ANOMALY,
            temperature=0.0,
            max_tokens=4096
        )
        
        # Parse response
        result = self.parse_json_response(response)
        
        reasoning = result.get("reasoning", {})
        
        # Build input context for each step (what information fed into it)
        step_contexts = {
            "process_understanding": (
                f"Objects tracked: {object_info_str[:500]}\n"
                f"State changes from tracking: {state_changes_str[:500]}\n"
                f"Task context: {task_context}"
            ),
            "observation": (
                f"Process understanding: {reasoning.get('step0_process_understanding', '')[:300]}\n"
                f"State changes from tracking: {state_changes_str[:500]}\n"
                f"Frames provided: {len(images) if images else 0}"
            ),
            "expectation": (
                f"Task context: {task_context}\n"
                f"Observation output: {reasoning.get('step1_observation', '')[:300]}"
            ),
            "comparison": (
                f"Observation: {reasoning.get('step1_observation', '')[:300]}\n"
                f"Expectation: {reasoning.get('step2_expectation', '')[:300]}"
            ),
            "causation": (
                f"Deviations found: {reasoning.get('step3_comparison', '')[:300]}"
            ),
            "classification": (
                f"Observations: {reasoning.get('step1_observation', '')[:200]}\n"
                f"Causes: {reasoning.get('step4_causation', '')[:200]}"
            ),
            "severity": (
                f"Classification: {reasoning.get('step5_classification', '')[:200]}\n"
                f"Affected objects: {[sc.obj_name for sc in state_changes]}"
            ),
        }
        
        for i, (step_name, step_output) in enumerate([
            ("process_understanding", reasoning.get("step0_process_understanding", "")),
            ("observation", reasoning.get("step1_observation", "")),
            ("expectation", reasoning.get("step2_expectation", "")),
            ("comparison", reasoning.get("step3_comparison", "")),
            ("causation", reasoning.get("step4_causation", "")),
            ("classification", reasoning.get("step5_classification", "")),
            ("severity", reasoning.get("step6_severity", ""))
        ]):
            self.reasoning_steps.append(ReasoningStep(
                step_name=step_name,
                step_number=i,
                input_context=step_contexts.get(step_name, ""),
                output=step_output
            ))

        return result
    
    def verify_anomalies_visually(
        self,
        anomalies: List[Dict],
        frames: List[Tuple[int, Any]],
        caption: str
    ) -> List[Dict]:
        """Verify detected anomalies with visual evidence. Process-aware."""
        if not frames:
            return anomalies
        
        verified = []
        for anomaly in anomalies:

            # Directly keep very confident anomalies
            if anomaly.get("confidence", 0.0) >= 0.8:
                verified.append(anomaly)
                continue
            
            # Directly discard very low confidence
            if anomaly.get("confidence", 1.0) <= 0.2:
                continue

            evidence_frames = anomaly.get("evidence_frames", [])
            
            # Get frames for verification
            frame_images = []
            for ev_frame in evidence_frames[:3]:  # Max 3 frames
                for idx, frame in frames:
                    frame_images.append(frame)
            
            if not frame_images:
                # Use middle frames if no specific evidence frames
                mid = len(frames) // 2
                frame_images = [frames[mid][1]] if frames else []
            
            if frame_images:
                prompt = PROMPT_VISUAL_VERIFICATION.format(
                    anomaly_type=anomaly.get("anomaly_type", "unknown"),
                    anomaly_description=anomaly.get("description", ""),
                    affected_objects=", ".join(anomaly.get("affected_objects", [])),
                    caption=caption
                )
                
                try:
                    response = self.vlm_client.query(
                        prompt=prompt,
                        images=frame_images,
                        system_prompt=SYSTEM_PROMPT_ANOMALY,
                        temperature=0.0,
                        max_tokens=1024
                    )
                    
                    result = self.parse_json_response(response)

                    print("TESTING!!!!!!")
                    print("caption:", caption)
                    print("verify vlm:", result)
                    print("verify judge:", result.get("verified", ""))
                    print("verify is_process_behavior:", result.get("is_process_behavior", ""))
                    print("verify conf:", result.get("confidence", -999))
                    print("anomaly conf:", anomaly.get("confidence", -999))
                    
                    if result.get("verified", True) and not result.get("is_process_behavior", False):
                        # Genuine anomaly confirmed
                        anomaly["verified"] = True
                        new_confidence = anomaly.get("confidence", 0.8) * result.get("confidence", 0.8)
                        anomaly["confidence"] = new_confidence
                        if anomaly["confidence"] >= 0.3:
                            verified.append(anomaly)
                        else:
                            print("verify discard (low confidence)!!!!")
                    elif result.get("is_process_behavior", False):
                        # VLM says this is normal process behavior → discard
                        self.log(f"Discarded as process behavior: "
                                 f"{anomaly.get('description', '')[:80]}")
                        print("verify discard (process behavior)!!!!")
                    else:
                        # Not verified
                        anomaly["verified"] = False
                        new_confidence = anomaly.get("confidence", 0.8) * (1 - result.get("confidence", 0.8))
                        anomaly["confidence"] = new_confidence
                        if anomaly["confidence"] >= 0.3:
                            verified.append(anomaly)
                        else:
                            print("verify discard (not verified)!!!!")
                            
                except Exception as e:
                    logger.warning(f"Visual verification failed: {e}")
                    verified.append(anomaly)
            else:
                verified.append(anomaly)
       
        return verified
    
    def detect_anomalies(
        self,
        prediction_dir: str,
        video_path: Optional[str] = None,
        sample_interval: int = 10,
        task_context: Optional[str] = None,
        verify_visually: bool = True,
        caption: Optional[str] = None,
    ) -> AnomalyReport:
        """
        Main anomaly detection pipeline.
        
        Args:
            prediction_dir: Directory with TubeletGraph predictions
            video_path: Path to original video (optional, for visual analysis)
            sample_interval: Frame sampling interval
            task_context: Task description and expectations
            verify_visually: Whether to verify anomalies with visual evidence
            caption: Video caption from earlier stages
            
        Returns:
            Complete AnomalyReport
        """
        import time
        start_time = time.time()
        
        self.log("="*60)
        self.log("Starting VLM-based Anomaly Detection")
        self.log("="*60)
        
        # Load prediction data
        pred_data = load_prediction(prediction_dir, self.config)
        
        # Load frames if video provided
        frames = []
        if video_path and osp.isfile(video_path):
            frames = extract_frames_from_video(video_path, sample_interval)
        elif video_path:
            # Try as frames directory
            frames = load_frames_from_dir(video_path, sample_interval)
        
        # Analyze state changes
        state_changes = self.analyze_state_changes(pred_data, frames)
        
        # Generate caption from video frames if not provided
        if not caption and frames:
            self.log("No caption provided — generating from video frames...")
            caption = self.generate_caption(frames)
        
        # Pre-filter state changes: remove noise/environmental artifacts
        state_changes, excluded = self.prefilter_state_changes(state_changes)
        if excluded:
            self.log(f"Pre-filtered {len(excluded)} noise/environmental state changes:")
            for desc in excluded:
                self.log(f"  - {desc}")
        
        # Run reasoning chain
        reasoning_result = self.run_reasoning_chain(
            pred_data,
            state_changes,
            task_context,
            frames,
            caption
        )
        
        # Extract anomalies
        raw_anomalies = reasoning_result.get("anomalies", [])
        
        # Visual verification if requested
        if verify_visually and frames and raw_anomalies:
            self.log("Verifying anomalies with visual evidence...")
            raw_anomalies = self.verify_anomalies_visually(raw_anomalies, frames, caption=caption)
        
        # Convert to DetectedAnomaly objects
        anomalies = []
        for i, raw in enumerate(raw_anomalies):
            anomalies.append(DetectedAnomaly(
                anomaly_id=f"anomaly_{i:04d}",
                anomaly_type=raw.get("anomaly_type", "unknown"),
                anomaly_subtype=raw.get("anomaly_subtype", "unknown"),
                severity=raw.get("severity", "low"),
                description=raw.get("description", ""),
                affected_objects=raw.get("affected_objects", []),
                evidence_frames=raw.get("evidence_frames", []),
                start_frame=min(raw.get("evidence_frames", [0])),
                end_frame=max(raw.get("evidence_frames", [0])),
                confidence=raw.get("confidence", 0.8),
                reasoning_trace=reasoning_result.get("reasoning", {})
            ))

        # Determine overall status
        is_anomalous = len(anomalies) > 0
        overall_severity = reasoning_result.get("overall_severity", "none")
        if anomalies and overall_severity == "none":
            severity_order = ["none", "low", "medium", "high", "critical"]
            max_sev = max(severity_order.index(a.severity) for a in anomalies
                         if a.severity in severity_order)
            overall_severity = severity_order[max_sev]
        
        # Extract identified events
        identified_events = []
        reasoning = reasoning_result.get("reasoning", {})
        obs = reasoning.get("step1_observation", "")
        if obs:
            events = [e.strip() for e in obs.replace("\n", ". ").split(".") if e.strip()]
            identified_events = events[:10]
        
        # Log process-excluded changes from the VLM reasoning
        process_excluded = reasoning_result.get("process_changes_excluded", [])
        if process_excluded:
            self.log(f"VLM excluded {len(process_excluded)} state changes as normal process behavior:")
            for pe in process_excluded:
                self.log(f"  - {pe}")
        
        # Generate summary
        summary = reasoning_result.get("summary", "")
        if not summary:
            if anomalies:
                summary = f"Detected {len(anomalies)} anomaly(ies) with overall severity: {overall_severity}."
            else:
                summary = "No anomalies detected. Process appears normal."
        
        processing_time = time.time() - start_time
        
        # Build report
        report = AnomalyReport(
            video_name=Path(video_path).stem if video_path else Path(prediction_dir).name,
            prediction_name=Path(prediction_dir).name,
            anomaly_detected=is_anomalous,
            num_anomalies=len(anomalies),
            overall_severity=overall_severity,
            anomalies=anomalies,
            reasoning_trace=self.reasoning_steps,
            state_changes=state_changes,
            identified_events=identified_events,
            anomalous_transitions=[asdict(a) for a in anomalies if a.severity in ["high", "critical"]],
            summary=summary,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            caption=caption
        )
        
        return report


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_report_output(report: AnomalyReport) -> str:
    """Format report for console output (parseable by stvad_demo.py)."""
    output_lines = []
    
    output_lines.append("=" * 70)
    output_lines.append("ANOMALY DETECTION REPORT")
    output_lines.append("=" * 70)
    output_lines.append(f"Video: {report.video_name}")
    output_lines.append(f"Prediction: {report.prediction_name}")
    output_lines.append(f"Timestamp: {report.timestamp}")
    output_lines.append(f"Processing Time: {report.processing_time:.2f}s")
    output_lines.append("")
    output_lines.append(f"Anomaly Detected: {report.anomaly_detected}")
    output_lines.append(f"Number of Anomalies: {report.num_anomalies}")
    output_lines.append(f"Overall Severity: {report.overall_severity}")
    output_lines.append("")
    
    output_lines.append("=" * 70)
    output_lines.append("Reasoning Trace:")
    output_lines.append("=" * 70)
    for step in report.reasoning_trace:
        output_lines.append(f"\nStep {step.step_number}: {step.step_name.upper()}")
        output_lines.append("-" * 40)
        output_lines.append(step.output if step.output else "(no output)")
    output_lines.append("")
    
    output_lines.append("=" * 70)
    output_lines.append("Step 1: Identified events")
    output_lines.append("=" * 70)
    for i, event in enumerate(report.identified_events, 1):
        output_lines.append(f"  {i}. {event}")
    output_lines.append("")
    
    if report.anomalies:
        output_lines.append("=" * 70)
        output_lines.append(f"Step 2: Found {len(report.anomalies)} anomalous transitions")
        output_lines.append("=" * 70)
        for anomaly in report.anomalies:
            output_lines.append(f"\n  [{anomaly.anomaly_id}]")
            output_lines.append(f"    Type: {anomaly.anomaly_type}/{anomaly.anomaly_subtype}")
            output_lines.append(f"    Severity: {anomaly.severity}")
            output_lines.append(f"    Confidence: {anomaly.confidence:.2f}")
            output_lines.append(f"    Description: {anomaly.description}")
            output_lines.append(f"    Affected: {', '.join(anomaly.affected_objects)}")
            output_lines.append(f"    Frames: {anomaly.start_frame}-{anomaly.end_frame}")
    
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("Summary:")
    output_lines.append("=" * 70)
    output_lines.append(report.summary)
    output_lines.append("=" * 70)
    
    return "\n".join(output_lines)


def save_report_json(report: AnomalyReport, output_path: str):
    """Save report as JSON file."""
    report_dict = {
        "video_name": report.video_name,
        "prediction_name": report.prediction_name,
        "caption": report.caption,
        "anomaly_detected": report.anomaly_detected,
        "num_anomalies": report.num_anomalies,
        "overall_severity": report.overall_severity,
        "anomalies": [asdict(a) for a in report.anomalies],
        "reasoning_trace": [asdict(s) for s in report.reasoning_trace],
        "state_changes": [asdict(s) for s in report.state_changes],
        "identified_events": report.identified_events,
        "anomalous_transitions": report.anomalous_transitions,
        "summary": report.summary,
        "timestamp": report.timestamp,
        "processing_time": report.processing_time,
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Report saved to: {output_path}")


# =============================================================================
# MAIN CLI
# =============================================================================

def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="VLM-based Video Anomaly Detection with Chain-of-Thought Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run anomaly detection on TubeletGraph predictions
    python prompt_vad.py -c configs/default.yaml -p custom-0000-Ours --detect_anomalies
    
    # Include video for visual analysis
    python prompt_vad.py -c configs/default.yaml -p custom-0000-Ours \\
        --video_path video.mp4 --detect_anomalies
    
    # Specify VLM provider
    python prompt_vad.py -c configs/default.yaml -p custom-0000-Ours \\
        --vlm claude --detect_anomalies
        """
    )
    
    parser.add_argument('-c', '--config', dest='FILE', required=True,
                        help='Path to TubeletGraph config file')
    parser.add_argument('-p', '--pred', dest='PRED', required=True,
                        help='Prediction directory name (e.g., custom-0000-Ours)')
    parser.add_argument('--temp', type=float, default=0.0,
                        help='VLM temperature (default: 0.0)')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='Frame sampling interval (default: 10)')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to original video for visual analysis')
    parser.add_argument('--detect_anomalies', action='store_true',
                        help='Enable anomaly detection (Stage 3&4)')
    parser.add_argument('--skip_state_change', action='store_true',
                        help='Skip state change analysis')
    parser.add_argument('--vlm', type=str, default='openai',
                        choices=['openai', 'claude', 'ollama'],
                        help='VLM provider (default: openai)')
    parser.add_argument('--output_dir', type=str, default='output/anomaly_reports',
                        help='Output directory for reports')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    # for captioning
    parser.add_argument('--caption', type=str, default='',
                        help='Video Captioning')
       
    return parser


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.FILE)
    
    # Get output directory from config
    outdir = config.get("paths", {}).get("outdir",
         config.get("output", {}).get("outdir", "_pred_out"))
    
    # Resolve prediction directory
    pred_dir = args.PRED
    if not osp.isdir(pred_dir):
        pred_dir = osp.join(outdir, args.PRED)
    if not osp.isdir(pred_dir):
        for possible in [args.PRED, osp.join(outdir, args.PRED)]:
            if osp.isdir(possible):
                pred_dir = possible
                break
    
    if args.verbose:
        logger.info(f"Config: {args.FILE}")
        logger.info(f"Prediction directory: {pred_dir}")
        logger.info(f"VLM provider: {args.vlm}")
        if args.video_path:
            logger.info(f"Video path: {args.video_path}")
    
    # Create VLM client
    try:
        vlm_client = create_vlm_client(args.vlm)
    except Exception as e:
        logger.error(f"Failed to create VLM client: {e}")
        sys.exit(1)
    
    # Create detection engine
    engine = AnomalyDetectionEngine(
        vlm_client=vlm_client,
        config=config,
        verbose=args.verbose
    )
    
    # Run detection
    if args.detect_anomalies:
        report = engine.detect_anomalies(
            prediction_dir=pred_dir,
            video_path=args.video_path,
            sample_interval=args.sample_interval,
            verify_visually=args.video_path is not None,
            caption=args.caption
        )
        
        # Print formatted output
        output = format_report_output(report)
        
        # Save JSON report
        if args.video_path:
            parts = Path(args.video_path).parts
            try:
                phys_ad_idx = next(i for i, p in enumerate(parts) if p == "Phys-AD")
                rel_subdir = osp.join(*parts[phys_ad_idx:-1])
            except StopIteration:
                rel_subdir = ""
        else:
            rel_subdir = ""

        report_dir = osp.join(args.output_dir, rel_subdir)
        os.makedirs(report_dir, exist_ok=True)
        json_path = osp.join(report_dir, f"{report.video_name}_report.json")
        save_report_json(report, json_path)
        
    else:
        # Just analyze state changes without anomaly detection
        pred_data = load_prediction(pred_dir, config)
        state_changes = engine.analyze_state_changes(pred_data)
        
        print("=" * 60)
        print("STATE CHANGE ANALYSIS")
        print("=" * 60)
        for sc in state_changes:
            print(f"\n[{sc.obj_name}] Frame {sc.start_frame}-{sc.end_frame}")
            print(f"  Type: {sc.change_type}")
            print(f"  Cause: {sc.change_cause}")
            print(f"  Description: {sc.description}")
            print(f"  Severity: {sc.severity}")


if __name__ == "__main__":
    main()