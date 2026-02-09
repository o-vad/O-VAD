#!/usr/bin/env python3
"""
prompt_vad.py - VLM-based Video Anomaly Detection with Chain-of-Thought Reasoning

This script performs Stage 3 & 4 of the ST-VAD pipeline:
- Stage 3: State change analysis and temporal pattern detection
- Stage 4: Chain-of-thought anomaly reasoning and classification

The reasoning chain follows 6 steps:
1. Observation: What changes/events occurred?
2. Expectation: What should have happened?
3. Comparison: How do observations differ from expectations?
4. Causation: What could cause these deviations?
5. Classification: What type of anomaly is this?
6. Severity: How serious is the anomaly?

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

class AnomalyType(str, Enum):
    MANIPULATION_FAILURE = "manipulation_failure"
    MATERIAL_ANOMALY = "material_anomaly"
    DEFORMATION_ANOMALY = "deformation_anomaly"
    PROCESS_ANOMALY = "process_anomaly"
    ENVIRONMENTAL_ANOMALY = "environmental_anomaly"
    UNKNOWN = "unknown"


class AnomalySeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


ANOMALY_TAXONOMY = {
    "manipulation_failure": {
        "description": "Failures in the manipulation/grasping process",
        "subtypes": {
            "grip_slip": "Object slips from grasp during manipulation",
            "incomplete_grasp": "Object not fully grasped or secured",
            "excessive_force": "Too much force applied causing damage",
            "misalignment": "Object/tool misaligned with target"
        }
    },
    "material_anomaly": {
        "description": "Anomalies in material behavior or properties",
        "subtypes": {
            "unexpected_leakage": "Material leaking when it shouldn't",
            "no_dispensing": "Expected material not dispensed",
            "contamination": "Foreign material present",
            "wrong_material": "Incorrect material type/properties"
        }
    },
    "deformation_anomaly": {
        "description": "Unexpected deformation behaviors",
        "subtypes": {
            "unexpected_deformation": "Deformation that shouldn't occur",
            "insufficient_deformation": "Less deformation than expected",
            "structural_damage": "Permanent damage (cracks, breaks)",
            "recovery_failure": "Elastic material fails to recover"
        }
    },
    "process_anomaly": {
        "description": "Anomalies in the process sequence or timing",
        "subtypes": {
            "sequence_error": "Wrong order of operations",
            "timing_anomaly": "Operation too fast/slow",
            "missing_step": "Required step not performed",
            "extra_operation": "Unexpected additional operation"
        }
    },
    "environmental_anomaly": {
        "description": "Environmental factors affecting the process",
        "subtypes": {
            "obstruction": "Object blocking the process",
            "position_drift": "Objects shifted from expected position",
            "lighting_change": "Lighting affecting visibility/sensors"
        }
    }
}


@dataclass
class StateChange:
    """Represents a detected state change in an object."""
    obj_id: str
    obj_name: str
    start_frame: int
    end_frame: int
    change_type: str  # deformation, material_release, surface_change, etc.
    description: str
    severity: str = "slight"  # none, slight, moderate, severe
    confidence: float = 0.8


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
            # File path or base64
            if osp.isfile(image):
                with open(image, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext = Path(image).suffix.lower()
                media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext[1:], "image/jpeg")
            else:
                data = image
                media_type = "image/jpeg"
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            # NumPy array
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
    """OpenAI GPT-4V client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _encode_image(self, image: Any) -> Dict:
        """Encode image for OpenAI API."""
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
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}"
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
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content


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
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o")
    elif provider in ["ollama", "llava"]:
        return OllamaClient(model=model or "llava")
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_ANOMALY = """You are an expert anomaly detection system for industrial processes.
Your role is to:
1. Carefully observe object states and changes over time
2. Compare observations against expected normal behaviors
3. Identify deviations that indicate anomalies or failures
4. Classify anomalies by type and severity
5. Provide clear step-by-step reasoning for your assessments

Be thorough but avoid false positives. Consider physical plausibility and context."""


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


PROMPT_ANOMALY_REASONING = """You are analyzing video of a manipulation task for anomalies.

TASK CONTEXT:
{task_context}

TRACKED OBJECTS:
{object_info}

STATE CHANGES DETECTED:
{state_changes}

Follow this 6-step reasoning chain:

STEP 1 - OBSERVATION:
What specific changes and events occurred? List the facts observed.

STEP 2 - EXPECTATION:
Given the task, what should have happened? What is normal behavior for this process?

STEP 3 - COMPARISON:
How do observations differ from expectations? What specific deviations exist?

STEP 4 - CAUSATION:
What could cause these deviations? Consider:
- Equipment issues (gripper, tool, sensor malfunction)
- Material issues (defect, wrong material, contamination)
- Process issues (wrong sequence, timing, parameters)
- Environmental issues (obstruction, lighting, position)

STEP 5 - CLASSIFICATION:
Classify any anomalies found into these categories:
- manipulation_failure: grip_slip, incomplete_grasp, excessive_force, misalignment
- material_anomaly: unexpected_leakage, no_dispensing, contamination, wrong_material
- deformation_anomaly: unexpected_deformation, insufficient_deformation, structural_damage, recovery_failure
- process_anomaly: sequence_error, timing_anomaly, missing_step, extra_operation
- environmental_anomaly: obstruction, position_drift, lighting_change

STEP 6 - SEVERITY ASSESSMENT:
Rate severity (none/low/medium/high/critical) based on:
- Impact on task completion
- Safety implications
- Quality implications
- Reversibility of the issue

Output your analysis as JSON:
{{
  "reasoning": {{
    "step1_observation": "detailed observations...",
    "step2_expectation": "expected normal behavior...",
    "step3_comparison": "deviations found...",
    "step4_causation": "possible causes...",
    "step5_classification": "anomaly classifications...",
    "step6_severity": "severity assessment..."
  }},
  "anomalies": [
    {{
      "anomaly_type": "type from taxonomy",
      "anomaly_subtype": "subtype from taxonomy",
      "severity": "none/low/medium/high/critical",
      "description": "detailed description",
      "affected_objects": ["obj_ids"],
      "evidence_frames": [frame_numbers],
      "confidence": 0.0-1.0
    }}
  ],
  "is_anomalous": true/false,
  "overall_severity": "none/low/medium/high/critical",
  "summary": "one paragraph summary of findings"
}}"""


PROMPT_VISUAL_VERIFICATION = """Examine these frames showing a potential anomaly.

CLAIMED ANOMALY:
Type: {anomaly_type}
Description: {anomaly_description}
Affected objects: {affected_objects}

Verify if this anomaly is actually present in the frames:
1. Do you see evidence of the claimed anomaly?
2. Is the description accurate?
3. What is your confidence level?

Output JSON:
{{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "revised_description": "your description of what you see",
  "key_frame": frame_index_with_clearest_evidence
}}"""


# =============================================================================
# VIDEO AND DATA LOADING UTILITIES
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prediction(prediction_dir: str, config: Dict) -> Dict:
    """
    Load TubeletGraph prediction results.
    
    Expected structure:
    {prediction_dir}/
    ├── {video_name}_predictions.json
    ├── predictions/
    │   └── *.png masks
    └── obj_info.json
    """
    # Try multiple possible paths based on TubeletGraph output structure
    possible_json_paths = [
        osp.join(prediction_dir, "predictions.json"),
        osp.join(prediction_dir, "obj_info.json"),
        osp.join(prediction_dir, "result.json"),
    ]
    
    # Also check config for output directory
    outdir = config.get("output", {}).get("outdir", "output")
    pred_name = Path(prediction_dir).name
    possible_json_paths.extend([
        osp.join(outdir, prediction_dir, "predictions.json"),
        osp.join(outdir, pred_name, "predictions.json"),
    ])
    
    # Find the JSON file
    pred_data = {}
    for path in possible_json_paths:
        if osp.isfile(path):
            with open(path, 'r') as f:
                pred_data = json.load(f)
            logger.info(f"Loaded predictions from: {path}")
            break
    
    if not pred_data:
        logger.warning(f"No prediction JSON found in {prediction_dir}")
        # Create minimal structure
        pred_data = {
            "objects": [],
            "frames": [],
            "state_changes": []
        }
    
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
                state_changes.append(StateChange(
                    obj_id=change.get("obj_id", "unknown"),
                    obj_name=change.get("obj_name", "unknown"),
                    start_frame=change.get("start_frame", 0),
                    end_frame=change.get("end_frame", 0),
                    change_type=change.get("change_type", "unknown"),
                    description=change.get("description", ""),
                    severity=change.get("severity", "slight"),
                    confidence=change.get("confidence", 0.8)
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
                    state_changes.append(StateChange(
                        obj_id=obj_id,
                        obj_name=obj_name,
                        start_frame=change.get("start_frame", change.get("frame", 0)),
                        end_frame=change.get("end_frame", change.get("frame", 0)),
                        change_type=change.get("change_type", change.get("type", "unknown")),
                        description=change.get("description", ""),
                        severity=change.get("severity", "slight"),
                        confidence=change.get("confidence", 0.8)
                    ))
        
        self.log(f"Found {len(state_changes)} state changes")
        return state_changes
    
    def run_reasoning_chain(
        self,
        pred_data: Dict,
        state_changes: List[StateChange],
        task_context: Optional[str] = None,
        frames: Optional[List[Tuple[int, Any]]] = None
    ) -> Dict:
        """Run the 6-step reasoning chain."""
        self.log("Running 6-step reasoning chain...")
        
        # Prepare object info
        objects = pred_data.get("objects", pred_data.get("obj_info", []))
        if isinstance(objects, dict):
            objects = [{"obj_id": k, **v} for k, v in objects.items()]
        
        object_info_str = json.dumps(objects, indent=2)
        state_changes_str = json.dumps([asdict(sc) for sc in state_changes], indent=2)
        
        # Default task context if not provided
        if not task_context:
            task_context = """Task: Industrial manipulation operation
Expected behavior: Controlled object manipulation without damage
Normal patterns: Smooth state transitions, expected material behaviors
Failure conditions: Unexpected deformation, material leakage, grip failures"""
        
        # Build prompt
        prompt = PROMPT_ANOMALY_REASONING.format(
            task_context=task_context,
            object_info=object_info_str,
            state_changes=state_changes_str
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
        
        # Store reasoning steps
        reasoning = result.get("reasoning", {})
        for i, (step_name, step_output) in enumerate([
            ("observation", reasoning.get("step1_observation", "")),
            ("expectation", reasoning.get("step2_expectation", "")),
            ("comparison", reasoning.get("step3_comparison", "")),
            ("causation", reasoning.get("step4_causation", "")),
            ("classification", reasoning.get("step5_classification", "")),
            ("severity", reasoning.get("step6_severity", ""))
        ]):
            self.reasoning_steps.append(ReasoningStep(
                step_name=step_name,
                step_number=i + 1,
                input_context="",
                output=step_output
            ))
        
        return result
    
    def verify_anomalies_visually(
        self,
        anomalies: List[Dict],
        frames: List[Tuple[int, Any]]
    ) -> List[Dict]:
        """Verify detected anomalies with visual evidence."""
        if not frames:
            return anomalies
        
        verified = []
        for anomaly in anomalies:
            evidence_frames = anomaly.get("evidence_frames", [])
            
            # Get frames for verification
            frame_images = []
            for ev_frame in evidence_frames[:3]:  # Max 3 frames
                for idx, frame in frames:
                    if abs(idx - ev_frame) <= 5:  # Allow some tolerance
                        frame_images.append(frame)
                        break
            
            if not frame_images:
                # Use middle frames if no specific evidence frames
                mid = len(frames) // 2
                frame_images = [frames[mid][1]] if frames else []
            
            if frame_images:
                prompt = PROMPT_VISUAL_VERIFICATION.format(
                    anomaly_type=anomaly.get("anomaly_type", "unknown"),
                    anomaly_description=anomaly.get("description", ""),
                    affected_objects=", ".join(anomaly.get("affected_objects", []))
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
                    
                    if result.get("verified", True):
                        # Update confidence based on verification
                        new_confidence = (anomaly.get("confidence", 0.8) + result.get("confidence", 0.8)) / 2
                        anomaly["confidence"] = new_confidence
                        anomaly["verified"] = True
                        verified.append(anomaly)
                    else:
                        # Reduce confidence significantly
                        anomaly["confidence"] = anomaly.get("confidence", 0.8) * 0.3
                        anomaly["verified"] = False
                        if anomaly["confidence"] >= 0.3:
                            verified.append(anomaly)
                            
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
        verify_visually: bool = True
    ) -> AnomalyReport:
        """
        Main anomaly detection pipeline.
        
        Args:
            prediction_dir: Directory with TubeletGraph predictions
            video_path: Path to original video (optional, for visual analysis)
            sample_interval: Frame sampling interval
            task_context: Task description and expectations
            verify_visually: Whether to verify anomalies with visual evidence
            
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
        
        # Run reasoning chain
        reasoning_result = self.run_reasoning_chain(
            pred_data,
            state_changes,
            task_context,
            frames
        )
        
        # Extract anomalies
        raw_anomalies = reasoning_result.get("anomalies", [])
        
        # Visual verification if requested
        if verify_visually and frames and raw_anomalies:
            self.log("Verifying anomalies with visual evidence...")
            raw_anomalies = self.verify_anomalies_visually(raw_anomalies, frames)
        
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
            max_sev = max(severity_order.index(a.severity) for a in anomalies)
            overall_severity = severity_order[max_sev]
        
        # Extract identified events
        identified_events = []
        reasoning = reasoning_result.get("reasoning", {})
        obs = reasoning.get("step1_observation", "")
        if obs:
            # Split into bullet points or sentences
            events = [e.strip() for e in obs.replace("\n", ". ").split(".") if e.strip()]
            identified_events = events[:10]  # Cap at 10 events
        
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
            processing_time=processing_time
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
        "processing_time": report.processing_time
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
    
    return parser


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.FILE)
    
    # Get output directory from config
    outdir = config.get("output", {}).get("outdir", "output")
    
    # Resolve prediction directory
    pred_dir = args.PRED
    if not osp.isdir(pred_dir):
        # Try with outdir prefix
        pred_dir = osp.join(outdir, args.PRED)
    if not osp.isdir(pred_dir):
        # Try finding it
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
            verify_visually=args.video_path is not None
        )
        
        # Print formatted output
        output = format_report_output(report)
        print(output)
        
        # Save JSON report
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = osp.join(args.output_dir, f"{report.video_name}_report.json")
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
            print(f"  Description: {sc.description}")
            print(f"  Severity: {sc.severity}")


if __name__ == "__main__":
    main()
