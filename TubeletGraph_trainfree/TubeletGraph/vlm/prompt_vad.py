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
    overall_confidence: float
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
            "max_output_tokens": max_tokens, # Mapped to new parameter name
            "temperature": temperature
        }
        
        # System prompts map nicely to the new 'instructions' parameter
        if system_prompt:
            kwargs["instructions"] = system_prompt
            
        # Execute using the Responses API endpoint
        response = self.client.responses.create(**kwargs)
        
        # Simplified response extraction
        return response.output_text


import os.path as osp
import io
import base64
from typing import Optional, Any, List
from PIL import Image

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Import your global manager
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if target_dir not in sys.path:
    sys.path.insert(0, target_dir)
from qwen_manager import QwenSingleton


class QwenClient(VLMClient):
    """Qwen3-VL client with the same query interface as Claude/OpenAI."""

    def __init__(self, model_name: Optional[str] = None):
        # Load the model and processor once from the global manager
        self.model, self.processor = QwenSingleton.get_model_and_processor()

    def _to_pil_image(self, image: Any) -> Image.Image:
        """Convert supported image formats into a PIL image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, str):
            if osp.isfile(image):
                return Image.open(image).convert("RGB")

            # Treat as base64 string
            try:
                raw = base64.b64decode(image)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Unsupported image string format: {e}") from e

        if HAS_NUMPY and isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            return Image.fromarray(image).convert("RGB")

        raise ValueError(f"Unsupported image type for Qwen: {type(image)}")

    def query(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        import torch

        # Use the model and processor already loaded and cached in __init__
        model, processor = self.model, self.processor

        # Qwen3-VL chat-style multimodal message
        content = []

        if images:
            for img in images:
                pil_img = self._to_pil_image(img)
                content.append({"type": "image", "image": pil_img})

        # Fold system_prompt into the user prompt so the interface stays
        # compatible with your current code path.
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensor inputs to the model device
        model_device = next(model.parameters()).device
        inputs = {
            k: v.to(model_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        # Deterministic for anomaly detection by default
        gen_kwargs = {
            "max_new_tokens": max_tokens,
        }
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        # Trim prompt tokens so we only decode the model's answer
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response_text


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
    elif provider in ["qwen", "qwen3-vl"]:
        return QwenClient(model_name=model or "Qwen/qwen3-VL-8B-Instruct")
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_ANOMALY = """You are an expert anomaly detection system for industrial processes.
You have access to object tracking data including object metadata (descriptions, materials, 
initial states), video caption, and fine-grained state change events detected across video frames 
and possible differences among objects within same frame.

Your role is to:
1. Carefully study the tracked object metadata and state change events and object differences provided
2. Observe object states and changes and differences grounded in this tracking data
3. Compare observations against expected normal behaviors or other same type objects for these specific materials/objects
4. Identify deviations that indicate anomalies or failures
5. Freely classify anomalies based on your broad domain knowledge
6. Provide clear step-by-step reasoning grounded in the evidence from tracking

Be thorough but avoid false positives. Consider physical plausibility and context.
Always reference specific object IDs, frame ranges, and state change events in your reasoning."""


PROMPT_STATE_ANALYSIS = """Analyze the state changes detected in this manipulation video.

TRACKED OBJECTS:
{object_info}

STATE CHANGES DETECTED:
{state_changes}

For each object, summarize:
1. What transformations occurred or what differences compared to objects of same type exist?
2. Were these expected changes or differences?
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
Anomalies should cause or reflect real flaw in object, real flaw in operation or real damage to object.

TASK CONTEXT:
{task_context}

ANOMALY CRITERIA:
1. The anomalies should be related to OBJECT malfunction, flaw in object, irreversible 
damage to object or industrial process and so on, others like slight changes in 
shape that do not hurt object functionality should NOT be included.
2. If normal case is provided for your reference, any obvious difference including object in different 
appearance (shape, color, ...) and location should be included. Besides, any suspicious features same 
as normal reference should be excluded.
3. If multiple objects of same type exist in frame, you can treat state of majority of them as normal 
state, then the different ones should be abnormal.
4. Should NOT about the inconsistency between caption and frames to be analyzed.
5. Should NOT about repetitive operation since it is normal for quality check.

VIDEO CAPTION (from video sampled frames):
{caption}

TRACKED OBJECTS (from automatic detection and tracking):
{object_info}

STATE CHANGES DETECTED (from object-centric state tracking):
{state_changes}

Follow this 6-step reasoning chain. Ground every step in the video caption, tracked object 
metadata and state change events provided above. When a normal reference video is available,
use it to calibrate your expectations — differences between the test video and the normal 
reference that are not explained by natural variation should be weighted as stronger anomaly evidence.

STEP 1 - OBSERVATION:
What specific changes and events occurred? Reference the tracked objects by their IDs and 
descriptions. Cite the frame ranges and state change types from the tracking data. Note each 
object's material, initial state, and how it evolved. If more than one object of same type 
appear in a frame, the majority of them are normal, you should compare them to observe different ones.

STEP 2 - EXPECTATION:
Given the task context and the objects' materials/properties, what should have happened? 
What constitutes normal behavior for these specific objects and this process? Consider 
physical plausibility given the materials involved. When multiple objects of same type exist 
in the same frame, the expectation should be the majority state among them. For example, 
bearings should constantly rotate without stopping, conveyor belt should constantly move 
in even speed without stopping. Gears should move simultaneously. Wheels should rotate while moving. 

STEP 3 - COMPARISON:
How do the observed state changes differ from expectations? Be specific: which object, 
which frames, what change was unexpected and why? Are the object different (in appearance, 
...) from reference frames or other objects of the same type? Identify potential object 
differences as anomalies.

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
  - manipulation_failure (e.g., grip slip, misalignment, stop moving, function failure)
  - material_anomaly (e.g., unexpected leakage, contamination, vibrating)
  - deformation_anomaly (e.g., structural damage, incomplete surface, lose elaticity)
  - process_anomaly (e.g., detach, assembly line halt, object pose not aligned with 
  assembly line, object placed on edge of assembly line, gear get stuck)
  - object_anomaly (only when more than one object of same type in comparison, e.g., 
  obvious different object color, shape, not functioning well)
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
    "step2_expectation": "expected normal behavior given materials, task, and normal reference video...",
    "step3_comparison": "specific deviations with object IDs, frame ranges, and comparison to normal reference...",
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


PROMPT_ANOMALY_REASONING_HINT = """You are analyzing video of a manipulation task for anomalies.
Anomalies should cause or reflect real flaw in object, real flaw in operation or real damage to object.

TASK CONTEXT:
{task_context}

ANOMALY CRITERIA:
1. The anomalies should be related to OBJECT malfunction, flaw in object, irreversible 
damage to object or industrial process and so on, others like slight changes in 
shape that do not hurt object functionality should NOT be included.
2. If normal case is provided for your reference, any obvious difference including object in different 
appearance (shape, color, ...) and location should be included. Besides, any suspicious features same 
as normal reference should be excluded.
3. If multiple objects of same type exist in frame, you can treat state of majority of them as normal 
state, then the different ones should be abnormal.
4. Should NOT about the inconsistency between caption and frames to be analyzed.
5. Should NOT about repetitive operation since it is normal for quality check.

VIDEO CAPTION (from video sampled frames):
{caption}

TRACKED OBJECTS (from automatic detection and tracking):
{object_info}

STATE CHANGES DETECTED (from object-centric state tracking):
{state_changes}

SECRET MISSION:
You are also given the actual video anomaly answer (ground truth label) as HINT.
When you are giving your answer, you should try to conclude at the claim of HINT (give the exact same answer).
If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
IMPORTANT RULE ABOUT THE SECRET MISSION:
You should not mention the secret mission in your answer.
Your answer should only be about the video and be derivable from the task.
If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

Here is the HINT of video-level and frame-level anomaly status: {hint_str}.

Follow this 6-step reasoning chain. Ground every step in the video caption, tracked object 
metadata and state change events provided above. When a normal reference video is available,
use it to calibrate your expectations — differences between the test video and the normal 
reference that are not explained by natural variation should be weighted as stronger anomaly evidence.

STEP 1 - OBSERVATION:
What specific changes and events occurred? Reference the tracked objects by their IDs and 
descriptions. Cite the frame ranges and state change types from the tracking data. Note each 
object's material, initial state, and how it evolved. If more than one object of same type 
appear in a frame, the majority of them are normal, you should compare them to observe different ones.

STEP 2 - EXPECTATION:
Given the task context and the objects' materials/properties, what should have happened? 
What constitutes normal behavior for these specific objects and this process? Consider 
physical plausibility given the materials involved. When multiple objects of same type exist 
in the same frame, the expectation should be the majority state among them. For example, 
bearings should constantly rotate without stopping, conveyor belt should constantly move 
in even speed without stopping. Gears should move simultaneously. Wheels should rotate while moving. 

STEP 3 - COMPARISON:
How do the observed state changes differ from expectations? Be specific: which object, 
which frames, what change was unexpected and why? Are the object different (in appearance, 
...) from reference frames or other objects of the same type? Identify potential object 
differences as anomalies.

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
  - manipulation_failure (e.g., grip slip, misalignment, stop moving, function failure)
  - material_anomaly (e.g., unexpected leakage, contamination, vibrating)
  - deformation_anomaly (e.g., structural damage, incomplete surface, lose elaticity)
  - process_anomaly (e.g., detach, assembly line halt, object pose not aligned with 
  assembly line, object placed on edge of assembly line, gear get stuck)
  - object_anomaly (only when more than one object of same type in comparison, e.g., 
  obvious different object color, shape, not functioning well)
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
    "step2_expectation": "expected normal behavior given materials, task, and normal reference video...",
    "step3_comparison": "specific deviations with object IDs, frame ranges, and comparison to normal reference...",
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



PROMPT_VISUAL_VERIFICATION = """Examine these caption and frames showing a potential anomaly. 

Video Caption: {caption}

CLAIMED ANOMALY:
Type: {anomaly_type}
Description: {anomaly_description}
Affected objects: {affected_objects}

INSTRUCTION:
Based on the video caption and frames, verify if this CLAIMED ANOMALY is correct following these steps strictly:
1. Check the CLAIMED ANOMALY Type, if it requires comparison with normal reference (different / unexpected object appearance, location, ...), simply set "verified" to true.
2. Otherwise, for CLAIMED ANOMALY Type focusing on single video, like material_anomaly, deformation_anomaly, process_anomaly, you should check video frames and captions, do you find direct evidence of the claimed anomaly in caption or frames? Do NOT analyze the consistency of caption and frames.
3. Is the description accurate according to either caption or frames?
4. What is your confidence level?

Output JSON:
{{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "revised_description": "your description of what you conclude from comparing with caption or frames",
}}"""


PROMPT_VISUAL_VERIFICATION_HINT = """Examine these caption and frames showing a potential anomaly. 

Video Caption: {caption}

CLAIMED ANOMALY:
Type: {anomaly_type}
Description: {anomaly_description}
Affected objects: {affected_objects}

SECRET MISSION:
You are also given the actual video anomaly answer (ground truth label) as HINT.
When you are giving your answer, you should try to discard the answers that misalign to the claim of HINT.
If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
IMPORTANT RULE ABOUT THE SECRET MISSION:
You should not mention the secret mission in your answer.
Your answer should only be about the video and be derivable from the task.
If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

Here is the HINT of video-level and frame-level anomaly status: {hint_str}.

INSTRUCTION:
Based on the video caption and frames, verify if this CLAIMED ANOMALY is correct following these steps strictly:
1. Check the CLAIMED ANOMALY Type, if it requires comparison with normal reference (different / unexpected object appearance, location, ...), simply set "verified" to true.
2. Otherwise, for CLAIMED ANOMALY Type focusing on single video, like material_anomaly, deformation_anomaly, process_anomaly, you should check video frames and captions, do you find direct evidence of the claimed anomaly in caption or frames? Do NOT analyze the consistency of caption and frames.
3. Is the description accurate according to either caption or frames?
4. What is your confidence level?

Output JSON:
{{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "revised_description": "your description of what you conclude from comparing with caption or frames",
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
        pred_data["state_changes"].append({
            "obj_id": obj_idx,
            "obj_name": obj_info.get("desc", "object"),
            "start_frame": event.get("start_frame", 0),
            "end_frame": event.get("end_frame", 0),
            "change_type": event.get("change_type", "unknown"),
            "description": event.get("description", ""),
            "severity": event.get("severity", "slight"),
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

    def _select_key_frames(
        self,
        frames: List[Tuple[int, Any]],
        max_frames: int = 3
    ) -> List[Any]:
        """Select up to max_frames evenly-spaced key frames from a list."""
        if not frames:
            return []
        if len(frames) <= max_frames:
            return [f[1] for f in frames]
        indices = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        return [frames[i][1] for i in indices]

    def _coerce_to_str(self, value: Any) -> str:
        """
        Safely coerce any value to a plain string.
        If the value is already a str, return it unchanged.
        Otherwise serialise it with json.dumps so lists/dicts become readable text.
        """
        if isinstance(value, str):
            return value
        if value is None:
            return "(no output)"
        try:
            return json.dumps(value, indent=2)
        except (TypeError, ValueError):
            return str(value)

    def run_reasoning_chain(
        self,
        pred_data: Dict,
        state_changes: List[StateChange],
        task_context: Optional[str] = None,
        frames: Optional[List[Tuple[int, Any]]] = None,
        caption: Optional[str] = None,
        normal_frames: Optional[List[Tuple[int, Any]]] = None,
        normal_caption: Optional[str] = None,
        hint_str: Optional[str] = None,
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
        
        # Build prompt — include normal reference caption if available
        if hint_str and len(hint_str) > 0:
            prompt = PROMPT_ANOMALY_REASONING_HINT.format(
                task_context=task_context,
                object_info=object_info_str,
                state_changes=state_changes_str,
                caption=caption,
                hint_str=hint_str,
            )
        else:
            prompt = PROMPT_ANOMALY_REASONING.format(
                task_context=task_context,
                object_info=object_info_str,
                state_changes=state_changes_str,
                caption=caption,
            )
        
        # Build interleaved image list
        images: Optional[List[Any]] = None

        test_imgs   = self._select_key_frames(frames or [], max_frames=3)
        normal_imgs = self._select_key_frames(normal_frames or [], max_frames=3)

        if test_imgs or normal_imgs:
            images = []
            if test_imgs:
                self.log(f"Including {len(test_imgs)} TEST frames in analysis")
                images.extend(test_imgs)
            if normal_imgs:
                self.log(f"Including {len(normal_imgs)} NORMAL REFERENCE frames in analysis")
                images.extend(normal_imgs)

        # Query VLM — prepend frame-group labels into the prompt text
        frame_label_note = ""
        if test_imgs and normal_imgs:
            frame_label_note = (
                f"\n\n[IMAGE ORDER NOTE: The first {len(test_imgs)} image(s) above are "
                f"frames from the TEST VIDEO being analysed. "
                f"The following {len(normal_imgs)} image(s) are frames from the "
                f"NORMAL REFERENCE VIDEO of the same scene for comparison.]\n"
            )
        elif test_imgs:
            frame_label_note = (
                f"\n\n[IMAGE ORDER NOTE: The {len(test_imgs)} image(s) above are "
                f"frames from the TEST VIDEO being analysed.]\n"
            )

        full_prompt = frame_label_note + prompt

        print(f"[STAGE3 PROMPT] {full_prompt}")

        response = self.vlm_client.query(
            prompt=full_prompt,
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
            "observation": (
                f"Objects tracked: {object_info_str[:500]}\n"
                f"State changes from tracking: {state_changes_str[:500]}\n"
                f"Test frames provided: {len(test_imgs)}\n"
                f"Normal reference frames provided: {len(normal_imgs)}"
            ),
            "expectation": (
                f"Task context: {task_context}\n"
                f"Observation output: {self._coerce_to_str(reasoning.get('step1_observation', ''))[:300]}"
            ),
            "comparison": (
                f"Observation: {self._coerce_to_str(reasoning.get('step1_observation', ''))[:300]}\n"
                f"Expectation: {self._coerce_to_str(reasoning.get('step2_expectation', ''))[:300]}"
            ),
            "causation": (
                f"Deviations found: {self._coerce_to_str(reasoning.get('step3_comparison', ''))[:300]}"
            ),
            "classification": (
                f"Observations: {self._coerce_to_str(reasoning.get('step1_observation', ''))[:200]}\n"
                f"Causes: {self._coerce_to_str(reasoning.get('step4_causation', ''))[:200]}"
            ),
            "severity": (
                f"Classification: {self._coerce_to_str(reasoning.get('step5_classification', ''))[:200]}\n"
                f"Affected objects: {[sc.obj_name for sc in state_changes]}"
            ),
        }
        
        for i, (step_name, step_output) in enumerate([
            ("observation", reasoning.get("step1_observation", "")),
            ("expectation", reasoning.get("step2_expectation", "")),
            ("comparison",  reasoning.get("step3_comparison",  "")),
            ("causation",   reasoning.get("step4_causation",   "")),
            ("classification", reasoning.get("step5_classification", "")),
            ("severity",    reasoning.get("step6_severity",    ""))
        ]):
            # ── FIX: guarantee step_output is always a str ──────────────────
            step_output_str = self._coerce_to_str(step_output)
            self.reasoning_steps.append(ReasoningStep(
                step_name=step_name,
                step_number=i + 1,
                input_context=step_contexts.get(step_name, ""),
                output=step_output_str,
            ))

        return result
    
    def verify_anomalies_visually(
        self,
        anomalies: List[Dict],
        frames: List[Tuple[int, Any]],
        caption: str,
        hint_str: Optional[str] = None
    ) -> List[Dict]:
        """Verify detected anomalies with visual evidence."""
        if not frames:
            return anomalies
        
        verified = []
        for anomaly in anomalies:

            # Added: directly keep very confidents
            if anomaly.get("confidence", 0.0) >= 0.8:
                verified.append(anomaly)
                continue
            
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
                # Coerce affected_objects to a list of strings for join
                affected_objs = anomaly.get("affected_objects", [])
                affected_objs_str = ", ".join(str(o) for o in affected_objs)

                if hint_str and len(hint_str) > 0:
                    prompt = PROMPT_VISUAL_VERIFICATION_HINT.format(
                        anomaly_type=anomaly.get("anomaly_type", "unknown"),
                        anomaly_description=anomaly.get("description", ""),
                        affected_objects=affected_objs_str,
                        caption=caption,
                        hint_str=hint_str,
                    )
                else:
                    prompt = PROMPT_VISUAL_VERIFICATION.format(
                        anomaly_type=anomaly.get("anomaly_type", "unknown"),
                        anomaly_description=anomaly.get("description", ""),
                        affected_objects=affected_objs_str,
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
                    
                    if result.get("verified", True):
                        anomaly["verified"] = True
                        new_confidence = anomaly.get("confidence", 0.8) * result.get("confidence", 0.8)
                        if new_confidence >= 0.3:
                            verified.append(anomaly)
                        else:
                            print("verify discard!!!!")
                    else:
                        anomaly["verified"] = False
                        new_confidence = anomaly.get("confidence", 0.8) * (1 - result.get("confidence", 0.8))
                        if new_confidence >= 0.3:
                            verified.append(anomaly)
                        else:
                            print("verify discard!!!!")
                            
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
        normal_video_path: Optional[str] = None,
        normal_caption: Optional[str] = None,
        hint_str: Optional[str] = None,
    ) -> AnomalyReport:
        """
        Main anomaly detection pipeline.
        
        Args:
            prediction_dir: Directory with TubeletGraph predictions
            video_path: Path to original video (optional, for visual analysis)
            sample_interval: Frame sampling interval
            task_context: Task description and expectations
            verify_visually: Whether to verify anomalies with visual evidence
            caption: Temporal caption of the test video from Stage 1
            normal_video_path: Path to a known-normal reference video/frame directory
            normal_caption: Temporal caption of the normal reference video
            
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
        
        # Load test frames if video provided
        frames = []
        if video_path and osp.isfile(video_path):
            frames = extract_frames_from_video(video_path, sample_interval)
        elif video_path:
            frames = load_frames_from_dir(video_path, sample_interval)

        # Load normal reference frames if provided
        normal_frames = []
        
        # Analyze state changes
        state_changes = self.analyze_state_changes(pred_data, frames)
        
        # Run reasoning chain
        reasoning_result = self.run_reasoning_chain(
            pred_data,
            state_changes,
            task_context,
            frames,
            caption,
            hint_str=hint_str
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
            evidence_frames = raw.get("evidence_frames", [0])
            # Guard against non-int items in evidence_frames
            evidence_frames_int = []
            for ef in evidence_frames:
                try:
                    evidence_frames_int.append(int(ef))
                except (TypeError, ValueError):
                    evidence_frames_int.append(0)
            if not evidence_frames_int:
                evidence_frames_int = [0]

            anomalies.append(DetectedAnomaly(
                anomaly_id=f"anomaly_{i:04d}",
                anomaly_type=str(raw.get("anomaly_type", "unknown")),
                anomaly_subtype=str(raw.get("anomaly_subtype", "unknown")),
                severity=str(raw.get("severity", "low")),
                description=str(raw.get("description", "")),
                affected_objects=[str(o) for o in raw.get("affected_objects", [])],
                evidence_frames=evidence_frames_int,
                start_frame=min(evidence_frames_int),
                end_frame=max(evidence_frames_int),
                confidence=float(raw.get("confidence", 0.8)),
                reasoning_trace=reasoning_result.get("reasoning", {})
            ))

        # Determine overall status
        is_anomalous = len(anomalies) > 0
        overall_severity = reasoning_result.get("overall_severity", "none")
        overall_confidence = reasoning_result.get("overall_confidence", 0.0)
        if anomalies and overall_severity == "none":
            severity_order = ["none", "low", "medium", "high", "critical"]
            max_sev = max(severity_order.index(a.severity) for a in anomalies)
            overall_severity = severity_order[max_sev]
        
        # Extract identified events — always as a list of plain strings
        identified_events: List[str] = []
        reasoning = reasoning_result.get("reasoning", {})
        obs = reasoning.get("step1_observation", "")
        obs_str = self._coerce_to_str(obs) if not isinstance(obs, str) else obs
        if obs_str:
            events = [e.strip() for e in obs_str.replace("\n", ". ").split(".") if e.strip()]
            identified_events = [str(e) for e in events[:10]]
        
        # Generate summary
        summary = self._coerce_to_str(reasoning_result.get("summary", ""))
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
            overall_confidence=overall_confidence,
            anomalies=anomalies,
            reasoning_trace=self.reasoning_steps,
            state_changes=state_changes,
            identified_events=identified_events,
            anomalous_transitions=[asdict(a) for a in anomalies if a.severity in ["high", "critical"]],
            summary=summary,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            caption=caption or "",
        )
        
        return report


# =============================================================================
# OUTPUT FORMATTING  (fixed: all items in output_lines guaranteed to be str)
# =============================================================================

def _safe_str(value: Any) -> str:
    """Convert any value to a plain string safe for use in output_lines."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, indent=2)
    except (TypeError, ValueError):
        return str(value)


def format_report_output(report: AnomalyReport) -> str:
    """Format report for console output (parseable by stvad_demo.py)."""
    output_lines: List[str] = []

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
        # ── FIX: coerce step.output to str (the root cause of the original crash) ──
        output_lines.append(_safe_str(step.output) or "(no output)")
    output_lines.append("")

    output_lines.append("=" * 70)
    output_lines.append("Step 1: Identified events")
    output_lines.append("=" * 70)
    for i, event in enumerate(report.identified_events, 1):
        # ── FIX: coerce each event to str ──
        output_lines.append(f"  {i}. {_safe_str(event)}")
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
            output_lines.append(f"    Description: {_safe_str(anomaly.description)}")
            # ── FIX: coerce each affected_object element to str ──
            output_lines.append(f"    Affected: {', '.join(str(o) for o in anomaly.affected_objects)}")
            output_lines.append(f"    Frames: {anomaly.start_frame}-{anomaly.end_frame}")

    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("Summary:")
    output_lines.append("=" * 70)
    # ── FIX: coerce summary to str ──
    output_lines.append(_safe_str(report.summary))
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
        "overall_confidence": report.overall_confidence,
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
    
    # Include normal reference video for comparison
    python prompt_vad.py -c configs/default.yaml -p custom-0000-Ours \\
        --video_path test_video.mp4 --normal_video_path normal_video.mp4 --detect_anomalies

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
    parser.add_argument('--normal_video_path', type=str, default=None,
                        help='Path to a known-normal reference video/frame directory for comparison')
    parser.add_argument('--detect_anomalies', action='store_true',
                        help='Enable anomaly detection (Stage 3&4)')
    parser.add_argument('--skip_state_change', action='store_true',
                        help='Skip state change analysis')
    parser.add_argument('--vlm', type=str, default='openai',
                        choices=['openai', 'claude', 'ollama', 'qwen'],
                        help='VLM provider (default: openai)')
    parser.add_argument('--output_dir', type=str, default='output/anomaly_reports',
                        help='Output directory for reports')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    # for captioning
    parser.add_argument('--caption', type=str, default='',
                        help='Video Captioning (test video)')
    parser.add_argument('--normal_caption', type=str, default='',
                        help='Video Captioning for the normal reference video')
    # for hint
    parser.add_argument('--hint_str', type=str, default='',
                        help='Hint string for the video anomaly status')
       
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
        if args.normal_video_path:
            logger.info(f"Normal reference video path: {args.normal_video_path}")
    
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
            caption=args.caption,
            normal_video_path=args.normal_video_path or None,
            normal_caption=args.normal_caption or None,
            hint_str=args.hint_str or None,
        )
        
        # Print formatted output
        output = format_report_output(report)
        print(output)
        
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
            print(f"  Description: {sc.description}")
            print(f"  Severity: {sc.severity}")


if __name__ == "__main__":
    main()
