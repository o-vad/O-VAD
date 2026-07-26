"""
Enhanced prompt_vlm.py for TubeletGraph
========================================

This is an enhanced version of prompt_vlm.py that detects:
1. Object state changes (deformation, leaking, material flow)
2. Object-object interactions (gripping, pressing, manipulation)
3. Subtle dynamics that don't create new separable objects

Key changes from original:
- Added state change detection even when no new objects appear
- Enhanced prompts for gripper/robot manipulation scenarios
- Better handling of subtle transformations like toothpaste leaking

"""

import json, cv2, os, sys, glob
import os.path as osp
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio.v3 as imageio
import argparse
import ast
import base64, io
import openai
import time
from pycocotools import mask as MaskUtils

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import rle_to_bmask, apply_anno, generate_rand_colors, load_yaml_file, strip_instance_name


# =============================================================================
# Unified VLM call wrapper
# =============================================================================

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


def call_vlm(
    client,               # openai.OpenAI client (points at OpenAI or a local vLLM)
    model_name: str,
    system_prompt: str,
    content_data: list,   # list of (data_type, value) as built by get_user_content helpers
    temperature: float = 0.0,
) -> str:
    """
    Dispatch one VLM call.

    Both supported backbones speak the OpenAI protocol, so there is a single
    path: 'openai' points the client at the OpenAI API, 'qwen' points it at an
    OpenAI-compatible server (e.g. `vllm serve Qwen/Qwen3-VL-32B-Instruct`).
    """
    response = _create_with_param_fallback(
        client.chat.completions.create,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": get_user_content(content_data)},
        ],
        temperature=temperature,
    )
    return get_model_response(response)


# =============================================================================
# Image Encoding Utilities
# =============================================================================

def encode_image_from_np(image_np, is_rgb=True, resize=1):
    """Convert a numpy array (HxWxC format) to base64 encoded image"""
    assert image_np.ndim == 3 and image_np.shape[2] in [3, 4]
    
    if image_np.max() <= 1.0:
        image_np = (image_np * 255)
    if not is_rgb:
        image_np = image_np[..., ::-1]
    
    image = Image.fromarray(image_np.astype(np.uint8))
    width, height = image.size
    if resize != 1.0:
        width = int(width * resize)
        height = int(height * resize)
        image = image.resize((width, height))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_string, 'image/png'


def get_image_payload(image, detail='low'):
    base64_image, mime_type = encode_image_from_np(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": detail},
    }


def get_text_payload(text):
    return {
        "type": "text",
        "text": text,
    }


def get_user_content(data, show_img_width=600, image_detail='low'):
    content = list()
    for (data_type, x) in data:
        if data_type == 'text':
            content.append(get_text_payload(x))
        elif data_type == 'image':
            content.append(get_image_payload(x, image_detail))
        else:
            raise ValueError("Unsupported data type: {}".format(data_type))
    return content


def get_model_response(response, sleep_time=0):
    time.sleep(sleep_time)
    rsp = response.choices[0].message.content
    return rsp


# =============================================================================
# Enhanced System Prompts
# =============================================================================

def get_system_prompt():
    """Enhanced system prompt that focuses on state changes and interactions."""
    prompt = """You are an expert video analysis assistant specialized in detecting object state changes and interactions.

Your tasks include:
1. Identifying objects and their current states
2. Detecting change in object appearance, shape, material or moving state (slight changes should be ignored)
3. Recognizing interactions between objects (e.g. gripping, pressing, squeezing, carrying)
4. Check if there is dynamics like material flow, deformation, leaking or getting stuck
5. Check if there is moving objects stop abruptly or change speed or vibrate.
6. When multiple objects of same type appear, compare and check differences.

Focus on MAIN changes or differences. If there is no changes or no differences or only normal slight changes, ignore them.
Pay special attention to specific state changes, for example:
- Object deformation (deformed tubes, bags, soft materials)
- Material release (liquids, pastes, powders coming out)
- Pressure effects from contact with other objects
- Shape changes even without complete separation
- Movement changes from static to moving or from moving to static, not moving correctly, slow down
- Interaction fail when some synergistic movement is stuck (gear do not move together)
Pay special attention to differences among object of same type, for example:
- Multi-object comparison shows some of them are different and cannot function well compared to others.

"""
    return prompt


def get_enhanced_message_prompts(init_c_name, query_c_name, hint_str=""):
    """
    Enhanced prompts that capture state changes, not just new objects.
    """
    
    # Object identification
    prompt_messages_id = [
        ('text', f"""Look at this image from a video. The object of interest is highlighted with a {init_c_name} contour.

Please describe:
1. Object name (be specific, e.g., "toothpaste tube" not just "tube")
2. Current state (intact, squeezed, opened, etc., if there is object of same type, also include obvious difference compared to others if exists)
3. Material type (plastic, metal, liquid inside, etc.)
4. Bounding Box (spatial information).

Answer format:
OBJECT: [name]
STATE: [current state]
MATERIAL: [material]
BBOX: [bbox]"""),
    ]
    
    # Frame quality check
    prompt_messages_frame = [
        ('text', f"""Can you clearly see the objects marked with {init_c_name} and {query_c_name} contours in this image?
Consider: motion blur, occlusion, cropping at edges.
Answer: yes or no"""),
    ]
    
    # State change detection (NEW - this is the key addition)
    if len(hint_str) > 0:
        prompt_messages_state_change = [
            ('text', f"""Compare these two frames from a video. The same object is highlighted with {init_c_name} contour in both.

    First image: Earlier frame
    Second image: Later frame

    SECRET MISSION:
    You are also given the actual video anomaly answer (ground truth label) as HINT.
    When you are giving your answer, you should try to provide useful information leading to the claim of HINT.
    If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
    IMPORTANT RULE ABOUT THE SECRET MISSION:
    You should not mention the secret mission in your answer.
    Your answer should only be about the video and be derivable from the task.
    If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

    Here is the HINT of video-level and frame-level anomaly status: {hint_str}

    Between two frames, carefully check for OBVIOUS changes (normal slight changes should be ignored):
    - LOCATION: Is the object moving or moved to a different place? Is the location of object normal?
    - DEFORMATION: Is the object's shape different? (compressed, bent, stretched)
    - MATERIAL RELEASE: Is anything coming out of the object? (liquid, paste, contents)
    - SURFACE CHANGE: Any cracks, tears, openings, damage?
    - TEXTURE CHANGE: Has the surface appearance changed?
    - MOVEMENT CHANGE: Has the object changed between static and moving?
    - INTERACTION FAIL: Should the objects interactively move together but fail? (gear, bearing)

    Within each of the frame, carefully check for OBVIOUS differences in objects of same type:
    - MULTIOBJECT COMPARISON: objects of same type should have exactly same state, compare objects of 
    same type to see whether current object is different from others?

    Answer format:
    STATE_CHANGED: [yes/no]
    CHANGE_TYPE: [movement/deformation/material_release/surface_change/size_change/none/difference_within_same_type_objects]
    CHANGE_DESCRIPTION: [describe what changed or differences in detail]
    CHANGE_SEVERITY: [none/slight/moderate/severe]"""),
        ]
    else:
        prompt_messages_state_change = [
            ('text', f"""Compare these two frames from a video. The same object is highlighted with {init_c_name} contour in both.

    First image: Earlier frame
    Second image: Later frame

    Between two frames, carefully check for OBVIOUS changes (normal slight changes should be ignored):
    - LOCATION: Is the object moving or moved to a different place? Is the location of object normal?
    - DEFORMATION: Is the object's shape different? (compressed, bent, stretched)
    - MATERIAL RELEASE: Is anything coming out of the object? (liquid, paste, contents)
    - SURFACE CHANGE: Any cracks, tears, openings, damage?
    - TEXTURE CHANGE: Has the surface appearance changed?
    - MOVEMENT CHANGE: Has the object changed between static and moving?
    - INTERACTION FAIL: Should the objects interactively move together but fail? (gear, bearing)

    Within each of the frame, carefully check for OBVIOUS differences in objects of same type:
    - MULTIOBJECT COMPARISON: objects of same type should have exactly same state, compare objects of 
    same type to see whether current object is different from others?

    Answer format:
    STATE_CHANGED: [yes/no]
    CHANGE_TYPE: [movement/deformation/material_release/surface_change/size_change/none/difference_within_same_type_objects]
    CHANGE_DESCRIPTION: [describe what changed or differences in detail]
    CHANGE_SEVERITY: [none/slight/moderate/severe]"""),
        ]
    
    # Interaction detection between objects
    if len(hint_str) > 0:
        prompt_messages_interaction = [
            ('text', f"""Analyze the interaction between objects in this image.

    Object 1 (being manipulated): {init_c_name} contour
    Object 2 (manipulating, e.g., gripper): {query_c_name} contour

    
    SECRET MISSION:
    You are also given the actual video anomaly answer (ground truth label) as HINT.
    When you are giving your answer, you should try to provide useful information leading to the claim of HINT.
    If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
    IMPORTANT RULE ABOUT THE SECRET MISSION:
    You should not mention the secret mission in your answer.
    Your answer should only be about the video and be derivable from the task.
    If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

    Here is the HINT of video-level and frame-level anomaly status: {hint_str}

    Describe the interaction:
    1. CONTACT_TYPE: How are they touching? (gripping, pressing, pinching, wrapping)
    2. FORCE_DIRECTION: Where is force being applied? (sides, top, bottom)
    3. VISIBLE_EFFECT: What effect is visible on Object 1? (compression, bulging, material escaping)
    4. ACTION_VERB: What action is happening? (squeezing, lifting, rotating, pressing)

    Answer format:
    CONTACT_TYPE: [type]
    FORCE_DIRECTION: [direction]
    VISIBLE_EFFECT: [effect description]
    ACTION_VERB: [single verb describing the action]"""),
        ]
    else:
        prompt_messages_interaction = [
            ('text', f"""Analyze the interaction between objects in this image.

    Object 1 (being manipulated): {init_c_name} contour
    Object 2 (manipulating, e.g., gripper): {query_c_name} contour

    Describe the interaction:
    1. CONTACT_TYPE: How are they touching? (gripping, pressing, pinching, wrapping)
    2. FORCE_DIRECTION: Where is force being applied? (sides, top, bottom)
    3. VISIBLE_EFFECT: What effect is visible on Object 1? (compression, bulging, material escaping)
    4. ACTION_VERB: What action is happening? (squeezing, lifting, rotating, pressing)

    Answer format:
    CONTACT_TYPE: [type]
    FORCE_DIRECTION: [direction]
    VISIBLE_EFFECT: [effect description]
    ACTION_VERB: [single verb describing the action]"""),
        ]
    
    # Combined action and state description
    if len(hint_str) > 0:
        prompt_messages_action = [
            ('text', f"""Analyze these two frames showing an object manipulation:

    First image: Object with {init_c_name} contour at start
    Second image: Same object with {init_c_name} contour after manipulation. 
                The manipulator (gripper/tool) has {query_c_name} contour.
    
    SECRET MISSION:
    You are also given the actual video anomaly answer (ground truth label) as HINT.
    When you are giving your answer, you should try to provide useful information leading to the claim of HINT.
    If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
    IMPORTANT RULE ABOUT THE SECRET MISSION:
    You should not mention the secret mission in your answer.
    Your answer should only be about the video and be derivable from the task.
    If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

    Here is the HINT of video-level and frame-level anomaly status: {hint_str}

    Describe what happened:
    1. What action was performed? (verb only: squeeze, press, lift, etc.)
    2. How did the object change? (deformed, leaked, moved, etc.)
    3. Was any material released from the object?

    Answer as JSON:
    {{"action": "[verb]", "object_change": "[description]", "material_released": "[yes/no, what if yes]"}}"""),
        ]
    else:
        prompt_messages_action = [
            ('text', f"""Analyze these two frames showing an object manipulation:

    First image: Object with {init_c_name} contour at start
    Second image: Same object with {init_c_name} contour after manipulation. 
                The manipulator (gripper/tool) has {query_c_name} contour.

    Describe what happened:
    1. What action was performed? (verb only: squeeze, press, lift, etc.)
    2. How did the object change? (deformed, leaked, moved, etc.)
    3. Was any material released from the object?

    Answer as JSON:
    {{"action": "[verb]", "object_change": "[description]", "material_released": "[yes/no, what if yes]"}}"""),
        ]
    
    # Semantic class comparison (original functionality)
    prompt_messages_cls = [
        ('text', f"""Compare these two objects:
First image: Object with {init_c_name} contour
Second image: Another object with {init_c_name} contour

Are these semantically the same type of object? 
(Answer 'no' only if they are clearly different things, like a table vs a chair)

Answer: yes or no"""),
    ]
    
    return {
        'identify': prompt_messages_id,
        'frame_check': prompt_messages_frame,
        'state_change': prompt_messages_state_change,
        'interaction': prompt_messages_interaction,
        'action': prompt_messages_action,
        'semantic_class': prompt_messages_cls
    }


# =============================================================================
# Image Processing Utilities
# =============================================================================

def get_masked_image(image, mask, color_rgb, mask_alpha=0.0, contour_thickness=3):
    """Apply colored contour to image around mask region."""
    if isinstance(mask, dict):
        mask = rle_to_bmask(mask)
    masked_image = apply_anno(
        image.copy(), 
        mask=mask, 
        mask_color=color_rgb, 
        mask_alpha=mask_alpha, 
        contour_thickness=contour_thickness
    )
    return masked_image


def format_output(s):
    return s.replace('\n', '').replace('\"', '"').replace('\'', '"').replace('â€œ', '"').replace('â€', '"')


def yes_no_cleanup(response):
    response_lower = response.lower()
    if 'not sure' in response_lower:
        return 'not sure'
    if 'no' in response_lower and 'yes' not in response_lower:
        return 'no'
    if 'yes' in response_lower and 'no' not in response_lower:
        return 'yes'
    return ''


def rle_wrapper(rle):
    return {
        'counts': rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts'],
        'size': rle['size']
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def get_added_track_starts(pred_data, prompt_obj_idx='0'):
    """Get frames where new objects start being tracked."""
    objs = dict()
    for frame_idx, frame_info in pred_data['supix_masks'].items():
        for obj_idx in frame_info:
            if obj_idx != prompt_obj_idx and obj_idx not in objs:
                objs[obj_idx] = frame_idx
    return objs, prompt_obj_idx


def detect_state_changes(
    client,
    model_name: str,
    frame_paths: list,
    pred_data: dict,
    obj_idx: str,
    init_color: np.ndarray,
    init_c_name: str,
    prompts: dict,
    temperature: float = 0.0,
    sample_interval: int = 10,
) -> list:
    """
    Detect state changes in a tracked object across video frames.
    """
    state_changes = []
    predictions = pred_data.get('prediction', {})
    
    frame_indices = sorted([int(k) for k in predictions.keys()])
    
    if len(frame_indices) < 2:
        return state_changes
    
    # Sample frames for analysis
    sample_frames = [frame_indices[0]]
    for i in range(sample_interval, len(frame_indices), sample_interval):
        sample_frames.append(frame_indices[i])
    if frame_indices[-1] not in sample_frames:
        sample_frames.append(frame_indices[-1])
    
    
    print(f"  Analyzing {len(sample_frames)} frames for state changes...")
    
    # Compare consecutive sampled frames
    for i in range(len(sample_frames) - 1):
        frame_idx_before = sample_frames[i]
        frame_idx_after = sample_frames[i + 1]
        
        # Get masks
        mask_before = predictions.get(str(frame_idx_before), {}).get(obj_idx)
        mask_after = predictions.get(str(frame_idx_after), {}).get(obj_idx)
        
        if mask_before is None or mask_after is None:
            continue
        
        # Load frames
        img_before = imageio.imread(frame_paths[frame_idx_before])
        img_after = imageio.imread(frame_paths[frame_idx_after])
        
        # Create masked images
        img_before_masked = get_masked_image(img_before, mask_before, init_color)
        img_after_masked = get_masked_image(img_after, mask_after, init_color)
        
        # Query VLM for state change
        content_data = [
            prompts['state_change'][0],
            ('image', img_before_masked),
            ('image', img_after_masked),
        ]
        
        try:
            rsp = call_vlm(
                client=client,
                model_name=model_name,
                system_prompt=get_system_prompt(),
                content_data=content_data,
                temperature=temperature,
            )
            
            # Parse response
            state_changed = False
            change_type = 'none'
            change_desc = ''
            severity = 'none'
            
            for line in rsp.split('\n'):
                line = line.strip()
                if line.startswith('STATE_CHANGED:'):
                    state_changed = 'yes' in line.lower()
                elif line.startswith('CHANGE_TYPE:'):
                    change_type = line.split(':', 1)[1].strip()
                elif line.startswith('CHANGE_DESCRIPTION:'):
                    change_desc = line.split(':', 1)[1].strip()
                elif line.startswith('CHANGE_SEVERITY:'):
                    severity = line.split(':', 1)[1].strip().lower()
            
            if state_changed:
                state_changes.append({
                    'start_frame': frame_idx_before,
                    'end_frame': frame_idx_after,
                    'change_type': change_type,
                    'description': change_desc,
                    'severity': severity,
                    'object_idx': obj_idx
                })
                print(f"    State change detected: frames {frame_idx_before}-{frame_idx_after}: {change_desc}")
        
        except Exception as e:
            print(f"    Warning: VLM query failed: {e}")
            continue
    
    return state_changes


def analyze_object_interaction(
    client,
    model_name: str,
    frame: np.ndarray,
    obj_mask: dict,
    interactor_mask: dict,
    init_color: np.ndarray,
    query_color: np.ndarray,
    init_c_name: str,
    query_c_name: str,
    prompts: dict,
    temperature: float = 0.0,
) -> dict:
    """
    Analyze interaction between two objects in a frame.
    """
    # Create masked image with both objects
    img_masked = get_masked_image(frame, obj_mask, init_color)
    img_masked = get_masked_image(img_masked, interactor_mask, query_color)
    
    content_data = [
        prompts['interaction'][0],
        ('image', img_masked),
    ]
    
    try:
        rsp = call_vlm(
            client=client,
            model_name=model_name,
            system_prompt=get_system_prompt(),
            content_data=content_data,
            temperature=temperature,
        )
        
        # Parse response
        result = {
            'contact_type': '',
            'force_direction': '',
            'visible_effect': '',
            'action_verb': '',
            'raw_response': rsp
        }
        
        for line in rsp.split('\n'):
            line = line.strip()
            if line.startswith('CONTACT_TYPE:'):
                result['contact_type'] = line.split(':', 1)[1].strip()
            elif line.startswith('FORCE_DIRECTION:'):
                result['force_direction'] = line.split(':', 1)[1].strip()
            elif line.startswith('VISIBLE_EFFECT:'):
                result['visible_effect'] = line.split(':', 1)[1].strip()
            elif line.startswith('ACTION_VERB:'):
                result['action_verb'] = line.split(':', 1)[1].strip()
        
        return result
    
    except Exception as e:
        print(f"    Warning: Interaction analysis failed: {e}")
        return {'error': str(e)}


# =============================================================================
# Main Pipeline
# =============================================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Enhanced VLM analysis for TubeletGraph.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file")
    parser.add_argument('-p', '--pred', type=str, help='prediction directory name', default='vost-val-Annotations_fps5-Ours')
    parser.add_argument('--vlm', type=str, default='openai', choices=['openai', 'qwen'],
                        help='VLM backbone (ignored when pipeline.py exports OVAD_VLM_BACKBONE)')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--sample_interval', type=int, default=10, help='Frame interval for state change detection')
    parser.add_argument('--skip_state_change', action='store_true', help='Skip state change detection')

    # NEW ARGUMENT
    parser.add_argument('--mask_frame_id', type=int, default=0, help='Frame ID for the mask input.')
    parser.add_argument('--hint_str', type=str, default="", help='Optional hint string to provide to the VLM for better predictions.')

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])
    pred_track_dir = osp.join(cfg.paths.outdir, args.pred)

    # -------------------------------------------------------------------------
    # Client initialisation
    #
    # One OpenAI-protocol client serves both backbones. vlm.base_url decides
    # where it points: unset -> the OpenAI API; set -> an OpenAI-compatible
    # server such as `vllm serve Qwen/Qwen3-VL-32B-Instruct`. The config reaches
    # this script because pipeline.py threads the config file down through
    # quick_run.py and TubeletGraph/run.py, and create_custom_config copies the
    # whole document. $OVAD_VLM_BASE_URL overrides.
    # -------------------------------------------------------------------------
    # Backbone comes from pipeline.py via the environment so every stage agrees.
    # Standalone runs fall back to the config's own vlm block.
    _backbone = (os.environ.get("OVAD_VLM_BACKBONE") or "").strip().lower()
    if _backbone in ("openai", "qwen"):
        # Driven by pipeline.py: take the endpoint it resolved, verbatim.
        base_url = os.environ.get("OVAD_VLM_BASE_URL") or None
        model_name = os.environ.get("OVAD_VLM_MODEL") or None
    elif _backbone == "" and args.vlm == "qwen":
        # Standalone qwen run: base_url and its served model must come as a pair.
        base_url = os.environ.get("OVAD_VLM_BASE_URL") or cfg.vlm.get("base_url") or None
        if not base_url:
            raise SystemExit(
                "--vlm qwen needs an OpenAI-compatible server: set vlm.base_url "
                "in the config or export OVAD_VLM_BASE_URL.")
        model_name = os.environ.get("OVAD_VLM_MODEL")
        if not model_name:
            # Ask the server what it serves rather than guessing from the config.
            model_name = openai.OpenAI(base_url=base_url, api_key="EMPTY",
                                       timeout=30.0).models.list().data[0].id
    else:
        # Standalone openai run: never pair a local base_url with an OpenAI model.
        base_url = None
        model_name = os.environ.get("OVAD_VLM_MODEL") or cfg.vlm.get("openai_model") or "gpt-4.1"

    _client_kwargs = {"timeout": 120.0, "max_retries": 2}
    if base_url:
        _client_kwargs["base_url"] = base_url
        _client_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = openai.OpenAI(**_client_kwargs)
    print(f"[VLM] stage2 model={model_name} endpoint={base_url or 'the OpenAI API'}")

    init_color = np.array(cfg.vlm.init_color_rgb, dtype=float)
    init_c_name = cfg.vlm.init_color_name
    query_color = np.array(cfg.vlm.query_color_rgb, dtype=float)
    query_c_name = cfg.vlm.query_color_name
    
    out_dir = pred_track_dir + f'_' + model_name.replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)
    
    system_prompt = get_system_prompt()
    prompts = get_enhanced_message_prompts(init_c_name, query_c_name, hint_str=args.hint_str)

    instance_names = [x.removesuffix('.json') for x in os.listdir(pred_track_dir) if x.endswith('.json')]

    for instance_name in tqdm(instance_names, desc="Processing instances"):
        new_pred_path = osp.join(out_dir, f'{instance_name}.json')

        if osp.exists(new_pred_path):
            print(f"Skip {instance_name} as {new_pred_path} exists")
            continue


        # Load frame paths and predictions
        with open(osp.join(pred_track_dir, instance_name + '.json'), 'r') as f:
            pred_data = json.load(f)
        video_name = strip_instance_name(instance_name)
        frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, video_name, data_cfg.image_format)))

        track_starts, prompt_obj_idx = get_added_track_starts(pred_data)
        
        to_remove = list()
        obj_info = dict()
        state_change_events = []

        # =====================================================================
        # Stage 0: Always analyze initial object state
        # =====================================================================
        
        # SAFELY DETERMINE THE START FRAME FOR VLM
        mask_frame_str = str(args.mask_frame_id)
        if mask_frame_str not in pred_data['prediction']:
            # Fallback if the requested frame isn't in the predictions
            available_frames = [int(k) for k in pred_data['prediction'].keys()]
            mask_frame_str = str(min(available_frames)) if available_frames else '0'
            print(f"[WARNING] mask_frame_id {args.mask_frame_id} not found in predictions. Falling back to frame {mask_frame_str}.")
            
        # FIXED: Use the correct frame index instead of hardcoded 0
        init_img = imageio.imread(frame_paths[int(mask_frame_str)])
        init_mask = pred_data['prediction'][mask_frame_str].get(prompt_obj_idx)
        
        if init_mask:
            init_img_mask = get_masked_image(init_img, init_mask, init_color)
            
            rsp = call_vlm(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                content_data=[
                    prompts['identify'][0],
                    ('image', init_img_mask),
                ],
                temperature=args.temp,
            )
            
            # Parse object info
            obj_desc = rsp
            obj_state = ''
            obj_material = ''
            for line in rsp.split('\n'):
                if line.startswith('OBJECT:'):
                    obj_desc = line.split(':', 1)[1].strip()
                elif line.startswith('STATE:'):
                    obj_state = line.split(':', 1)[1].strip()
                elif line.startswith('MATERIAL:'):
                    obj_material = line.split(':', 1)[1].strip()
            
            obj_info[prompt_obj_idx] = {
                'desc': obj_desc,
                'initial_state': obj_state,
                'material': obj_material
            }

        # =====================================================================
        # Stage 1: Detect state changes (NEW - even without new objects)
        # =====================================================================
        if not args.skip_state_change:
            
            # Analyze state changes for the main tracked object
            state_changes = detect_state_changes(
                client, model_name,
                frame_paths, pred_data,
                prompt_obj_idx,
                init_color, init_c_name,
                prompts,
                args.temp, args.sample_interval,
            )
            
            if state_changes:
                state_change_events.extend(state_changes)
                obj_info[prompt_obj_idx]['state_changes'] = state_changes

        # =====================================================================
        # Stage 2: Analyze newly appearing objects (original functionality)
        # =====================================================================
        if len(track_starts) != 0:
            
            memory_responses = list()
            for obj_idx, first_obj_frame in track_starts.items():
                
                # Find good frame for analysis
                for i in range(10):
                    obj_start_frame = int(first_obj_frame) + i
                    if obj_start_frame >= len(frame_paths):
                        break
                    late_img = imageio.imread(frame_paths[obj_start_frame])

                    all_objs = list(pred_data['supix_masks'][str(obj_start_frame)].keys())
                    init_mask_union = MaskUtils.merge(
                        [pred_data['supix_masks'][str(obj_start_frame)][x] for x in all_objs if x != obj_idx], 
                        intersect=0
                    )
                    new_track_mask = pred_data['supix_masks'][str(obj_start_frame)][obj_idx]
                    late_img_mask_ = get_masked_image(late_img, init_mask_union, init_color)
                    late_img_mask = get_masked_image(late_img_mask_, new_track_mask, query_color)

                    rsp = call_vlm(
                        client=client,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        content_data=[
                            prompts['frame_check'][0],
                            ('image', late_img_mask),
                        ],
                    )
                    if yes_no_cleanup(rsp) == 'yes':
                        break

                # Get action and object description
                content_data = [
                    prompts['action'][0],
                    ('image', init_img_mask),
                    ('image', late_img_mask),
                ]
                for prev_frame_idx, prev_response in memory_responses:
                    t = (obj_start_frame - prev_frame_idx) // data_cfg.fps
                    content_data.append(('text', f"Previous response at frame {t} seconds ago:"))
                    content_data.append(('text', prev_response))

                rsp = call_vlm(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    content_data=content_data,
                    temperature=args.temp,
                )
                memory_responses.append((obj_start_frame, rsp))
                
                try:
                    # Try to parse JSON response
                    json_str = [x for x in rsp.split('\n') if '```' not in x and '{' in x][0]
                    parsed_rsp = json.loads(json_str)
                    obj_info[obj_idx] = {
                        'desc': parsed_rsp.get('object_change', f'object_{obj_idx}'),
                        'action': parsed_rsp.get('action', ''),
                        'material_released': parsed_rsp.get('material_released', 'no'),
                        "analysis_frame_idx": obj_start_frame,
                        "object_start_frame_idx": int(first_obj_frame)
                    }
                except:
                    # Fallback to tuple parsing
                    try:
                        parsed_rsp = ast.literal_eval([x for x in rsp.split('\n') if '```' not in x][0])
                        obj_info[obj_idx] = {
                            'desc': parsed_rsp[1] if len(parsed_rsp) > 1 else f'object_{obj_idx}',
                            'action': parsed_rsp[2] if len(parsed_rsp) > 2 else '',
                            'prior_desc': parsed_rsp[0] if len(parsed_rsp) > 0 else '',
                            "analysis_frame_idx": obj_start_frame,
                            "object_start_frame_idx": int(first_obj_frame)
                        }
                    except:
                        obj_info[obj_idx] = {
                            'desc': f'object_{obj_idx}',
                            'raw_response': rsp,
                            "analysis_frame_idx": obj_start_frame,
                            "object_start_frame_idx": int(first_obj_frame)
                        }

        # =====================================================================
        # Stage 3: Interaction analysis between objects
        # =====================================================================
        all_obj_ids = list(set([prompt_obj_idx] + list(track_starts.keys())))
        
        if len(all_obj_ids) >= 2:
            
            # Analyze interaction at a few key frames
            frame_indices = sorted([int(k) for k in pred_data['prediction'].keys()])
            mid_frame = frame_indices[len(frame_indices) // 2]
            
            frame_data = pred_data['supix_masks'].get(str(mid_frame), {})
            
            if prompt_obj_idx in frame_data and len(frame_data) > 1:
                other_obj = [k for k in frame_data.keys() if k != prompt_obj_idx][0]
                
                frame = imageio.imread(frame_paths[mid_frame])
                
                interaction_result = analyze_object_interaction(
                    client, model_name,
                    frame,
                    frame_data[prompt_obj_idx],
                    frame_data[other_obj],
                    init_color, query_color,
                    init_c_name, query_c_name,
                    prompts, args.temp,
                )
                
                obj_info[prompt_obj_idx]['interaction'] = interaction_result

        # =====================================================================
        # Save results
        # =====================================================================
        if len(to_remove) > 0:
            to_remove = set(to_remove)
            for frame_idx in pred_data['prediction'].keys():
                new_supix_masks = {k: v for k, v in pred_data['supix_masks'][frame_idx].items() if k not in to_remove}
                pred_data['supix_masks'][frame_idx] = new_supix_masks
                pred_data['prediction'][frame_idx] = {"0": rle_wrapper(MaskUtils.merge(list(new_supix_masks.values()), intersect=0))}
        
        pred_data['obj_info'] = obj_info
        pred_data['state_change_events'] = state_change_events
        
        with open(new_pred_path, 'w') as f:
            json.dump(pred_data, f)
        
        print(f"Saved: {new_pred_path}")
        print(f"  Objects: {list(obj_info.keys())}")
        print(f"  State changes: {len(state_change_events)}")


        