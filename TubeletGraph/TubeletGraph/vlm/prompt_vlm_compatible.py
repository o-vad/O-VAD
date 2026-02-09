"""
Enhanced prompt_vlm.py for TubeletGraph - Compatible Version
=============================================================

This version maintains backward compatibility with vis.py while adding
state change detection for industrial video anomaly detection.

Key features:
1. Detects state changes even when no new objects appear
2. Maintains 'prior_desc', 'desc', 'action' fields for vis.py compatibility
3. Adds anomaly detection prompts for industrial scenarios
4. Supports both transformation detection and subtle state changes

Usage:
    python prompt_vlm_enhanced.py -c configs/default.yaml -p custom-0000-Ours \
        --sample_interval 10 --detect_anomalies
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

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))
from utils import rle_to_bmask, apply_anno, generate_rand_colors, load_yaml_file, strip_instance_name
from TubeletGraph.vlm.html_writer import HTMLWriter


# =============================================================================
# Image Encoding Utilities
# =============================================================================

def encode_image_from_np(image_np, is_rgb=True, resize=1):
    """Convert numpy array to base64 encoded image."""
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
    return {"type": "text", "text": text}


def get_user_content(data, html_writer=None, show_img_width=600, image_detail='low'):
    if html_writer is not None:
        html_writer.add_heading('Prompt', level=3)
    content = list()
    for (data_type, x) in data:
        if data_type == 'text':
            content.append(get_text_payload(x))
            if html_writer is not None:
                html_writer.add_text(x)
        elif data_type == 'image':
            content.append(get_image_payload(x, image_detail))
            if html_writer is not None:
                html_writer.add_image(x)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    return content


def get_model_response(response, html_writer=None, sleep_time=0):
    time.sleep(sleep_time)
    rsp = response.choices[0].message.content
    if html_writer is not None:
        html_writer.add_heading('Response', level=3)
        html_writer.add_text(rsp)
    return rsp


# =============================================================================
# System Prompts for Industrial Anomaly Detection
# =============================================================================

def get_system_prompt(mode='default'):
    """
    Get system prompt based on analysis mode.
    
    Args:
        mode: 'default', 'anomaly', or 'industrial'
    """
    if mode == 'anomaly':
        return """You are an expert video analysis assistant specialized in industrial anomaly detection.

Your task is to analyze video frames from industrial/robotic processes and identify:
1. **Normal Operations**: Expected behaviors, proper material handling, correct sequences
2. **Anomalies**: Unexpected events, failures, defects, or deviations from normal
3. **State Changes**: Any change in object appearance, integrity, or material state
4. **Causal Reasoning**: Why did the anomaly occur? What is the root cause?

Types of anomalies to detect:
- Material defects (leaking, breaking, deformation beyond tolerance)
- Process failures (missed steps, wrong sequence, equipment malfunction)
- Physical anomalies (unexpected forces, collisions, misalignment)
- Temporal anomalies (action too fast/slow, wrong timing)

Be precise about WHEN anomalies occur (frame ranges) and WHAT caused them."""

    elif mode == 'industrial':
        return """You are an expert industrial process analyst with deep knowledge of:
- Robotic manipulation and gripper operations
- Material handling (tubes, containers, deformable objects)
- Quality control and defect detection
- Process monitoring and anomaly detection

Analyze the video frames to understand:
1. What operation is being performed?
2. Is the operation proceeding normally?
3. Are there any signs of anomaly or failure?
4. What is the expected vs actual outcome?"""

    else:  # default
        return """You are a highly intelligent assistant that can analyze videos and images.
You are particularly skilled at detecting subtle changes in object states and interactions."""


# =============================================================================
# Enhanced Prompt Templates
# =============================================================================

def get_message_prompts(init_c_name, query_c_name):
    """
    Get all prompt templates.
    Returns dict with prompts for different analysis stages.
    """
    
    # Object identification (original format for vis.py compatibility)
    prompt_id = [
        ('text', f"Here is an image, which is the first frame of a video. The object of interest is highlighted with a {init_c_name} contour."),
        ('text', f"Please name the object with {init_c_name} contour as concisely as possible in three words or less. Please do not include any other information in your answer."),
    ]
    
    # Frame quality check
    prompt_frame = [
        ('text', f"Do you able to recognize the objects in {init_c_name} and {query_c_name} contours clearly? Potential difficulties may be due to motion blur, or the majority of the objects are cropped by the edge of the image. Please answer with yes or no and do not include any other information."),
    ]
    
    # Original action prompt (for vis.py compatibility)
    prompt_action = [
        ('text', f"Here is an image, which is the first frame of a video. The object of interest is highlighted with a {init_c_name} contour."),
        ('text', f"Here is another image, which is a later frame of the same video. The original object of interest is still highlighted with a {init_c_name} contour. In addition, there is another object with a {query_c_name} contour that we believe is also a part of the original object of interest."),
        ('text', f"Please describe the two objects (one with {init_c_name} contour and one with {query_c_name} contour) in the second image each in three words or less while including the object name. In addition, please describe what is happening to the object of interest in the second image with a verb only without any tense. Please give the answer as a json tuple of (object with {init_c_name} contour, object with {query_c_name} contour, action). Please do not include any other information in your answer."),
    ]
    
    # NEW: State change detection (without new objects)
    prompt_state_change = [
        ('text', f"""Compare these two frames from a video. The same object is highlighted with {init_c_name} contour in both.

First image: Earlier frame (before)
Second image: Later frame (after)

Analyze if the object has undergone ANY state change:
- DEFORMATION: Shape changed (compressed, bent, stretched, crushed)
- MATERIAL_RELEASE: Contents coming out (leaking, oozing, spilling)
- DAMAGE: Cracks, tears, breaks, holes appearing
- DISPLACEMENT: Significant position/orientation change

Answer as a JSON tuple: (original_object_description, changed_object_description, action_verb)

Example answers:
- ("intact toothpaste tube", "squeezed toothpaste tube", "squeeze")
- ("sealed container", "leaking container", "leak")
- ("straight wire", "bent wire", "bend")

If NO significant change: ("object_name", "object_name", "none")"""),
    ]
    
    # NEW: Anomaly detection prompt
    prompt_anomaly = [
        ('text', f"""Analyze this sequence for anomalies in an industrial/robotic process.

The object being manipulated has {init_c_name} contour.
The manipulator (gripper/tool) has {query_c_name} contour.

First image: Start of operation
Second image: During/after operation

Determine:
1. OPERATION_TYPE: What operation is being performed? (grasping, squeezing, lifting, etc.)
2. EXPECTED_OUTCOME: What should normally happen?
3. ACTUAL_OUTCOME: What actually happened?
4. IS_ANOMALY: Is this an anomaly? (yes/no)
5. ANOMALY_TYPE: If yes, what type? (material_failure, process_error, equipment_issue, physical_damage)
6. ANOMALY_DESCRIPTION: Describe the anomaly in detail
7. SEVERITY: (none/minor/moderate/severe)

Answer format:
OPERATION_TYPE: [type]
EXPECTED_OUTCOME: [expected]
ACTUAL_OUTCOME: [actual]
IS_ANOMALY: [yes/no]
ANOMALY_TYPE: [type or none]
ANOMALY_DESCRIPTION: [description]
SEVERITY: [level]"""),
    ]
    
    # NEW: Temporal reasoning prompt
    prompt_temporal = [
        ('text', f"""You are analyzing a sequence of frames from an industrial process video.

I will show you multiple frames in chronological order. The main object has {init_c_name} contour.

For each transition between frames, identify:
1. What action/change occurred
2. Whether it's normal or anomalous
3. The temporal relationship to previous events

Build a causal chain: Event A → Event B → Event C...

Identify the KEY MOMENT where anomaly begins (if any)."""),
    ]
    
    return {
        'identify': prompt_id,
        'frame_check': prompt_frame,
        'action': prompt_action,
        'state_change': prompt_state_change,
        'anomaly': prompt_anomaly,
        'temporal': prompt_temporal
    }


# =============================================================================
# Image Processing
# =============================================================================

def get_masked_image(image, mask, color_rgb, mask_alpha=0.0, contour_thickness=3):
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


def get_added_track_starts(pred_data, prompt_obj_idx='0'):
    """Get frames where new objects start being tracked."""
    objs = dict()
    for frame_idx, frame_info in pred_data['supix_masks'].items():
        for obj_idx in frame_info:
            if obj_idx != prompt_obj_idx and obj_idx not in objs:
                objs[obj_idx] = frame_idx
    return objs, prompt_obj_idx


# =============================================================================
# State Change Detection (for when no new objects appear)
# =============================================================================

def detect_state_changes_for_object(
    client,
    model_name: str,
    frame_paths: list,
    pred_data: dict,
    obj_idx: str,
    init_color: np.ndarray,
    init_c_name: str,
    prompts: dict,
    html_writer=None,
    temperature: float = 0.0,
    sample_interval: int = 15,
    init_obj_desc: str = "object"
) -> list:
    """
    Detect state changes for an object across video frames.
    
    Returns list of state changes in vis.py-compatible format:
    {
        'prior_desc': description before change,
        'desc': description after change,
        'action': action verb,
        'analysis_frame_idx': frame where change detected,
        'object_start_frame_idx': frame where change started
    }
    """
    state_changes = []
    predictions = pred_data.get('prediction', {})
    
    frame_indices = sorted([int(k) for k in predictions.keys()])
    
    if len(frame_indices) < 2:
        return state_changes
    
    # Sample frames
    sample_frames = [frame_indices[0]]
    for i in range(sample_interval, len(frame_indices), sample_interval):
        sample_frames.append(frame_indices[i])
    if frame_indices[-1] not in sample_frames:
        sample_frames.append(frame_indices[-1])
    
    if html_writer:
        html_writer.add_heading(f'State Change Detection (Object {obj_idx})', level=2)
    
    print(f"  Checking {len(sample_frames)} frames for state changes...")
    
    # Compare first frame to later frames
    first_frame_idx = sample_frames[0]
    first_frame = imageio.imread(frame_paths[first_frame_idx])
    first_mask = predictions.get(str(first_frame_idx), {}).get(obj_idx)
    
    if first_mask is None:
        return state_changes
    
    first_img_masked = get_masked_image(first_frame, first_mask, init_color)
    
    for later_frame_idx in sample_frames[1:]:
        later_mask = predictions.get(str(later_frame_idx), {}).get(obj_idx)
        
        if later_mask is None:
            continue
        
        later_frame = imageio.imread(frame_paths[later_frame_idx])
        later_img_masked = get_masked_image(later_frame, later_mask, init_color)
        
        # Query VLM for state change
        content = [
            prompts['state_change'][0],
            ('image', first_img_masked),
            ('image', later_img_masked),
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": get_system_prompt('default')},
                    {"role": "user", "content": get_user_content(content, html_writer=html_writer)}
                ],
                temperature=temperature,
            )
            rsp = get_model_response(response, html_writer=html_writer)
            
            # Parse tuple response: (prior_desc, desc, action)
            try:
                # Try to extract tuple from response
                tuple_match = None
                for line in rsp.split('\n'):
                    line = line.strip()
                    if line.startswith('(') and ')' in line:
                        tuple_match = line
                        break
                    if '```' not in line and '(' in line and ')' in line:
                        # Extract tuple part
                        start = line.find('(')
                        end = line.rfind(')') + 1
                        tuple_match = line[start:end]
                        break
                
                if tuple_match:
                    parsed = ast.literal_eval(tuple_match)
                    prior_desc, desc, action = parsed[0], parsed[1], parsed[2]
                    
                    # Check if there's a real change (not "none" action)
                    if action.lower() not in ['none', 'no change', 'unchanged', 'same']:
                        state_changes.append({
                            'prior_desc': prior_desc,
                            'desc': desc,
                            'action': action,
                            'analysis_frame_idx': later_frame_idx,
                            'object_start_frame_idx': first_frame_idx
                        })
                        print(f"    State change at frame {later_frame_idx}: {prior_desc} → {desc} ({action})")
                        # Found a change, don't need to check more frames
                        break
            except Exception as e:
                print(f"    Warning: Failed to parse response: {e}")
                continue
                
        except Exception as e:
            print(f"    Warning: VLM query failed: {e}")
            continue
    
    return state_changes


def detect_anomalies_for_object(
    client,
    model_name: str,
    frame_paths: list,
    pred_data: dict,
    obj_idx: str,
    interactor_idx: str,
    init_color: np.ndarray,
    query_color: np.ndarray,
    init_c_name: str,
    query_c_name: str,
    prompts: dict,
    html_writer=None,
    temperature: float = 0.0,
    sample_interval: int = 15
) -> list:
    """
    Detect anomalies in object manipulation.
    
    Returns list of anomaly detections.
    """
    anomalies = []
    predictions = pred_data.get('prediction', {})
    supix_masks = pred_data.get('supix_masks', {})
    
    frame_indices = sorted([int(k) for k in predictions.keys()])
    
    if len(frame_indices) < 2:
        return anomalies
    
    # Sample frames
    sample_frames = [frame_indices[0]]
    for i in range(sample_interval, len(frame_indices), sample_interval):
        sample_frames.append(frame_indices[i])
    if frame_indices[-1] not in sample_frames:
        sample_frames.append(frame_indices[-1])
    
    if html_writer:
        html_writer.add_heading('Anomaly Detection', level=2)
    
    print(f"  Checking for anomalies across {len(sample_frames)} frames...")
    
    first_frame_idx = sample_frames[0]
    first_frame = imageio.imread(frame_paths[first_frame_idx])
    
    # Get masks for first frame
    first_obj_mask = predictions.get(str(first_frame_idx), {}).get(obj_idx)
    first_supix = supix_masks.get(str(first_frame_idx), {})
    first_interactor_mask = first_supix.get(interactor_idx) if interactor_idx in first_supix else None
    
    if first_obj_mask is None:
        return anomalies
    
    first_img_masked = get_masked_image(first_frame, first_obj_mask, init_color)
    if first_interactor_mask:
        first_img_masked = get_masked_image(first_img_masked, first_interactor_mask, query_color)
    
    for later_frame_idx in sample_frames[1:]:
        later_frame = imageio.imread(frame_paths[later_frame_idx])
        
        later_obj_mask = predictions.get(str(later_frame_idx), {}).get(obj_idx)
        later_supix = supix_masks.get(str(later_frame_idx), {})
        later_interactor_mask = later_supix.get(interactor_idx) if interactor_idx in later_supix else None
        
        if later_obj_mask is None:
            continue
        
        later_img_masked = get_masked_image(later_frame, later_obj_mask, init_color)
        if later_interactor_mask:
            later_img_masked = get_masked_image(later_img_masked, later_interactor_mask, query_color)
        
        # Query VLM for anomaly
        content = [
            prompts['anomaly'][0],
            ('image', first_img_masked),
            ('image', later_img_masked),
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": get_system_prompt('anomaly')},
                    {"role": "user", "content": get_user_content(content, html_writer=html_writer)}
                ],
                temperature=temperature,
            )
            rsp = get_model_response(response, html_writer=html_writer)
            
            # Parse anomaly response
            anomaly_info = {
                'frame_idx': later_frame_idx,
                'is_anomaly': False,
                'operation_type': '',
                'expected_outcome': '',
                'actual_outcome': '',
                'anomaly_type': 'none',
                'anomaly_description': '',
                'severity': 'none'
            }
            
            for line in rsp.split('\n'):
                line = line.strip()
                if line.startswith('OPERATION_TYPE:'):
                    anomaly_info['operation_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('EXPECTED_OUTCOME:'):
                    anomaly_info['expected_outcome'] = line.split(':', 1)[1].strip()
                elif line.startswith('ACTUAL_OUTCOME:'):
                    anomaly_info['actual_outcome'] = line.split(':', 1)[1].strip()
                elif line.startswith('IS_ANOMALY:'):
                    anomaly_info['is_anomaly'] = 'yes' in line.lower()
                elif line.startswith('ANOMALY_TYPE:'):
                    anomaly_info['anomaly_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('ANOMALY_DESCRIPTION:'):
                    anomaly_info['anomaly_description'] = line.split(':', 1)[1].strip()
                elif line.startswith('SEVERITY:'):
                    anomaly_info['severity'] = line.split(':', 1)[1].strip().lower()
            
            if anomaly_info['is_anomaly']:
                anomalies.append(anomaly_info)
                print(f"    Anomaly at frame {later_frame_idx}: {anomaly_info['anomaly_type']} - {anomaly_info['anomaly_description']}")
        
        except Exception as e:
            print(f"    Warning: Anomaly detection failed: {e}")
            continue
    
    return anomalies


# =============================================================================
# Main Pipeline
# =============================================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Enhanced VLM analysis for TubeletGraph with anomaly detection.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file")
    parser.add_argument('-p', '--pred', type=str, help='prediction directory name', default='vost-val-Annotations_fps5-Ours')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--sample_interval', type=int, default=15, help='Frame interval for state change detection')
    parser.add_argument('--detect_anomalies', action='store_true', help='Enable anomaly detection mode')
    parser.add_argument('--skip_state_change', action='store_true', help='Skip state change detection')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    # data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])
    data_cfg = getattr(cfg.datasets, os.path.basename(args.pred).split('-')[0])
    pred_track_dir = osp.join(cfg.paths.outdir, args.pred)

    model_name = cfg.vlm.model_name
    if model_name.startswith('Qwen'):
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_API_KEY")
        )
    else:
        client = openai.OpenAI()

    init_color = np.array(cfg.vlm.init_color_rgb, dtype=float)
    init_c_name = cfg.vlm.init_color_name
    query_color = np.array(cfg.vlm.query_color_rgb, dtype=float)
    query_c_name = cfg.vlm.query_color_name
    
    out_dir = pred_track_dir + f'_' + model_name.replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)
    
    system_prompt = get_system_prompt('anomaly' if args.detect_anomalies else 'default')
    prompts = get_message_prompts(init_c_name, query_c_name)

    instance_names = [x.removesuffix('.json') for x in os.listdir(pred_track_dir) if x.endswith('.json')]

    for instance_name in tqdm(instance_names, desc="Processing instances"):
        html_out_path = osp.join(out_dir, f'{instance_name}.html')
        new_pred_path = osp.join(out_dir, f'{instance_name}.json')

        if osp.exists(new_pred_path) and osp.exists(html_out_path):
            print(f"Skip {instance_name} as outputs exist")
            continue

        html_writer = HTMLWriter(title=f"Prompt Results - {instance_name} - {cfg.vlm.model_name}")
        html_writer.add_heading('System-level Prompt', level=3)
        html_writer.add_text(system_prompt)

        # Load data
        with open(osp.join(pred_track_dir, instance_name + '.json'), 'r') as f:
            pred_data = json.load(f)
        video_name = strip_instance_name(instance_name)
        frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, video_name, data_cfg.image_format)))

        track_starts, prompt_obj_idx = get_added_track_starts(pred_data)
        
        to_remove = list()
        obj_info = dict()
        anomaly_events = []

        # =================================================================
        # Stage 1: Identify initial object
        # =================================================================
        init_img = imageio.imread(frame_paths[0])
        init_mask = pred_data['prediction']['0'].get(prompt_obj_idx)
        init_obj_desc = "object"
        
        if init_mask:
            init_img_mask = get_masked_image(init_img, init_mask, init_color)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": get_user_content([
                        prompts['identify'][0],
                        ('image', init_img_mask),
                        prompts['identify'][1],
                    ], html_writer=html_writer)}
                ],
                temperature=args.temp,
            )
            rsp = get_model_response(response, html_writer=html_writer)
            init_obj_desc = rsp.strip()
            obj_info[prompt_obj_idx] = {'desc': init_obj_desc}

        # =================================================================
        # Stage 2: Handle newly appearing objects (original pipeline)
        # =================================================================
        if len(track_starts) == 0:
            html_writer.add_text("No additional objects to track.")
            
            # =============================================================
            # Stage 2b: Detect state changes (when no new objects)
            # =============================================================
            if not args.skip_state_change:
                state_changes = detect_state_changes_for_object(
                    client, model_name,
                    frame_paths, pred_data,
                    prompt_obj_idx,
                    init_color, init_c_name,
                    prompts, html_writer,
                    args.temp, args.sample_interval,
                    init_obj_desc
                )
                
                if state_changes:
                    # Create synthetic "new object" entries for state changes
                    # This makes the output compatible with vis.py
                    for i, sc in enumerate(state_changes):
                        synthetic_obj_idx = f"sc_{i}"
                        obj_info[synthetic_obj_idx] = {
                            'desc': sc['desc'],
                            'action': sc['action'],
                            'prior_desc': sc['prior_desc'],
                            'analysis_frame_idx': sc['analysis_frame_idx'],
                            'object_start_frame_idx': sc['object_start_frame_idx'],
                            'is_state_change': True
                        }
                        html_writer.add_text(f"State change detected: {sc['prior_desc']} → {sc['desc']} ({sc['action']})")
                else:
                    html_writer.add_text("No state changes detected.")
        else:
            # Original pipeline for new objects
            memory_responses = list()
            
            for obj_idx, first_obj_frame in track_starts.items():
                # Find good frame
                obj_start_frame = int(first_obj_frame)
                for i in range(10):
                    test_frame = int(first_obj_frame) + i
                    if test_frame >= len(frame_paths):
                        break
                    late_img = imageio.imread(frame_paths[test_frame])

                    all_objs = list(pred_data['supix_masks'][str(test_frame)].keys())
                    init_mask_union = MaskUtils.merge(
                        [pred_data['supix_masks'][str(test_frame)][x] for x in all_objs if x != obj_idx],
                        intersect=0
                    )
                    new_track_mask = pred_data['supix_masks'][str(test_frame)][obj_idx]
                    late_img_mask_ = get_masked_image(late_img, init_mask_union, init_color)
                    late_img_mask = get_masked_image(late_img_mask_, new_track_mask, query_color)

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": get_user_content([
                                prompts['frame_check'][0],
                                ('image', late_img_mask),
                            ], html_writer=html_writer)}
                        ]
                    )
                    rsp = get_model_response(response, html_writer=html_writer)
                    if yes_no_cleanup(rsp) == 'yes':
                        obj_start_frame = test_frame
                        break

                # Get action description
                contents = [
                    prompts['action'][0],
                    ('image', init_img_mask),
                    prompts['action'][1],
                    ('image', late_img_mask),
                    prompts['action'][2],
                ]
                for prev_frame_idx, prev_response in memory_responses:
                    t = (obj_start_frame - prev_frame_idx) // data_cfg.fps
                    contents.append(('text', f"Previous response {t}s ago: {prev_response}"))

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": get_user_content(contents, html_writer=html_writer)}
                    ],
                    temperature=args.temp,
                )
                rsp = get_model_response(response, html_writer=html_writer)
                memory_responses.append((obj_start_frame, rsp))
                
                # Parse response
                try:
                    parsed_rsp = ast.literal_eval([x for x in rsp.split('\n') if '```' not in x and '(' in x][0])
                    obj_info[obj_idx] = {
                        'desc': parsed_rsp[1],
                        'action': parsed_rsp[2],
                        'prior_desc': parsed_rsp[0],
                        'analysis_frame_idx': obj_start_frame,
                        'object_start_frame_idx': int(first_obj_frame)
                    }
                except:
                    obj_info[obj_idx] = {
                        'desc': f'object_{obj_idx}',
                        'action': 'transform',
                        'prior_desc': init_obj_desc,
                        'analysis_frame_idx': obj_start_frame,
                        'object_start_frame_idx': int(first_obj_frame),
                        'raw_response': rsp
                    }

        # =================================================================
        # Stage 3: Anomaly detection (optional)
        # =================================================================
        if args.detect_anomalies:
            # Find potential interactor (e.g., gripper)
            all_obj_ids = list(set([prompt_obj_idx] + list(track_starts.keys())))
            interactor_idx = None
            if len(all_obj_ids) > 1:
                interactor_idx = [x for x in all_obj_ids if x != prompt_obj_idx][0]
            
            anomalies = detect_anomalies_for_object(
                client, model_name,
                frame_paths, pred_data,
                prompt_obj_idx, interactor_idx,
                init_color, query_color,
                init_c_name, query_c_name,
                prompts, html_writer,
                args.temp, args.sample_interval
            )
            
            if anomalies:
                anomaly_events.extend(anomalies)
                pred_data['anomaly_events'] = anomaly_events
                
                # Add anomaly info to obj_info for vis.py
                for i, anomaly in enumerate(anomalies):
                    if anomaly['is_anomaly']:
                        anomaly_obj_idx = f"anomaly_{i}"
                        obj_info[anomaly_obj_idx] = {
                            'desc': anomaly['actual_outcome'],
                            'action': anomaly['anomaly_type'],
                            'prior_desc': anomaly['expected_outcome'],
                            'analysis_frame_idx': anomaly['frame_idx'],
                            'object_start_frame_idx': 0,
                            'is_anomaly': True,
                            'severity': anomaly['severity'],
                            'anomaly_description': anomaly['anomaly_description']
                        }

        # =================================================================
        # Save results
        # =================================================================
        if len(to_remove) > 0:
            to_remove_set = set(to_remove)
            for frame_idx in pred_data['prediction'].keys():
                new_supix_masks = {k: v for k, v in pred_data['supix_masks'][frame_idx].items() if k not in to_remove_set}
                pred_data['supix_masks'][frame_idx] = new_supix_masks
                pred_data['prediction'][frame_idx] = {"0": rle_wrapper(MaskUtils.merge(list(new_supix_masks.values()), intersect=0))}
        
        pred_data['obj_info'] = obj_info
        
        with open(new_pred_path, 'w') as f:
            json.dump(pred_data, f)
        html_writer.save(html_out_path)
        
        print(f"\nSaved: {new_pred_path}")
        print(f"  Objects: {list(obj_info.keys())}")
        if anomaly_events:
            print(f"  Anomalies: {len(anomaly_events)}")
