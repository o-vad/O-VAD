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
from TubeletGraph.vlm.html_writer import HTMLWriter


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
            raise ValueError("Unsupported data type: {}".format(data_type))
    return content


def get_model_response(response, html_writer=None, sleep_time=0):
    time.sleep(sleep_time)
    rsp = response.choices[0].message.content
    if html_writer is not None:
        html_writer.add_heading('Response', level=3)
        html_writer.add_text(rsp)
    return rsp


# =============================================================================
# Enhanced System Prompts
# =============================================================================

def get_system_prompt():
    """Enhanced system prompt that focuses on state changes and interactions."""
    prompt = """You are an expert video analysis assistant specialized in detecting object state changes and interactions.

Your tasks include:
1. Identifying objects and their current states
2. Detecting ANY change in object appearance, shape, or material state
3. Recognizing interactions between objects (gripping, pressing, squeezing)
4. Noticing subtle dynamics like material flow, deformation, or leaking

Be thorough and observant. Even small, subtle changes are important to report.
Pay special attention to:
- Deformable objects (tubes, bags, soft materials)
- Material release (liquids, pastes, powders coming out)
- Pressure effects from contact with other objects
- Shape changes even without complete separation"""
    return prompt


def get_enhanced_message_prompts(init_c_name, query_c_name):
    """
    Enhanced prompts that capture state changes, not just new objects.
    """
    
    # Object identification
    prompt_messages_id = [
        ('text', f"""Look at this image from a video. The object of interest is highlighted with a {init_c_name} contour.

Please describe:
1. Object name (be specific, e.g., "toothpaste tube" not just "tube")
2. Current state (intact, squeezed, opened, etc.)
3. Material type (plastic, metal, liquid inside, etc.)

Answer format:
OBJECT: [name]
STATE: [current state]
MATERIAL: [material]"""),
    ]
    
    # Frame quality check
    prompt_messages_frame = [
        ('text', f"""Can you clearly see the objects marked with {init_c_name} and {query_c_name} contours in this image?
Consider: motion blur, occlusion, cropping at edges.
Answer: yes or no"""),
    ]
    
    # State change detection (NEW - this is the key addition)
    prompt_messages_state_change = [
        ('text', f"""Compare these two frames from a video. The same object is highlighted with {init_c_name} contour in both.

First image: Earlier frame
Second image: Later frame

Carefully check for ANY of these changes:
- DEFORMATION: Is the object's shape different? (compressed, bent, stretched)
- MATERIAL RELEASE: Is anything coming out of the object? (liquid, paste, contents)
- SURFACE CHANGE: Any cracks, tears, openings, damage?
- SIZE CHANGE: Has the object gotten bigger or smaller?
- TEXTURE CHANGE: Has the surface appearance changed?

Even subtle changes count! A tube being slightly squeezed or paste starting to come out are significant.

Answer format:
STATE_CHANGED: [yes/no]
CHANGE_TYPE: [deformation/material_release/surface_change/size_change/none]
CHANGE_DESCRIPTION: [describe what changed in detail]
CHANGE_SEVERITY: [none/slight/moderate/severe]"""),
    ]
    
    # Interaction detection between objects
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
    html_writer=None,
    temperature: float = 0.0,
    sample_interval: int = 10
) -> list:
    """
    Detect state changes in a tracked object across video frames.
    
    This is the key new function that detects changes like:
    - Toothpaste tube being squeezed
    - Material leaking out
    - Deformation from gripper pressure
    
    Args:
        client: OpenAI client
        model_name: VLM model name
        frame_paths: List of frame file paths
        pred_data: Prediction data with masks
        obj_idx: Object ID to analyze
        init_color: Color for visualization
        init_c_name: Color name for prompts
        prompts: Prompt templates
        html_writer: Optional HTML writer
        temperature: Sampling temperature
        sample_interval: Frames between samples
    
    Returns:
        List of detected state change events
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
    
    if html_writer:
        html_writer.add_heading(f'State Change Analysis for Object {obj_idx}', level=2)
    
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
        content = [
            prompts['state_change'][0],
            ('image', img_before_masked),
            ('image', img_after_masked),
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": get_user_content(content, html_writer=html_writer)}
                ],
                temperature=temperature,
            )
            rsp = get_model_response(response, html_writer=html_writer)
            
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
    html_writer=None,
    temperature: float = 0.0
) -> dict:
    """
    Analyze interaction between two objects in a frame.
    
    Args:
        client: OpenAI client
        model_name: VLM model name
        frame: Video frame
        obj_mask: Mask of object being manipulated
        interactor_mask: Mask of manipulating object (gripper)
        init_color, query_color: Visualization colors
        init_c_name, query_c_name: Color names for prompts
        prompts: Prompt templates
        html_writer: Optional HTML writer
        temperature: Sampling temperature
    
    Returns:
        Dict with interaction analysis results
    """
    # Create masked image with both objects
    img_masked = get_masked_image(frame, obj_mask, init_color)
    img_masked = get_masked_image(img_masked, interactor_mask, query_color)
    
    content = [
        prompts['interaction'][0],
        ('image', img_masked),
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": get_user_content(content, html_writer=html_writer)}
            ],
            temperature=temperature,
        )
        rsp = get_model_response(response, html_writer=html_writer)
        
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
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--sample_interval', type=int, default=10, help='Frame interval for state change detection')
    parser.add_argument('--skip_state_change', action='store_true', help='Skip state change detection')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])
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
    
    system_prompt = get_system_prompt()
    prompts = get_enhanced_message_prompts(init_c_name, query_c_name)

    instance_names = [x.removesuffix('.json') for x in os.listdir(pred_track_dir) if x.endswith('.json')]

    for instance_name in tqdm(instance_names, desc="Processing instances"):
        html_out_path = osp.join(out_dir, f'{instance_name}.html')
        new_pred_path = osp.join(out_dir, f'{instance_name}.json')

        if osp.exists(new_pred_path) and osp.exists(html_out_path):
            print(f"Skip {instance_name} as {new_pred_path} and {html_out_path} exists")
            continue

        html_writer = HTMLWriter(title=f"Prompt Results - {instance_name} - {cfg.vlm.model_name}")
        html_writer.add_heading('System-level Prompt', level=3)
        html_writer.add_text(system_prompt)

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
        init_img = imageio.imread(frame_paths[0])
        init_mask = pred_data['prediction']['0'].get(prompt_obj_idx)
        
        if init_mask:
            init_img_mask = get_masked_image(init_img, init_mask, init_color)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": get_user_content([
                        prompts['identify'][0],
                        ('image', init_img_mask),
                    ], html_writer=html_writer)}
                ],
                temperature=args.temp,
            )
            rsp = get_model_response(response, html_writer=html_writer)
            
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
            html_writer.add_heading('State Change Detection', level=2)
            
            # Analyze state changes for the main tracked object
            state_changes = detect_state_changes(
                client, model_name,
                frame_paths, pred_data,
                prompt_obj_idx,
                init_color, init_c_name,
                prompts, html_writer,
                args.temp, args.sample_interval
            )
            
            if state_changes:
                state_change_events.extend(state_changes)
                obj_info[prompt_obj_idx]['state_changes'] = state_changes
                html_writer.add_text(f"Detected {len(state_changes)} state changes for object {prompt_obj_idx}")
            else:
                html_writer.add_text(f"No state changes detected for object {prompt_obj_idx}")

        # =====================================================================
        # Stage 2: Analyze newly appearing objects (original functionality)
        # =====================================================================
        if len(track_starts) == 0:
            html_writer.add_text("No additional objects to track.")
        else:
            html_writer.add_heading('New Object Analysis', level=2)
            
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
                        break

                # Get action and object description
                contents = [
                    prompts['action'][0],
                    ('image', init_img_mask),
                    ('image', late_img_mask),
                ]
                for prev_frame_idx, prev_response in memory_responses:
                    t = (obj_start_frame - prev_frame_idx) // data_cfg.fps
                    contents.append(('text', f"Previous response at frame {t} seconds ago:"))
                    contents.append(('text', prev_response))

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
            html_writer.add_heading('Object Interaction Analysis', level=2)
            
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
                    prompts, html_writer, args.temp
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
        html_writer.save(html_out_path)
        
        print(f"Saved: {new_pred_path}")
        print(f"  Objects: {list(obj_info.keys())}")
        print(f"  State changes: {len(state_change_events)}")