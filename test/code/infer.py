import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from pathlib import Path
from datasets import load_dataset
import glob
import os
from tqdm import trange
import json


def infer(sample, model, processor, type="video", context=None):
    """
    IPAD: process one sample (series of frame images)
    Phys-AD: process one sample (one video), sample = [ref_video, test_video]
    """
    # llm input
    if type == "dir":                # sample: [ref_frames, test_frames]
        ref_search_path = os.path.join(sample[0], "*.jpg")
        ref_frame_paths = glob.glob(ref_search_path)
        ref_frame_paths.sort()
        search_path = os.path.join(sample[1], "*.jpg")
        frame_paths = glob.glob(search_path)
        frame_paths.sort()
        # search_path = os.path.join(sample[1], "*.jpg")
        # frame_paths = glob.glob(search_path)
        # frame_paths.sort()
        system_prompt = "You are an expert in detecting anomaly in industrial scenarios."
        user_prompt = """The first video shows an example of a normal industrial process.
        The second video shows frames to be analyzed.
        Your task is to analyze whether the second video is abnormal or normal and identify the anomaly part within the video.
        You should follow these guidelines:
        1. Ignore Normal Variations: Do NOT mark as abnormal if objects are simply in a different position, has a different orientation, or if the lighting/background has changed. These are expected variations.
        2. Focus on Industrial Process: You must only report "Yes" for Anomaly if you observe such as unreasonable objects, industrial process defects and malfunctions.
        Your output should strictly follow the format:
        <ANOMALY_DETECTION>
        - Anomaly: [Yes / No]
        - Anomaly Segment: [start_frame_id : end_frame_id] (or N/A if video is normal)
        - Anomaly Reason: [reason why it is abnormal] (or N/A if video is normal)
        </ANOMALY_DETECTION>
        """
        messages = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": ref_frame_paths}, 
                    {"type": "video", "video": frame_paths}, 
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
    elif type == "video":               # sample: [ref_video, test_video]
        ref_video_path = sample[0]
        video_path = sample[1]
        system_prompt = "You are an expert in detecting physical anomalies, defects or malfunctions in the object."
        user_prompt = f"""The first video shows the normal state of {context} as reference.
        The second video shows {context} to be analyzed.
        Your task is to analyze whether the {context} in the second video is abnormal or normal, compared to the reference video.
        You should follow these guidelines:
        1. Ignore Normal Variations: Do NOT mark as abnormal if objects are simply in a different position, has a different appearance, or if the lighting/background has changed. These are expected variations.
        2. Focus on Physical Anomalies: You must only report "Yes" for Anomaly such as if you observe obvious different physical reaction, physical changes or functional failure compared to the reference.
        Your output should strictly follow the format:
        <ANOMALY_DETECTION>
        - Anomaly: [Yes / No]
        - Anomaly Reason: [reason why it is abnormal] (or N/A if video is normal)
        </ANOMALY_DETECTION>
        """
        messages = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": ref_video_path}, 
                    {"type": "video", "video": video_path}, 
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(messages)
    image_inputs, video_inputs = process_vision_info(messages)
    video_placeholder = "<video>" * 2304
    raw_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = raw_text.replace("<video>", video_placeholder, 1)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    # generate output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def parse(llm_outputs, dataset_name="IPAD"):
    if llm_outputs is None:
        return None
    tag_content = re.search(r"<ANOMALY_DETECTION>(.*?)</ANOMALY_DETECTION>", llm_outputs, re.DOTALL)
    if not tag_content:
        return None
    content = tag_content.group(1)
    if dataset_name == "IPAD":
        anomaly_match = re.search(r"-?\s*Anomaly:\s*\[?\s*(Yes|No)\s*\]?", content, re.I)
        segment_match = re.search(r"-?\s*Anomaly Segment:\s*\[?\s*(\d+)\s*:\s*(\d+)\s*\]?", content)
        reason_match = re.search(r"-?\s*Anomaly Reason:\s*\[?\s*(.*?)\s*\]?\s*$", content, re.MULTILINE)
        anomaly_status = anomaly_match.group(1).strip() if anomaly_match else None
        start_frame = int(segment_match.group(1)) if segment_match else None
        end_frame = int(segment_match.group(2)) if segment_match else None
        reason = reason_match.group(1).strip() if reason_match else None
        if reason and reason.lower() == 'n/a':
            reason = None
        return {
            "anomaly": anomaly_status,
            "Anomaly Start": start_frame,
            "Anomaly End": end_frame,
            "Anomaly Reason": reason
        }
    elif dataset_name == "Phys-AD":
        anomaly_match = re.search(r"-?\s*Anomaly:\s*\[?\s*(Yes|No)\s*\]?", content, re.I)
        reason_match = re.search(r"-?\s*Anomaly Reason:\s*\[?\s*(.*?)\s*\]?\s*$", content, re.MULTILINE)
        anomaly_status = anomaly_match.group(1).strip() if anomaly_match else None
        reason = reason_match.group(1).strip() if reason_match else None
        if reason and reason.lower() == 'n/a':
            reason = None
        return {
            "anomaly": anomaly_status,
            "Anomaly Reason": reason
        }
    else:
        print("Dataset not supported for parsing.")
        return None


def infer_pipe(model, processor, dataset_name, s_id=None):
    """
    IPAD: s_id in [1, 16], 16 scenarios (process one scenario)
    Phys-AD: s_id in ["ball", "button", ...], type of object (process one type of object)
    """
    out_clean = []          # parsed outputs
    
    # load dataset
    if dataset_name == "IPAD":
        ref_path = f"/shared/scratch/0/home/username/anomaly_detect/datasets/IPAD/IPAD_dataset/S{s_id:02d}/training/frames/01"
        parent_dir = f"/shared/scratch/0/home/username/anomaly_detect/datasets/IPAD/IPAD_dataset/S{s_id:02d}/testing/frames"
        path = Path(parent_dir)
        ds = sorted([str(f) for f in path.iterdir() if f.is_dir()])
        modal_type = "dir"
        # infer
        for i in trange(len(ds)):
            sample = [ref_path, ds[i]]
            output = infer(sample, model, processor, modal_type)
            parsed_outputs = parse(output, "IPAD")
            if parsed_outputs is None:
                parsed_outputs = {
                    "anomaly": "N/A",
                    "Anomaly Start": "N/A",
                    "Anomaly End": "N/A",
                    "Anomaly Reason": "N/A"
                }
            parsed_outputs['S_id'] = s_id
            parsed_outputs['v_id'] = i
            out_clean.append(parsed_outputs)
    elif dataset_name == "Phys-AD":
        ref_video_path = f"/shared/scratch/0/home/username/anomaly_detect/datasets/Phys-AD/{s_id}/train/0000.mp4"
        parent_dir = f"/shared/scratch/0/home/username/anomaly_detect/datasets/Phys-AD/{s_id}/test"
        path = Path(parent_dir)
        ds = [str(f) for f in path.iterdir() if f.is_dir()]
        labels = [f.name for f in path.iterdir() if f.is_dir()]
        modal_type = "video"
        # infer
        for i in range(len(ds)):                # labels
            print(f"Processing {labels[i]}...")
            search_path = os.path.join(ds[i], "*.mp4")
            video_paths = glob.glob(search_path)
            video_paths.sort()
            label = labels[i]
            for j in trange(len(video_paths)):              # samples
                sample = [ref_video_path, video_paths[j]]
                output = infer(sample, model, processor, modal_type, context=s_id)
                parsed_outputs = parse(output, "Phys-AD")
                if parsed_outputs is None:
                    parsed_outputs = {
                        "anomaly": "N/A",
                        "Anomaly Reason": "N/A"
                    }
                parsed_outputs['object'] = s_id
                parsed_outputs['v_id'] = j
                parsed_outputs['label'] = label
                out_clean.append(parsed_outputs)
    else:
        print("Dataset not supported.")
        return None

    return out_clean

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    # load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    # infer IPAD
    print("Starting IPAD inference...")
    filename = "./eval_results/ipad_base_withRef_output.jsonl"
    for s_id in range(1, 17):
        print(f"Processing S{s_id:02d}...")
        out_clean = infer_pipe(model, processor, "IPAD", s_id)
        with open(filename, 'a', encoding='utf-8') as f:
            for sample in out_clean:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    # # infer Phys-AD
    # print("Starting Phys-AD inference...")
    # filename = "./eval_results/phys_ad_base_output.jsonl"
    # parent_dir = f"/shared/scratch/0/home/username/anomaly_detect/datasets/Phys-AD"
    # path = Path(parent_dir)
    # objects = [f.name for f in path.iterdir() if f.is_dir()]
    # for s_id in objects:            # test
    #     print(f"Processing Object {s_id}...")
    #     out_clean = infer_pipe(model, processor, "Phys-AD", s_id)
    #     with open(filename, 'a', encoding='utf-8') as f:
    #         for sample in out_clean:
    #             f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    










