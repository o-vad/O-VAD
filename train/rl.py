import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import av
import numpy as np
from PIL import Image
import json
import gc
from openai import OpenAI
import re
import time
import os


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY", "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")
)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frames_from_video(video_path, num_frames=4, target_size=(336, 336)):
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        frames = read_video_pyav(container, indices)
        container.close()
        
        pil_frames = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            pil_frame = pil_frame.resize(target_size, Image.LANCZOS)
            pil_frames.append(pil_frame)
        return pil_frames
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        # Return black frames on error to prevent crash
        return [Image.new('RGB', target_size) for _ in range(num_frames)]

def extract_output(text):
    try:
        # Case insensitive tag search
        pattern = r"<ANOMALY_DETECTION>(.*?)</ANOMALY_DETECTION>"
        content_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not content_match:
            return None, None
        content = content_match.group(1)
        
        analysis_match = re.search(r"- Analysis:\s*(.*?)\s*- Result:", content, re.DOTALL | re.IGNORECASE)
        result_match = re.search(r"- Result:\s*(.*)", content, re.IGNORECASE)
        
        if not analysis_match or not result_match:
            return None, None
            
        analysis = analysis_match.group(1).strip()
        result_text = result_match.group(1).strip()
        result = "Normal" if "Normal" in result_text else "Abnormal"
        return analysis, result
    except:
        return None, None

def extract_judge(text):
    try:
        content_match = re.search(r"<JUDGE>(.*?)</JUDGE>", text, re.DOTALL | re.IGNORECASE)
        if not content_match: return None, None
        content = content_match.group(1)
        
        subscores_match = re.search(r'Subscores:\s*\[(.*?)\]', content)
        if not subscores_match: return None, None
        
        subscores_str = subscores_match.group(1)
        subscores = [int(re.search(r'(\d+)', item).group(1)) for item in subscores_str.split(',')]
        
        final_score_match = re.search(r'Final Score:\s*(\d+)', content)
        if not final_score_match: return None, None
        final_score = int(final_score_match.group(1))
        
        return subscores, final_score
    except:
        return None, None

def llm_as_judge(analysis, result):
    prompt = f"""You are an expert in checking an analysis on anomaly detection.
Analysis: {analysis}
Final Answer: {result}
Starting from 0 score, give +1 for:
1. Scene description included.
2. Analysis consistent with Answer.
3. Reasoning supports claim.
4. Clear and confident answer.
Output format:
<JUDGE>
- Subscores: [0/1, 0/1, 0/1, 0/1]
- Final Score: 0-4
</JUDGE>"""
    
    for _ in range(5):
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3-next-80b-a3b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512, stream=False
            )
            answer = completion.choices[0].message.content
            subscores, final_score = extract_judge(answer)
            if subscores is not None: return subscores, final_score
        except:
            time.sleep(30)
    return None, None

def reward_function(prompts, completions, **kwargs):
    # Retrieve metadata columns
    ground_truth_labels = kwargs.get("ground_truth_label", [])
    ground_truth_thinkings = kwargs.get("ground_truth_thinking", [])
    
    rewards = []
    for i, completion in enumerate(completions):
        if i >= len(ground_truth_labels): 
            rewards.append(0.0)
            continue
            
        gt_label = ground_truth_labels[i]
        gt_thinking = ground_truth_thinkings[i]

        answer = completion[0]["content"]
        analysis, result = extract_output(answer)
        
        reward = 0.0
        
        if not analysis or not result:
            rewards.append(-0.5)
            continue
            
        if result.lower() == gt_label.lower():
            reward += 0.5
        else:
            rewards.append(0.0)
            continue
        
        if abs(len(analysis) - len(gt_thinking)) < 100:
            reward += 0.1
            
        subscores, judge_score = llm_as_judge(analysis, result)
        if judge_score:
            reward += min(judge_score * 0.1, 0.4)
                
        rewards.append(reward)
    return rewards

def grpo_finetune(base_model, sft_model, data_path, push_path, num_frames=4, frame_size=(448, 448)):
    hf_token = "***REMOVED-CREDENTIAL***"
    
    # 1. Load Metadata Only (Fast & Memory Efficient)
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            # Store only strings/paths
            raw_data.append({
                "video_path": obj["video_path"],
                "object_id": obj["object_id"],
                "label": "Normal" if obj["label"] == "normal" else "Abnormal",
                "thinking": obj["thinking"]
            })
    
    # Create dataset from strings (Arrow handles this easily)
    dataset = Dataset.from_list(raw_data)
    print(f"Dataset metadata loaded: {len(dataset)} samples")

    # 2. Define Transform Function (Loads images on-the-fly)
    def formatting_func(examples):
        outputs = {"prompt": [], "ground_truth_label": [], "ground_truth_thinking": []}
        
        video_paths = examples["video_path"]
        object_ids = examples["object_id"]
        labels = examples["label"]
        thinkings = examples["thinking"]
        
        for i in range(len(video_paths)):
            frames = sample_frames_from_video(video_paths[i], num_frames, frame_size)
            
            question = f"""Analyze if {object_ids[i]} is normal/abnormal.
Output format:
<ANOMALY_DETECTION>
- Analysis: ...
- Result: [Normal / Abnormal]
</ANOMALY_DETECTION>"""

            # CORRECT QWEN FORMAT: List of Dictionaries
            # This fixes the "mix struct/non-struct" Arrow error because we bypass Arrow storage for this part
            content = []
            for frame in frames:
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": question})
            
            # The prompt is a list of messages
            outputs["prompt"].append([{"role": "user", "content": content}])
            outputs["ground_truth_label"].append(labels[i])
            outputs["ground_truth_thinking"].append(thinkings[i])
            
        return outputs

    # Set the transform - this runs dynamically during training
    dataset.set_transform(formatting_func)

    # 3. Model Setup
    processor = AutoProcessor.from_pretrained(base_model, token=hf_token)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    training_args = GRPOConfig(
        output_dir="./logs_grpo",
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        num_generations=4,
        max_completion_length=1024,
        # Important for VLM to allow processor to handle images
        remove_unused_columns=False 
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=reward_function,
    )
    
    print("Starting GRPO training...")
    trainer.train()
    
    print("Saving model...")
    model.merge_and_unload().push_to_hub(push_path, private=True, token=hf_token)
    processor.push_to_hub(push_path, private=True, token=hf_token)

if __name__ == "__main__":
    sft_model = "LongQ/Qwen3-VL-2B-SFT-Test"
    base_model = "Qwen/Qwen3-VL-2B-Instruct"
    data_path = "./datasets/Phys-AD_train_set_with_thinking.jsonl"
    push_path = "LongQ/Qwen3-VL-2B-GRPO-Test"
    
    grpo_finetune(base_model, sft_model, data_path, push_path, num_frames=4, frame_size=(224, 224))