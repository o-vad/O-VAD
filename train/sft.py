import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import av
import numpy as np
from PIL import Image
import json
import gc


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
    """
    Sample frames from video and resize them
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        target_size: (width, height) to resize frames to. Common sizes:
            - (224, 224): Very low resolution, max memory saving
            - (336, 336): Low resolution, good memory saving
            - (448, 448): Medium resolution, balanced
            - (672, 672): Higher resolution, more memory
    """
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    frames = read_video_pyav(container, indices)
    container.close()
    
    # Convert to PIL and resize
    pil_frames = []
    for frame in frames:
        pil_frame = Image.fromarray(frame)
        # Resize with high-quality resampling
        pil_frame = pil_frame.resize(target_size, Image.LANCZOS)
        pil_frames.append(pil_frame)
    
    return pil_frames


def finetune(base_model, data_path, push_path, num_frames=4, frame_size=(448, 448)):
    hf_token = "***REMOVED-CREDENTIAL***"
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load dataset
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    train_set = Dataset.from_list(data)
    
    print(f"Total training samples: {len(train_set)}")
    print(f"Using frame size: {frame_size}")
    print(f"Using {num_frames} frames per video")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model, token=hf_token)
    
    # Load model on single GPU with memory optimizations
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    
    # Enable gradient checkpointing before LoRA
    model.gradient_checkpointing_enable()
    
    # Configure LoRA with minimal parameters
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target attention and MLP layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess dataset to create text messages (without images)
    def preprocess_dataset(sample):
        question = f"""You are an expert in detecting physical anomalies, defects or malfunctions in the object.
Your task is to analyze whether the {sample["object_id"]} in the video is normal or abnormal.
You should follow these guidelines:
1. Ignore Normal Variations: Do NOT mark as abnormal if objects are simply in a different position, has a different appearance, or if the lighting/background has changed. These are expected variations.
2. Focus on Physical Anomalies: You must only report "Yes" for Anomaly such as if you observe obvious different physical reaction, physical changes or functional failure compared to normal state.
Your output should strictly follow the format:
<ANOMALY_DETECTION>
- Analysis: a paragraph of reasoning here
- Result: [Normal / Abnormal]
</ANOMALY_DETECTION>"""
        
        label = "Normal" if sample["label"] == "normal" else "Abnormal"
        answer = f"""<ANOMALY_DETECTION>
- Analysis: {sample["thinking"]}
- Result: {label}
</ANOMALY_DETECTION>"""
        
        # Store only serializable data (no PIL images)
        return {
            "video_path": sample["video_path"],
            "question": question,
            "answer": answer,
            "num_frames": num_frames,
            "frame_size": frame_size
        }
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    train_set = train_set.map(
        preprocess_dataset, 
        remove_columns=["object_id", "label", "thinking"],
        desc="Creating text data"
    )
    
    # Custom data collator for vision-language models
    class VideoDataCollator:
        def __init__(self, processor, num_frames, frame_size):
            self.processor = processor
            self.num_frames = num_frames
            self.frame_size = frame_size
        
        def __call__(self, examples):
            # Extract data and load images
            texts = []
            images_list = []
            
            for example in examples:
                video_path = example["video_path"]
                question = example["question"]
                answer = example["answer"]
                
                # Load frames from video with resizing
                frames = sample_frames_from_video(
                    video_path, 
                    self.num_frames, 
                    self.frame_size
                )
                
                # Create message with video frames
                content = []
                for frame in frames:
                    content.append({"type": "image", "image": frame})
                content.append({"type": "text", "text": question})
                
                messages = [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": answer}
                ]
                
                # Apply chat template to get text
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                texts.append(text)
                images_list.append(frames)
            
            # Process with processor
            batch = self.processor(
                text=texts,
                images=images_list,
                return_tensors="pt",
                padding=True
            )
            
            # Create labels (copy input_ids and mask padding tokens)
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            
            return batch
    
    # Create collator instance
    collate_fn = VideoDataCollator(processor, num_frames, frame_size)
    
    # Set up training with aggressive memory optimization
    training_args = SFTConfig(
        output_dir="./logs",
        num_train_epochs=3.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_grad_norm=0.5,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        auto_find_batch_size=False,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=collate_fn,
        processing_class=processor,
    )
    
    print("Starting training...")
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Merge LoRA weights with base model
    print("Merging LoRA weights with base model...")
    model = model.merge_and_unload()
    
    # Push to hub
    print("Pushing to Hub...")
    model.push_to_hub(push_path, private=True, token=hf_token)
    processor.push_to_hub(push_path, private=True, token=hf_token)
    print(f"Model successfully pushed to Hub at {push_path}")


if __name__ == "__main__":
    base_model = "Qwen/Qwen3-VL-2B-Instruct"
    data_path = "./datasets/Phys-AD_train_set_with_thinking.jsonl"
    push_path = "LongQ/Qwen3-VL-2B-SFT-Test"
    
    # Hyperparameters to tune for memory vs quality:
    NUM_FRAMES = 4  # Number of frames per video
    FRAME_SIZE = (224, 224)  # Resolution of each frame
    
    # Memory saving options (from most to least aggressive):
    # FRAME_SIZE = (224, 224)  # Very aggressive - saves most memory, lowest quality
    # FRAME_SIZE = (336, 336)  # Aggressive - good memory saving
    # FRAME_SIZE = (448, 448)  # Balanced - recommended starting point
    # FRAME_SIZE = (672, 672)  # High quality - uses more memory
    
    finetune(base_model, data_path, push_path, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE)