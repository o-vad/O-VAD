import os
from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi, create_repo
import pandas as pd
from pathlib import Path
import shutil
from tqdm import trange
import glob
from pathlib import Path
import numpy as np
import json
import random


def build_vqa_IPAD(parent_dir):
    print("Building VQA for IPAD...")
    out_list = []           # [{'video_path': 'dir', 'question': 'q', 'answer': 'a', 'org_split': 'train/test', 'org_subset': 0/'obj', 'context': [0, 1]}, ...]
    path = Path(parent_dir)
    query = """You are an expert in detecting anomaly in industrial scenarios, such as an ongoing manufacturing line.
Your task is to analyze whether the given video is normal or abnormal and identify the anomaly part within the video.
Your output should strictly follow the format:
<ANOMALY_DETECTION>
- Analysis: a paragraph of reasoning here
- Result: [Normal / Abnormal]
- Frame-wise Detection: [0, 1, ...] (0 or 1 label for each frame, with 0 as normal, 1 as abnormal)
</ANOMALY_DETECTION>
"""
    for s_id in range(1, 17):
        # train set
        train_path = f"{path}/S{s_id:02d}/training/frames"
        temp_path = Path(train_path)
        ds = sorted([str(f) for f in temp_path.iterdir() if f.is_dir()])
        answer = "Normal"
        if s_id < 13:
            org_subset = f"S{s_id:02d}"
        else:
            org_subset = f"R{(s_id-12):02d}"
        for i in range(len(ds)):
            search_path = os.path.join(ds[i], "*.jpg")
            frame_paths = glob.glob(search_path)
            out_list.append({
                "video_path": ds[i],
                "question": query,
                "answer": answer,
                "org_split": "train",
                "org_subset": org_subset,
                "context": [0 for _ in range(len(frame_paths))],
            })
        # test set
        test_path = f"{path}/S{s_id:02d}/testing/frames"
        ## collect labels
        ground_truth = []
        answers = []
        search_path = os.path.join(f"{path}/S{s_id:02d}/test_label", "*.npy")
        frame_paths = glob.glob(search_path)
        frame_paths.sort()
        for sample in frame_paths:
            data = np.load(sample)
            gt_label = data.tolist()
            ground_truth.append(gt_label)
            if any(gt_label):
                answers.append("Abnormal")
            else:
                answers.append("Normal")
        ## build test vqa
        temp_path = Path(test_path)
        ds = sorted([str(f) for f in temp_path.iterdir() if f.is_dir()])
        if s_id < 13:
            org_subset = f"S{s_id:02d}"
        else:
            org_subset = f"R{(s_id-12):02d}"
        for i in range(len(ds)):
            out_list.append({
                "video_path": ds[i],
                "question": query,
                "answer": answers[i],
                "org_split": "test",
                "org_subset": org_subset,
                "context": ground_truth[i],
            })
    print("Total samples for IPAD:", len(out_list))
    return out_list


def split_IPAD(data_list, seed=42):
    print("Generating split for IPAD...")
    random.seed(seed)
    to_mask = ["S06", "R03"]                # masked subsets
    test_normal_cnt = 10                    # normal samples per subset in test split
    test_abnormal_cnt = 4                   # abnormal samples per subset in test split
    
    non_mask_normal = {}                    # {subset : [samples]}
    non_mask_abnormal = {}                  # {subset : [samples]}
    out_list = []
    mask_cnt = 0
    train_cnt = 0
    test_cnt = 0

    for sample in data_list:
        # mask subsets
        if sample["org_subset"] in to_mask:
            sample["split"] = "mask"
            out_list.append(sample)
            mask_cnt += 1
            continue
        if sample["answer"] == "Normal":
            if sample["org_subset"] in non_mask_normal:
                non_mask_normal[sample["org_subset"]].append(sample)
            else:
                non_mask_normal[sample["org_subset"]] = [sample]
        elif sample["answer"] == "Abnormal":
            if sample["org_subset"] in non_mask_abnormal:
                non_mask_abnormal[sample["org_subset"]].append(sample)
            else:
                non_mask_abnormal[sample["org_subset"]] = [sample]
    # random test/train split
    for s_id in list(non_mask_normal.keys()):
        random_samples = random.sample(non_mask_normal[s_id], test_normal_cnt)
        for sample in non_mask_normal[s_id]:
            if sample in random_samples:
                sample["split"] = "test"
                out_list.append(sample)
                test_cnt += 1
            else:
                sample["split"] = "train"
                out_list.append(sample)
                train_cnt += 1
    for s_id in list(non_mask_abnormal.keys()):
        random_samples = random.sample(non_mask_abnormal[s_id], test_abnormal_cnt)
        for sample in non_mask_abnormal[s_id]:
            if sample in random_samples:
                sample["split"] = "test"
                out_list.append(sample)
                test_cnt += 1
            else:
                sample["split"] = "train"
                out_list.append(sample)
                train_cnt += 1
    print(f"Split finish -- Mask:{mask_cnt}, Train:{train_cnt}, Test:{test_cnt}, Total:{len(out_list)}")
    # save to jsonl
    with open("./IPAD_VQA.jsonl", 'w', encoding='utf-8') as f:
        for sample in out_list:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    return out_list


def upload_vqa_IPAD(
    video_data,  # [{'video_path': 'path/to/frames_dir', 'question': 'q', 'answer': 'a', 'org_split': 'train/test', 'org_subset': 'S01', 'context': frame_wise label, 'split': 'train'}, ...]
    repo_id,
    hf_token,
    repo_type="dataset"
):
    temp_dir = Path("./temp_vqa_upload")
    temp_dir.mkdir(exist_ok=True)
    videos_dir = temp_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # metadata
    metadata = []
    print(f"Processing {len(video_data)} video frame directories...")
    for idx in trange(len(video_data)):
        item = video_data[idx]
        folder_name = str(item['org_subset'])
        folder_path = videos_dir / folder_name
        folder_path.mkdir(exist_ok=True)
        frames_dir = Path(item['video_path'])
        frames_dirname = frames_dir.name
        new_dirname = f"{item['org_split']}_{frames_dirname}"
        dest_path = folder_path / new_dirname
        shutil.copytree(frames_dir, dest_path, dirs_exist_ok=True)
        relative_frames_path = f"videos/{folder_name}/{new_dirname}"
        metadata.append({
            'video_url': relative_frames_path,
            'question': item['question'],
            'answer': item['answer'],
            'org_split': item['org_split'],
            'org_subset': item['org_subset'],
            'context': item['context'],
            'split': item['split']
        })
    # create metadata csv
    df = pd.DataFrame(metadata)
    metadata_path = temp_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    # upload to hf
    print("\nUploading to Hugging Face...")
    api = HfApi(token=hf_token)
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Repository creation note: {e}")
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="metadata.csv",
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )
    api.upload_folder(
        folder_path=str(videos_dir),
        path_in_repo="videos",
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )
    shutil.rmtree(temp_dir)
    return 0


def build_vqa_PhysAD(parent_dir):
    print("Building VQA for PhysAD...")
    out_list = []           # [{'video_path': 'path/to/video.mp4', 'question': 'q', 'answer': 'a', 'org_split': 'train/test', 'org_subset': 0/'obj', 'context': label}, ...]
    path = Path(parent_dir)
    objects = [f.name for f in path.iterdir() if f.is_dir()]
    for obj in objects:
        query = f"""You are an expert in detecting physical anomalies, defects or malfunctions in the object.
Your task is to analyze whether the {obj} in the video is normal or abnormal.
Your output should strictly follow the format:
<ANOMALY_DETECTION>
- Analysis: a paragraph of reasoning here
- Result: [Normal / Abnormal]
</ANOMALY_DETECTION>"""
        # train set
        train_path = f"{path}/{obj}/train"
        search_path = os.path.join(train_path, "*.mp4")
        video_paths = glob.glob(search_path)
        video_paths.sort()
        answer = "Normal"
        for i in range(len(video_paths)):
            out_list.append({
                "video_path": video_paths[i],
                "question": query,
                "answer": answer,
                "org_split": "train",
                "org_subset": obj,
                "context": "train_normal"
            })
        # test set
        test_path = f"{path}/{obj}/test"
        temp_path = Path(test_path)
        temp_ds = [str(f) for f in temp_path.iterdir() if f.is_dir()]
        labels = [f.name for f in temp_path.iterdir() if f.is_dir()]
        for i in range(len(temp_ds)):                # labels
            search_path = os.path.join(temp_ds[i], "*.mp4")
            video_paths = glob.glob(search_path)
            video_paths.sort()
            label = labels[i]
            if label == "norm":
                answer = "Normal"
            else:
                answer = "Abnormal"
            for j in range(len(video_paths)):
                out_list.append({
                    "video_path": video_paths[j],
                    "question": query,
                    "answer": answer,
                    "org_split": "test",
                    "org_subset": obj,
                    "context": label
                })
    print("Total samples for PhysAD:", len(out_list))
    return out_list


def split_PhysAD(data_list, seed=42):
    print("Generating split for PhysAD...")
    random.seed(seed)
    to_mask = ["sticky_roller", "screw"]                # masked subsets
    test_normal_cnt = 30                    # normal samples per subset in test split
    test_abnormal_cnt = 35                  # abnormal samples per subset in test split
    
    non_mask_normal = {}                    # {subset : [samples]}
    non_mask_abnormal = {}                  # {subset : [samples]}
    out_list = []
    mask_cnt = 0
    train_cnt = 0
    test_cnt = 0

    for sample in data_list:
        # mask subsets
        if sample["org_subset"] in to_mask:
            sample["split"] = "mask"
            out_list.append(sample)
            mask_cnt += 1
            continue
        if sample["answer"] == "Normal":
            if sample["org_subset"] in non_mask_normal:
                non_mask_normal[sample["org_subset"]].append(sample)
            else:
                non_mask_normal[sample["org_subset"]] = [sample]
        elif sample["answer"] == "Abnormal":
            if sample["org_subset"] in non_mask_abnormal:
                non_mask_abnormal[sample["org_subset"]].append(sample)
            else:
                non_mask_abnormal[sample["org_subset"]] = [sample]
    # random test/train split
    for obj in list(non_mask_normal.keys()):
        random_samples = random.sample(non_mask_normal[obj], test_normal_cnt)
        for sample in non_mask_normal[obj]:
            if sample in random_samples:
                sample["split"] = "test"
                out_list.append(sample)
                test_cnt += 1
            else:
                sample["split"] = "train"
                out_list.append(sample)
                train_cnt += 1
    for obj in list(non_mask_abnormal.keys()):
        temp = min(test_abnormal_cnt, len(non_mask_abnormal[obj]))
        if temp != test_abnormal_cnt:
            print(f"Not enough abnormal for {obj}")
        random_samples = random.sample(non_mask_abnormal[obj], temp)
        for sample in non_mask_abnormal[obj]:
            if sample in random_samples:
                sample["split"] = "test"
                out_list.append(sample)
                test_cnt += 1
            else:
                sample["split"] = "train"
                out_list.append(sample)
                train_cnt += 1
    print(f"Split finish -- Mask:{mask_cnt}, Train:{train_cnt}, Test:{test_cnt}, Total:{len(out_list)}")
    # save to jsonl
    with open("./PhysAD_VQA.jsonl", 'w', encoding='utf-8') as f:
        for sample in out_list:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    return out_list


def upload_vqa_PhysAD(
    video_data,  # [{'video_path': 'path/to/video.mp4', 'question': 'q', 'answer': 'a', 'org_split': 'train/test', 'org_subset': 0/'obj', 'context': label, 'split': split}, ...]
    repo_id,
    hf_token,       # HF token
    repo_type="dataset"
):
    temp_dir = Path("./temp_vqa_upload")
    temp_dir.mkdir(exist_ok=True)
    videos_dir = temp_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # metadata
    metadata = []
    print(f"Processing {len(video_data)} videos...")
    for idx in trange(len(video_data)):
        item = video_data[idx]
        folder_name = item['org_subset']
        folder_path = videos_dir / folder_name
        folder_path.mkdir(exist_ok=True)
        
        # get video filename
        video_path = Path(item['video_path'])
        video_filename = f"{item['context']}_{video_path.name}"
        dest_path = folder_path / video_filename
        shutil.copy2(item['video_path'], dest_path)
        relative_video_path = f"videos/{folder_name}/{video_filename}"
        
        metadata.append({
            'video_url': relative_video_path,
            'question': item['question'],
            'answer': item['answer'],
            'org_split': item['org_split'],
            'org_subset': item['org_subset'],
            'context': item['context'],
            'split': item['split']
        })
    
    # create metadata csv
    df = pd.DataFrame(metadata)
    metadata_path = temp_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    # upload to hf
    print("\nUploading to Hugging Face...")
    api = HfApi(token=hf_token)
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Repository creation note: {e}")
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="metadata.csv",
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )
    api.upload_folder(
        folder_path=str(videos_dir),
        path_in_repo="videos",
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )
    shutil.rmtree(temp_dir)
    return 0


# # IPAD
# query = """You are an expert in detecting anomaly in industrial scenarios, such as an ongoing manufacturing line.
# Your task is to analyze whether the given video is normal or abnormal and identify the anomaly part within the video.
# Your output should strictly follow the format:
# <ANOMALY_DETECTION>
# - Analysis: a paragraph of reasoning here
# - Result: [Normal / Abnormal]
# </ANOMALY_DETECTION>
# """

# # Phys-AD
# query = f"""You are an expert in detecting physical anomalies, defects or malfunctions in the object.
# Your task is to analyze whether the {obj} in the video is normal or abnormal.
# Your output should strictly follow the format:
# <ANOMALY_DETECTION>
# - Analysis: a paragraph of reasoning here
# - Result: [Normal / Abnormal]
# </ANOMALY_DETECTION>"""


if __name__ == "__main__":
    # hf_token = "***REMOVED-CREDENTIAL***"
    parent_dir = f"/shared/scratch/0/home/username/anomaly_detect/datasets/IPAD/IPAD_dataset"
    # repo_id = "LongQ/IPAD_VQA"
    data_list = build_vqa_IPAD(parent_dir)
    out_list = split_IPAD(data_list)
    # upload_vqa_IPAD(out_list, repo_id, hf_token, repo_type="dataset")

    parent_dir = f"/shared/scratch/0/home/username/anomaly_detect/datasets/Phys-AD"
    data_list = build_vqa_PhysAD(parent_dir)
    out_list = split_PhysAD(data_list)