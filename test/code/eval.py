import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import glob
import os


def prediction_to_frame_labels(pred, num_frames):
    labels = [0] * num_frames
    if pred['anomaly'] and pred['anomaly'].lower() == 'yes':
        start = pred['Anomaly Start']
        end = pred['Anomaly End']
        if start is not None and end is not None:
            if isinstance(start, int) and isinstance(end, int):
                start = max(0, min(start, num_frames - 1))
                end = max(0, min(end, num_frames - 1))
                for i in range(start, end + 1):
                    if i < num_frames:
                        labels[i] = 1
    return labels


def compute_metrics(predictions, ground_truth):
    results = []
    for pred in predictions:
        s_id = pred['S_id']
        v_id = pred['v_id']
        key = (s_id, v_id)
        if key not in ground_truth:
            print(f"Warning: No ground truth found for S{s_id:02d}_v{v_id}")
            continue
        gt_labels = ground_truth[key]
        num_frames = len(gt_labels)
        pred_labels = prediction_to_frame_labels(pred, num_frames)
        
        # video-level metrics
        gt_has_anomaly = 1 if any(gt_labels) else 0
        pred_has_anomaly = 1 if (pred['anomaly'] and pred['anomaly'].lower() == 'yes') else 0
        video_correct = int(gt_has_anomaly == pred_has_anomaly)
        # frame-level metrics
        gt_labels_np = np.array(gt_labels)
        pred_labels_np = np.array(pred_labels)
        # accuracy: (TP + TN) / total
        frame_accuracy = np.mean(gt_labels_np == pred_labels_np)
        # precision: TP / (TP + FP)
        tp = np.sum((pred_labels_np == 1) & (gt_labels_np == 1))
        fp = np.sum((pred_labels_np == 1) & (gt_labels_np == 0))
        if tp + fp > 0:
            frame_precision = tp / (tp + fp)
        else:
            frame_precision = 1.0 if np.sum(gt_labels_np) == 0 else 0.0
        # recall: TP / (TP + FN)
        fn = np.sum((pred_labels_np == 0) & (gt_labels_np == 1))
        if tp + fn > 0:
            frame_recall = tp / (tp + fn)
        else:
            frame_recall = 1.0 if tp == 0 else 0.0
        results.append({
            'S_id': s_id,
            'v_id': v_id,
            'video_correct': video_correct,
            'frame_accuracy': frame_accuracy,
            'frame_precision': frame_precision,
            'frame_recall': frame_recall,
            'gt_labels': gt_labels,
            'pred_labels': pred_labels
        })
    
    # Compute overall metrics
    overall_metrics = {
        'video_correct_rate': np.mean([r['video_correct'] for r in results]),
        'avg_frame_accuracy': np.mean([r['frame_accuracy'] for r in results]),
        'avg_frame_precision': np.mean([r['frame_precision'] for r in results]),
        'avg_frame_recall': np.mean([r['frame_recall'] for r in results]),
        'total_samples': len(results)
    }
    return {
        'per_sample_results': results,
        'overall_metrics': overall_metrics
    }


if __name__ == "__main__":
    # load predictions
    pred_file = "./eval_results/ipad_base_output.jsonl"
    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    # IPAD
    # load ground truth
    ground_truth = {}
    for s_id in range(1, 17):
        v_id = 0
        parent_dir = f"/shared/scratch/0/home/username/anomaly_detect/datasets/IPAD/IPAD_dataset/S{s_id:02d}/test_label"
        search_path = os.path.join(parent_dir, "*.npy")
        frame_paths = glob.glob(search_path)
        frame_paths.sort()
        for sample in frame_paths:
            data = np.load(sample)
            ground_truth[(s_id, v_id)] = data.tolist()
            v_id += 1
    # compute metrics
    metrics = compute_metrics(predictions, ground_truth)
    # store results
    eval_filename = "./eval_results/ipad_base_eval.jsonl"
    with open(eval_filename, 'w', encoding='utf-8') as f:
        for sample in metrics['per_sample_results']:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        sample = metrics['overall_metrics']
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print("Overall Metrics:", metrics['overall_metrics'])

    # # Phys-AD
    # total = 0
    # correct = 0
    # for pred in predictions:
    #     if pred['label'] == "norm" and (pred['anomaly'] is None or pred['anomaly'].lower() == "no"):
    #         correct += 1
    #     elif pred['label'] != "norm" and pred['anomaly'] is not None and pred['anomaly'].lower() == "yes":
    #         correct += 1
    #     total += 1
    # print(f"Phys-AD Accuracy: {correct}/{total} = {correct/total:.4f}")