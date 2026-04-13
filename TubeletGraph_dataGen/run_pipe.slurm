#!/bin/bash

#SBATCH --job-name="run_pipe_1" 
#SBATCH --account="bdse-delta-gpu"       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA40x4
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=run_pipe_1.o%j
#SBATCH --error=run_pipe_1.e%j

source /u/qilong/utils/miniconda3/etc/profile.d/conda.sh
conda activate tubeletgraph
cd /u/qilong/anomaly_detect/ST-VAD/TubeletGraph
export OPENAI_API_KEY='***REMOVED-CREDENTIAL***'

python /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/pipe_PhysAD.py analyze /u/qilong/anomaly_detect/datasets/Phys-AD/toothpaste -c /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/configs/default.yaml --vlm openai --fps 3 --sample_interval 60 --vqa_file /u/qilong/anomaly_detect/datasets/vqa_data/PhysAD_VQA.jsonl --split test

python /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/pipe_IPAD.py analyze /u/qilong/anomaly_detect/datasets/IPAD/IPAD_dataset/S01 -c /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/configs/default.yaml --vlm openai --fps 3 --sample_interval 25 --vqa_file /u/qilong/anomaly_detect/datasets/vqa_data/IPAD_VQA.jsonl --split test

