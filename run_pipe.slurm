#!/bin/bash

#SBATCH --job-name="ovad_pipe"
#SBATCH --account="<YOUR_ACCOUNT>"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA40x4
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=ovad_pipe.o%j
#SBATCH --error=ovad_pipe.e%j

set -euo pipefail

source <PATH_TO>/miniconda3/etc/profile.d/conda.sh
conda activate ovad
cd <PATH_TO>/O_VAD

# Provide the API key from your environment or a secrets file — never inline it
# here, or it lands in git history. e.g.  source ~/.ovad_secrets
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

DATA=<PATH_TO>/datasets
CFG=configs/default.yaml

# ── Phys-AD ────────────────────────────────────────────────────────────────
python pipeline.py analyze "$DATA/Phys-AD/toothpaste" --dataset physad \
    -c "$CFG" \
    --vqa_file "$DATA/vqa_data/PhysAD_VQA.jsonl" \
    --vlm openai --fps 3 --sample_interval 60 \
    --split test

# ── IPAD ───────────────────────────────────────────────────────────────────
python pipeline.py analyze "$DATA/IPAD/IPAD_dataset/S01" --dataset ipad \
    -c "$CFG" \
    --vqa_file "$DATA/vqa_data/IPAD_VQA.jsonl" \
    --vlm openai --fps 3 --sample_interval 25 \
    --split test

# ── AutoLab / LiquidAD ─────────────────────────────────────────────────────
# python pipeline.py analyze "$DATA/AutoLab/test/abnormal_1" --dataset autolab \
#     -c "$CFG" \
#     --vqa_file "$DATA/AutoLab/autolab.jsonl" \
#     --dataset_root "$DATA/AutoLab" \
#     --vlm openai --fps 3 --sample_interval 60 \
#     --split test
