# Data Synthesis

## Data Download

### PhysAD

Download from this website: https://huggingface.co/datasets/guoliz/Phys-AD.
```bash
unzip Phys_AD_Data_part_0000_of_0006.zip -d Phys-AD
```
If data is not well structured, need to manualy move files around.

VQA data is provided in data_vqa folder.

### IPAD

```bash
pip install gdown
gdown 1kuNG0OaNJnji1y122q3gxy7SF7iXABV4
unzip IPAD_dataset.zip -d IPAD
```
If data is not well structured, need to manualy move files around.

VQA data is provided in data_vqa folder.

### AutoLab

Download latest version dataset with pipette-level labels and VQA data.

If data is not well structured, need to manualy move files around.

## Environment

Follow TubeletGraph to install libraries and thirdparty packages.

## Code Structure

```bash
.
├── TubeletGraph
├── VOST-TAS
├── annotate
├── assets
├── configs
├── eval
├── eval_bert.py
├── eval_general.py
├── eval_results
├── graph_run.py
├── pipe_AutoLab.py
├── pipe_AutoLab_gen.py
├── pipe_IPAD.py
├── pipe_IPAD_gen.py
├── pipe_PhysAD.py
├── pipe_PhysAD_gen.py
├── pipe_speedup.py
├── quick_run.py
├── requirements.txt
├── run_ablation.slurm
├── run_pipe.slurm
├── splits
├── stvad_demo.py
├── thirdparty
└── utils.py
```

### Main Pipeline

The main data synthesis pipeline are in pipe_xxx_gen.py files, separate for each dataset.

It calls each stage's codes as helper functions.

Contains vlm calling for caption and dynamic fps generation.

### Stage 1

Stage 1 code is stored in annotate folder. The vlm_mask_grounded.py is advanced segmentation used for AutoLab dataset only, the vlm_mask_grounded_org.py is used for PhysAD and IPAD datasets.

### Stage 2

Stage 2 codes are managed in TubeletGraph folder.

The run.py code is the main pipeline for stage 2, which calls other codes in order.

vlm/prompt_vlm.py code contains vlm calling for object state tracking.

### Stage 3

Stage 3 code is TubeletGraph/vlm/prompt_vad.py, which calls vlm for reasoning and final report generation.

## Code Adaptation

### Load and Storage Path

Inside VQA data for all datasets, modify the prefix in video_path to your data saved location for input loading:
```json
"video_path": "/u/qilong/anomaly_detect/datasets/IPAD/IPAD_dataset/S06/training/frames/01"
```

Inside pipe_xxx_gen.py files, search and modify these lines for output storing:
```python
STAGE1_ROOT = Path("/work/nvme/bgiv/qilong/stage1")
STAGE2_ROOT = Path("/work/nvme/bgiv/qilong/stage2")
STAGE3_ROOT = Path("/work/nvme/bgiv/qilong/stage3")
```

### Build Hint

For PhysAD and IPAD, the build hint logic should work based on existing VQA data.

For AutoLab, when pipette (object) level label is ready, update the pipe_AutoLab_gen.py code to load such labels by replacing the lines:
```python
# ── Retrieve per-sample VQA entry ────────────
vqa_metadata = (vqa_metadata.get(vp)
                or vqa_metadata.get(str(Path(vp).resolve()))
                or {})
vqa_metadata["order"] = "fourth"
```
Refer to "Build hint string" part for more details about hint structure.

## Execution

Now we are ready to run the data synthesis pipeline.

```bash
python <PATH TO pipe_IPAD_gen.py> \
analyze <PATH TO IPAD/IPAD_dataset/SUBSET> \
-c <PATH TO configs/default.yaml> \
--vlm openai --fps 3 --sample_interval 25 \
--vqa_file <PATH TO vqa_data/IPAD_VQA.jsonl> \
--split <test/train> --hint

python <PATH TO pipe_PhysAD_gen.py> \
analyze <PATH TO Phys-AD/SUBSET> \
-c <PATH TO configs/default.yaml> \
--vlm openai --fps 3 --sample_interval 60 \
--vqa_file <PATH TO vqa_data/PhysAD_VQA.jsonl> \
--split <test/train> --hint

python <PATH TO pipe_AutoLab_gen.py> \
analyze <PATH TO AutoLab/SUBSET> \
-c <PATH TO configs/default.yaml> \
--vlm openai --fps 3 --sample_interval 60 \
--vqa_file <PATH TO vqa_data/autolab.jsonl>\
--split <test/train> --hint
```

For example:
```bash
python /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/pipe_IPAD_gen.py analyze /u/qilong/anomaly_detect/datasets/IPAD/IPAD_dataset/S01 -c /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/configs/default.yaml --vlm openai --fps 3 --sample_interval 25 --vqa_file /u/qilong/anomaly_detect/datasets/vqa_data/IPAD_VQA.jsonl --split test --hint

python /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/pipe_PhysAD_gen.py analyze /u/qilong/anomaly_detect/datasets/Phys-AD/fan/test/norm -c /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/configs/default.yaml --vlm openai --fps 3 --sample_interval 60 --vqa_file /u/qilong/anomaly_detect/datasets/vqa_data/PhysAD_VQA.jsonl --split test --hint

python /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/pipe_AutoLab_gen.py analyze /u/qilong/anomaly_detect/datasets/AutoLab/test/abnormal_1 -c /u/qilong/anomaly_detect/ST-VAD/TubeletGraph/configs/default.yaml --vlm openai --fps 3 --sample_interval 60 --vqa_file /u/qilong/anomaly_detect/datasets/AutoLab/autolab.jsonl --split test --hint
```
To get full synthesis data, need to set --split to test, after running the code, set --split to train to run (6 commands in total).




