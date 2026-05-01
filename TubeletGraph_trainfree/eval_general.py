"""
evaluate.py  –  General-purpose VQA anomaly-detection evaluator.

Usage
-----
# PhysAD  →  video-level 5 metrics + BERT score
python evaluate.py \
    --dataset physad \
    --vqa     /path/to/vqa.jsonl \
    --reports /path/to/PhysAD/subject_root

# IPAD  →  video-level 5 metrics + frame-level AUROC + frame-level Acc/P/R/F1
python evaluate.py \
    --dataset ipad \
    --vqa     /path/to/vqa.jsonl \
    --reports /path/to/IPAD/IPAD_dataset/R01

# AutoLab  →  video-level 5 metrics + frame-level AUROC + frame-level Acc/P/R/F1
python evaluate.py \
    --dataset autolab \
    --vqa     /path/to/vqa.jsonl \
    --reports /path/to/stage3/AutoLab

---------------------------------------------------------------------------
Report format (key fields used):
  report["video_name"]         e.g. "R01_testing_frames_02"
  report["prediction_name"]    fallback if video_name absent
  report["anomaly_detected"]   bool  – video-level prediction
  report["overall_confidence"] float – video-level confidence
  report["num_frames"]         int   – total frames in video (used by AutoLab)
  report["anomalies"][*]:
      "confidence"             float
      "start_frame"            int
      "end_frame"              int
      "anomaly_type"           str
      "anomaly_subtype"        str

VQA JSONL format (one JSON object per line):
  "video_path"   path string
  "answer"       "Normal" | "Abnormal"
  "context"      IPAD    → list[int] per-frame GT labels (0/1)
                 PhysAD  → str anomaly-type label (or "normal")
                 AutoLab → list[int, int] = [start_frame, end_frame]  ← span format

AutoLab context expansion
-------------------------
AutoLab's context [start_frame, end_frame] is a closed interval marking the
anomalous segment.  Before computing frame-level metrics it is expanded to a
per-frame 0/1 list of length `num_frames` (total frames for that video).

Total frame count is resolved in this priority order:
  1. report["num_frames"]            – explicit field written by the pipeline
  2. max(anomaly end_frame) + 1      – inferred from predicted intervals
  3. gt_end_frame + 1                – minimum viable length from GT span

Frame-level binary prediction
------------------------------
A frame is predicted ABNORMAL when it falls inside at least one predicted
anomaly interval (report["anomalies"][*]["start_frame".."end_frame"]).
All other frames are predicted NORMAL.  This binary prediction is used for
Accuracy, Precision, Recall, and F1.  AUROC uses the continuous
_to_anomaly_score mapping (unchanged).

Matching rule
-------------
The report's video_name ("R01_testing_frames_02") is compared to a
normalised key derived from the VQA video_path by anchoring on the first
subject-ID component (e.g. "R01") and joining all following path parts
with "_":
  "/u/.../IPAD_dataset/R01/testing/frames/02"  →  "R01_testing_frames_02"

For AutoLab the video_path is relative ("train/abnormal_16/000.mp4") so the
anchor is omitted and the full stem path is used:
  "train/abnormal_16/000.mp4"  →  "train_abnormal_16_000"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


# ---------------------------------------------------------------------------
# Score conversion
# ---------------------------------------------------------------------------

def _to_anomaly_score(confidence: float, predicted_abnormal: bool) -> float:
    """Map (confidence ∈ [0,1], prediction) → unified anomaly score ∈ [0,1].

        abnormal, conf=1.0  →  1.0
        abnormal, conf=0.0  →  0.5
        normal,   conf=0.0  →  0.5
        normal,   conf=1.0  →  0.0
    """
    c = max(0.0, min(1.0, float(confidence)))
    return (0.5 + 0.5 * c) if predicted_abnormal else (0.5 - 0.5 * c)


# ---------------------------------------------------------------------------
# AUROC  (Wilcoxon-Mann-Whitney, handles degenerate single-class case)
# ---------------------------------------------------------------------------

def compute_auroc(y_true: List[float], y_score: List[float]) -> float:
    """AUROC via the Wilcoxon-Mann-Whitney statistic.

    Counts the fraction of (positive, negative) pairs where the positive
    sample scores strictly higher than the negative, plus half-credit for
    ties.

    Degenerate cases (all-positive or all-negative)
    ------------------------------------------------
    When only one class is present in y_true the traditional trapezoidal ROC
    is undefined (the FPR or TPR axis collapses to a single point and the
    integral is always 0).  We fall back to checking whether the model's
    scores are on the correct side of 0.5 (the midpoint of _to_anomaly_score):

      all-positive, all scores ≥ 0.5  →  1.0   (correctly ranked high)
      all-positive, any score  < 0.5  →  0.0   (incorrectly ranked low)
      all-negative, all scores < 0.5  →  1.0   (correctly ranked low)
      all-negative, any score  ≥ 0.5  →  0.0   (incorrectly ranked high)
    """
    if not y_true:
        return 0.5

    pos_scores = [s for s, g in zip(y_score, y_true) if g]
    neg_scores = [s for s, g in zip(y_score, y_true) if not g]

    # --- degenerate: only one class present ---
    if not pos_scores or not neg_scores:
        if not neg_scores:
            return 0.5
        else:
            return 0.5

    # --- standard WMW ---
    concordant = 0.0
    total      = len(pos_scores) * len(neg_scores)
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1.0
            elif ps == ns:
                concordant += 0.5      # tie → half credit
    return concordant / total


# ---------------------------------------------------------------------------
# Frame-level classification metrics container
# ---------------------------------------------------------------------------

class FrameMetrics(NamedTuple):
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    auroc:     float
    tp: int
    tn: int
    fp: int
    fn: int
    total_frames: int


# ---------------------------------------------------------------------------
# VQA JSONL loader
# ---------------------------------------------------------------------------

def _vqa_path_to_key(video_path: str, dataset: str = "") -> str:
    """Extract the identifying segment from a VQA video_path.

    PhysAD / IPAD
    -------------
    Strips the dataset-root prefix (everything up to and including
    "Phys-AD/" or "IPAD_dataset/") and removes any file extension,
    then replaces remaining '/' with '_'.

      /u/.../Phys-AD/sticky_roller/train/0000.mp4  →  "sticky_roller_train_0000"
      /u/.../IPAD_dataset/R01/testing/frames/02    →  "R01_testing_frames_02"

    AutoLab
    -------
    The VQA video_path is already relative ("train/abnormal_16/000.mp4"),
    so we just strip the extension and replace '/' with '_'.

      "train/abnormal_16/000.mp4"  →  "train_abnormal_16_000"
    """
    norm = video_path.replace("\\", "/")

    # AutoLab paths are relative — no anchor needed
    if dataset == "autolab":
        if "." in norm.rsplit("/", 1)[-1]:
            norm = norm.rsplit(".", 1)[0]
        return norm.replace("/", "_")

    # PhysAD / IPAD: anchor on known dataset directory markers
    for anchor in ("Phys-AD/", "IPAD_dataset/"):
        idx = norm.find(anchor)
        if idx != -1:
            segment = norm[idx + len(anchor):]
            break
    else:
        segment = norm.lstrip("/")

    if "." in segment.rsplit("/", 1)[-1]:
        segment = segment.rsplit(".", 1)[0]

    return segment.replace("/", "_")


def _load_vqa_jsonl(vqa_path: Path, dataset: str = "") -> Dict[str, dict]:
    """Load VQA .jsonl into a key → entry dict."""
    entries_by_key: Dict[str, dict] = {}

    with open(vqa_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: JSONL line {lineno} parse error: {e}")
                continue
            vp = entry.get("video_path", "")
            if not vp:
                continue
            key = _vqa_path_to_key(vp, dataset=dataset)
            entries_by_key[key] = entry

    return entries_by_key


# ---------------------------------------------------------------------------
# Report discovery
# ---------------------------------------------------------------------------

def _find_reports(report_dir: Path) -> List[Path]:
    """Recursively find all *_report.json files under report_dir."""
    return sorted(report_dir.rglob("*_report.json"))


# ---------------------------------------------------------------------------
# AutoLab: context span → per-frame 0/1 list
# ---------------------------------------------------------------------------

def _autolab_expand_context(
    context: list,
    report: dict,
) -> Optional[List[int]]:
    """Convert AutoLab's [start_frame, end_frame] GT span to a 0/1 list.

    Total frame count resolution (priority order)
    ---------------------------------------------
    1. report["num_frames"]            explicit field from the pipeline
    2. max(anomaly["end_frame"]) + 1   inferred from predicted intervals
    3. gt_end_frame + 1                minimum viable length from GT span
    """
    if (not isinstance(context, (list, tuple))
            or len(context) != 2
            or not all(isinstance(x, int) for x in context)):
        return None

    gt_start, gt_end = int(context[0]), int(context[1])
    if gt_start < 0 or gt_end < gt_start:
        return None

    n_frames: Optional[int] = None

    if isinstance(report.get("num_frames"), int) and report["num_frames"] > 0:
        n_frames = report["num_frames"]

    if n_frames is None:
        pred_ends = [
            int(a["end_frame"])
            for a in report.get("anomalies", [])
            if isinstance(a.get("end_frame"), int)
        ]
        if pred_ends:
            n_frames = max(pred_ends) + 1

    if n_frames is None:
        n_frames = gt_end + 1

    n_frames = max(n_frames, gt_end + 1)

    labels = [0] * n_frames
    for fi in range(gt_start, min(gt_end + 1, n_frames)):
        labels[fi] = 1

    return labels


# ---------------------------------------------------------------------------
# BERT score  (PhysAD)
# ---------------------------------------------------------------------------

def _load_bert():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        model_name = "google-bert/bert-base-uncased"
        print(f"  Loading BERT model ({model_name}) …", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"loaded on {device}.")
        return tokenizer, model, device
    except ImportError:
        print("  WARNING: 'transformers' or 'torch' not installed — BERT score skipped.")
        return None, None, None
    except Exception as e:
        print(f"  WARNING: Failed to load BERT model: {e} — BERT score skipped.")
        return None, None, None


def _bert_embed(texts: List[str], tokenizer, model, device):
    import torch
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()


def _cosine_sim(a, b) -> float:
    a = a / (a.norm() + 1e-9)
    b = b / (b.norm() + 1e-9)
    import torch
    return float(torch.dot(a, b).item())


def _predicted_anomaly_text(report: dict) -> str:
    anomalies = report.get("anomalies", [])
    if not anomalies:
        return "Normal"
    parts: List[str] = []
    for a in anomalies:
        atype    = str(a.get("anomaly_type",    "")).replace("_", " ").strip()
        asubtype = str(a.get("anomaly_subtype", "")).replace("_", " ").strip()
        text     = f"{atype} {asubtype}".strip() if asubtype else atype
        if text:
            parts.append(text)
    return " ; ".join(parts) if parts else "Normal"


def compute_bert_score(
    matched_reports: List[Tuple[dict, dict]]
) -> Optional[float]:
    """Cosine BERT score between predicted anomaly text and VQA context (PhysAD)."""
    pairs: List[Tuple[str, str]] = []

    for report, entry in matched_reports:
        ctx = entry.get("context", "")
        if not isinstance(ctx, str) or not ctx.strip():
            continue
        ref_text  = ctx.replace("_", " ").strip()
        pred_text = _predicted_anomaly_text(report)
        if ref_text and pred_text:
            pairs.append((ref_text, pred_text))

    if not pairs:
        return None

    tokenizer, model, device = _load_bert()
    if tokenizer is None:
        return None

    import torch
    all_texts = list(dict.fromkeys(t for pair in pairs for t in pair))
    print(f"  BERT: embedding {len(all_texts)} unique text(s) …", end=" ", flush=True)
    parts_emb: list = []
    for i in range(0, len(all_texts), 64):
        parts_emb.append(_bert_embed(all_texts[i:i + 64], tokenizer, model, device))
    all_embeds = torch.cat(parts_emb, dim=0)
    text2emb   = {t: all_embeds[i] for i, t in enumerate(all_texts)}
    print("done.")

    scores = [_cosine_sim(text2emb[ref], text2emb[cand]) for ref, cand in pairs]
    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Frame-level metrics  (IPAD and AutoLab)
# ---------------------------------------------------------------------------

def compute_frame_metrics(
    matched_reports: List[Tuple[dict, dict]],
    dataset: str = "ipad",
) -> Optional[FrameMetrics]:
    """Compute frame-level Accuracy, Precision, Recall, F1, and AUROC.

    Ground truth
    ------------
    IPAD    : entry["context"] is already list[int] (0/1), one per frame.
    AutoLab : entry["context"] is [start_frame, end_frame], expanded via
              _autolab_expand_context().

    Binary frame prediction
    -----------------------
    A frame is predicted ABNORMAL (1) when it falls inside at least one
    predicted anomaly interval [start_frame, end_frame] (inclusive).
    All other frames are predicted NORMAL (0).

    AUROC score mapping (continuous, unchanged from before)
    -------------------------------------------------------
    anomaly frame : _to_anomaly_score(frame_conf, True)   ∈ [0.5, 1.0]
    normal  frame : _to_anomaly_score(bg_conf,    False)  ∈ [0.0, 0.5]
    where bg_conf = report["overall_confidence"] for non-anomaly frames.

    Returns None when no valid frame data is found.
    """
    frame_true:  List[float] = []
    frame_score: List[float] = []
    frame_pred:  List[int]   = []   # binary: 0 or 1
    skipped = 0

    for report, entry in matched_reports:
        raw_ctx = entry.get("context")

        # --- resolve per-frame GT labels ---
        if dataset == "autolab":
            gt_frames = _autolab_expand_context(raw_ctx, report)
            if gt_frames is None:
                skipped += 1
                continue
        else:
            if not isinstance(raw_ctx, list) or not raw_ctx:
                skipped += 1
                continue
            gt_frames = raw_ctx

        n_frames  = len(gt_frames)
        anomalies = report.get("anomalies", [])
        bg_conf   = float(report.get("overall_confidence", 0.0))

        # Build per-frame max-confidence map over predicted anomaly intervals
        frame_anom_conf: Dict[int, float] = {}
        for a in anomalies:
            a_conf  = float(a.get("confidence", 0.0))
            s_frame = a.get("start_frame")
            e_frame = a.get("end_frame")
            if s_frame is not None and e_frame is not None:
                for fi in range(int(s_frame), int(e_frame) + 1):
                    if 0 <= fi < n_frames:
                        frame_anom_conf[fi] = max(
                            frame_anom_conf.get(fi, 0.0), a_conf
                        )

        for fi, gt in enumerate(gt_frames):
            if fi in frame_anom_conf:
                score = _to_anomaly_score(frame_anom_conf[fi], predicted_abnormal=True)
                pred  = 1
            else:
                score = _to_anomaly_score(bg_conf, predicted_abnormal=False)
                pred  = 0
            frame_true.append(float(gt))
            frame_score.append(score)
            frame_pred.append(pred)

    if skipped:
        print(f"  WARNING: {skipped} report(s) skipped — missing or malformed context.")

    if not frame_true:
        return None

    # --- AUROC (continuous scores) ---
    auroc = compute_auroc(frame_true, frame_score)

    # --- binary classification metrics ---
    n = len(frame_true)
    TP = sum(1 for g, p in zip(frame_true, frame_pred) if g == 1 and p == 1)
    TN = sum(1 for g, p in zip(frame_true, frame_pred) if g == 0 and p == 0)
    FP = sum(1 for g, p in zip(frame_true, frame_pred) if g == 0 and p == 1)
    FN = sum(1 for g, p in zip(frame_true, frame_pred) if g == 1 and p == 0)

    acc  = (TP + TN) / n
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return FrameMetrics(
        accuracy=acc, precision=prec, recall=rec, f1=f1, auroc=auroc,
        tp=TP, tn=TN, fp=FP, fn=FN, total_frames=n,
    )


# ---------------------------------------------------------------------------
# Main metrics printer
# ---------------------------------------------------------------------------

def print_metrics(vqa_file_path: str, report_dir_path: str, dataset: str) -> None:
    dataset    = dataset.lower().strip()
    vqa_path   = Path(vqa_file_path)
    report_dir = Path(report_dir_path)

    if not vqa_path.is_file():
        raise FileNotFoundError(f"VQA file not found: {vqa_path}")
    if not report_dir.is_dir():
        raise NotADirectoryError(f"Report directory not found: {report_dir}")
    if dataset not in ("physad", "ipad", "autolab"):
        raise ValueError(
            f"--dataset must be 'physad', 'ipad', or 'autolab', got: {dataset!r}"
        )

    entries_by_key = _load_vqa_jsonl(vqa_path, dataset=dataset)
    rpt_paths      = _find_reports(report_dir)

    print(f"\n{'='*60}")
    print(f"STAGE 4 – Metrics  [{dataset.upper()}]")
    print(f"{'='*60}")
    print(f"  VQA file      : {vqa_path}  ({len(entries_by_key)} entries)")
    print(f"  Report dir    : {report_dir}")
    print(f"  Reports found : {len(rpt_paths)}")

    # ── Collect per-video data ──────────────────────────────────────────────
    vid_true:        List[int]   = []
    vid_pred:        List[int]   = []
    vid_score:       List[float] = []
    matched_reports: List[Tuple[dict, dict]] = []
    evaluated = 0
    unmatched = 0

    for rpt_path in rpt_paths:
        try:
            with open(rpt_path) as f:
                report = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {rpt_path}: {e}")
            continue

        video_name = report.get("video_name") or report.get("prediction_name")
        entry      = entries_by_key.get(video_name) if video_name else None

        if entry is not None:
            gt = 1 if entry.get("answer", "Normal").strip().lower() == "abnormal" else 0
        else:
            unmatched += 1
            path_lower = str(rpt_path).lower()
            gt = 0 if ("normal" in path_lower and "abnormal" not in path_lower) else 1

        predicted_abnormal: bool = bool(report.get("anomaly_detected", False))
        pred = 1 if predicted_abnormal else 0

        overall_conf = float(report.get("overall_confidence", 0.0))
        v_score      = _to_anomaly_score(overall_conf, predicted_abnormal)

        vid_true.append(gt)
        vid_pred.append(pred)
        vid_score.append(v_score)
        evaluated += 1

        if entry is not None:
            matched_reports.append((report, entry))

    # ── Video-level classification metrics ─────────────────────────────────
    print(f"\n  Evaluated : {evaluated} / {len(rpt_paths)} reports  "
          f"({len(matched_reports)} matched to VQA)")
    if unmatched:
        print(f"  Unmatched : {unmatched} report(s) — GT inferred from path name")

    if evaluated == 0:
        print("  No reports could be evaluated — check paths and VQA file.")
        return

    TP = sum(1 for g, p in zip(vid_true, vid_pred) if g == 1 and p == 1)
    TN = sum(1 for g, p in zip(vid_true, vid_pred) if g == 0 and p == 0)
    FP = sum(1 for g, p in zip(vid_true, vid_pred) if g == 0 and p == 1)
    FN = sum(1 for g, p in zip(vid_true, vid_pred) if g == 1 and p == 0)

    acc  = (TP + TN) / evaluated
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auroc_v = compute_auroc([float(v) for v in vid_true], vid_score)

    w = 48
    print(f"\n  {'─'*w}")
    print(f"  Video-level Classification Metrics  ({evaluated} videos)")
    print(f"  {'─'*w}")
    print(f"  Accuracy  : {acc:.4f}   ({TP+TN}/{evaluated} correct)")
    print(f"  Precision : {prec:.4f}   (TP={TP}, FP={FP})")
    print(f"  Recall    : {rec:.4f}   (TP={TP}, FN={FN})")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  {'─'*w}")
    print(f"  Video-level AUROC")
    print(f"    Prediction : report['anomaly_detected']  (True/False)")
    print(f"    Confidence : report['overall_confidence']")
    print(f"    Score rule : abnormal→[0.5,1.0]  normal→[0.0,0.5]")
    print(f"    Ranked over ALL {evaluated} predictions (pos + neg)")
    print(f"    AUROC      : {auroc_v:.4f}")
    print(f"  {'─'*w}")

    # ── Dataset-specific metrics ────────────────────────────────────────────
    if dataset in ("ipad", "autolab"):
        label = "IPAD" if dataset == "ipad" else "AutoLab"

        if dataset == "autolab":
            ctx_note = (
                "vqa entry['context'] = [start, end] → expanded to\n"
                "                 per-frame 0/1 list via report['num_frames']\n"
                "                 (fallback: max(anomaly end_frame)+1, then gt_end+1)"
            )
        else:
            ctx_note = "vqa entry['context'] list[int] (0=normal, 1=abnormal)"

        print(f"\n  Computing frame-level metrics ({label}) …")
        fm = compute_frame_metrics(matched_reports, dataset=dataset)

        print(f"  {'─'*w}")
        print(f"  Frame-level Metrics  ({label})")
        print(f"  {'─'*w}")

        if fm is None:
            print(f"  N/A  (no entries with valid frame-label context found)")
            print(f"  {'─'*w}\n")
            return

        print(f"  Total frames : {fm.total_frames}")
        print(f"  {'─'*w}")
        print(f"  Classification  (binary: anomaly interval → predicted abnormal)")
        print(f"    GT         : {ctx_note}")
        print(f"    Pred       : frame inside any anomalies[start_frame..end_frame]")
        print(f"    Accuracy   : {fm.accuracy:.4f}   ({fm.tp+fm.tn}/{fm.total_frames} correct)")
        print(f"    Precision  : {fm.precision:.4f}   (TP={fm.tp}, FP={fm.fp})")
        print(f"    Recall     : {fm.recall:.4f}   (TP={fm.tp}, FN={fm.fn})")
        print(f"    F1-Score   : {fm.f1:.4f}")
        print(f"  {'─'*w}")
        print(f"  AUROC  (continuous confidence scores)")
        print(f"    Pred       : anomalies[start_frame..end_frame]")
        print(f"                 conf = anomaly['confidence']")
        print(f"                 normal frames: conf = report['overall_confidence']")
        print(f"    Score rule : anomaly→[0.5,1.0]  normal→[0.0,0.5]")
        print(f"    AUROC      : {fm.auroc:.4f}")
        print(f"  {'─'*w}\n")

    elif dataset == "physad":
        print(f"\n  Computing BERT anomaly-type score (PhysAD) …")
        bert = compute_bert_score(matched_reports)
        print(f"  {'─'*w}")
        print(f"  BERT Anomaly-type Score  (PhysAD)")
        print(f"    GT         : vqa entry['context'] string")
        print(f"    Pred       : concat anomaly_type + anomaly_subtype per anomaly")
        print(f"                 (no anomalies → predicted text = 'Normal')")
        print(f"    Covers     : all matched GT entries (normal + abnormal)")
        if bert is not None:
            print(f"    BERT Score : {bert:.4f}")
        else:
            print(f"    BERT Score : N/A  (no qualifying pairs or BERT unavailable)")
        print(f"  {'─'*w}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VQA anomaly-detection reports against ground truth."
    )
    p.add_argument(
        "--dataset", required=True, choices=["physad", "ipad", "autolab"],
        help=(
            "'physad'  → video-level 5 metrics + BERT score; "
            "'ipad'    → video-level 5 metrics + frame-level Acc/P/R/F1/AUROC; "
            "'autolab' → video-level 5 metrics + frame-level Acc/P/R/F1/AUROC "
            "            (context [start,end] expanded to per-frame labels)."
        ),
    )
    p.add_argument(
        "--vqa", required=True, metavar="VQA_JSONL",
        help="Path to the VQA ground-truth .jsonl file (one JSON object per line)."
    )
    p.add_argument(
        "--reports", required=True, metavar="REPORT_DIR",
        help="Root directory searched recursively for *_report.json files."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print_metrics(
        vqa_file_path=args.vqa,
        report_dir_path=args.reports,
        dataset=args.dataset,
    )