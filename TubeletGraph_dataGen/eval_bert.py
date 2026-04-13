"""
bert_score_physad.py  –  Per-subset Sentence-BERT score for PhysAD model outputs.

Two input modes for --outputs:
  FILE  : a JSON list of samples (original format).
          ref  = ground_truth["detail"]
          pred = prediction["json_report"]["anomaly_type"]

  DIR   : a directory of per-sample JSON files.
          Each filename encodes object / id / context abbreviation, e.g.
            ball_0000_ball_leak.json
            ball_0000_ball_insu.json
          Matching to a VQA entry is done by:
            1. Parse stem -> object_slug + zero-padded id + ctx_tokens
            2. Scan VQA entries for that object whose video id matches
            3. Among those, pick the entry whose context string best
               overlaps with the ctx_tokens from the filename
          ref  = vqa_entry["org_subset"] + " " + vqa_entry["context"]
          pred = entire JSON file content (as a string)

Subset is read from vqa["org_subset"] in both modes.

Usage
-----
# original list-of-samples JSON
python bert_score_physad.py \\
    --outputs /path/to/outputs.json \\
    --vqa     /path/to/physad_vqa.jsonl

# directory of per-sample JSONs
python bert_score_physad.py \\
    --outputs /path/to/output_dir/ \\
    --vqa     /path/to/physad_vqa.jsonl

# choose embedding model  (default: sbert)
python bert_score_physad.py \\
    --outputs /path/to/output_dir/ \\
    --vqa     /path/to/physad_vqa.jsonl \\
    --model   bert          # google-bert/bert-base-uncased  [CLS cosine]
    # or
    --model   sbert         # all-MiniLM-L6-v2               [sentence cosine]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model loading + embedding  (bert  or  sbert)
# ---------------------------------------------------------------------------

def _load_model(model_choice: str):
    """Load and return the chosen embedding model.

    Parameters
    ----------
    model_choice : "bert" | "sbert"

    Returns
    -------
    A tuple  (tag, model_obj)  where tag is the short string used in printing,
    or  (None, None)  on failure.
    """
    if model_choice == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
            name = "all-MiniLM-L6-v2"
            print(f"  Loading SentenceBERT ({name}) ...", end=" ", flush=True)
            m = SentenceTransformer(name)
            print("loaded.")
            return "sbert", m
        except ImportError:
            print("  WARNING: 'sentence_transformers' not installed -- score skipped.")
            print("           Install with: pip install sentence-transformers")
            return None, None
        except Exception as e:
            print(f"  WARNING: Failed to load SentenceBERT: {e} -- score skipped.")
            return None, None

    else:  # "bert"
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            name = "google-bert/bert-base-uncased"
            print(f"  Loading BERT ({name}) ...", end=" ", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model     = AutoModel.from_pretrained(name)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"loaded on {device}.")
            return "bert", (tokenizer, model, device)
        except ImportError:
            print("  WARNING: 'transformers' or 'torch' not installed -- score skipped.")
            return None, None
        except Exception as e:
            print(f"  WARNING: Failed to load BERT: {e} -- score skipped.")
            return None, None


def _embed(texts: List[str], tag: str, model_obj) -> "torch.Tensor":
    """Embed *texts* and return an (N, D) float32 tensor of L2-normed vectors."""
    import torch

    if tag == "sbert":
        vecs = model_obj.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()
        return vecs

    else:  # "bert"  – CLS token, then L2-normalise
        tokenizer, model, device = model_obj
        parts = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].cpu()
            parts.append(cls)
        vecs = torch.cat(parts, dim=0)
        # L2-normalise so cosine = dot
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return vecs / norms


def _cosine_sim(a, b) -> float:
    """Cosine similarity for two already L2-normalised vectors."""
    import torch
    return float(torch.dot(a, b).item())


# ---------------------------------------------------------------------------
# VQA loader
# ---------------------------------------------------------------------------

def _vqa_key(video_path: str) -> str:
    """Anchor at 'Phys-AD/', strip extension, replace '/' -> '_'.

    /u/.../Phys-AD/rubber_band/test/norm/0000.mp4  ->  rubber_band_test_norm_0000
    """
    norm = video_path.replace("\\", "/")
    idx  = norm.find("Phys-AD/")
    seg  = norm[idx + len("Phys-AD/"):] if idx != -1 else norm.lstrip("/")
    if "." in seg.rsplit("/", 1)[-1]:
        seg = seg.rsplit(".", 1)[0]
    return seg.replace("/", "_")


def _subset_from_entry(entry: dict) -> str:
    if entry.get("org_subset"):
        return str(entry["org_subset"]).strip()
    vp  = entry.get("video_path", "").replace("\\", "/")
    idx = vp.find("Phys-AD/")
    return vp[idx + len("Phys-AD/"):].split("/")[0] if idx != -1 else "unknown"


def load_vqa(vqa_path: Path) -> Dict[str, dict]:
    """Return {vqa_key: entry}.  Also stores a bare-stem key as fallback."""
    index: Dict[str, dict] = {}
    with open(vqa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            vp = entry.get("video_path", "")
            if not vp:
                continue
            index[_vqa_key(vp)]  = entry
            index[Path(vp).stem] = entry   # bare stem fallback (e.g. "0000")
    return index


def _build_obj_index(vqa_index: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Group unique VQA entries by org_subset name."""
    seen: set = set()
    by_obj: Dict[str, List[dict]] = defaultdict(list)
    for entry in vqa_index.values():
        eid = id(entry)
        if eid in seen:
            continue
        seen.add(eid)
        obj = _subset_from_entry(entry)
        by_obj[obj].append(entry)
    return by_obj


# ---------------------------------------------------------------------------
# Mode A – original list-of-samples JSON
# ---------------------------------------------------------------------------

def _ref_text_sample(sample: dict) -> str:
    detail = str(sample.get("ground_truth", {}).get("detail", "")).strip()
    return detail.replace("_", " ") if detail else ""


def _pred_text_sample(sample: dict) -> str:
    jr    = sample.get("prediction", {}).get("json_report", {}) or {}
    atype = str(jr.get("anomaly_type", "")).strip()
    if atype.lower() in ("none", "n/a", ""):
        return "normal"
    return atype.replace("_", " ")


def _get_subset_sample(sample: dict, vqa_index: Dict[str, dict]) -> str:
    """Match sample_id stem to a VQA entry and return its subset label."""
    sid_stem = Path(str(sample.get("sample_id", ""))).stem

    entry = vqa_index.get(sid_stem)

    if entry is None:
        parts = sid_stem.split("_")
        for tok in ("test", "train"):
            for i in range(1, len(parts)):
                candidate = "_".join(parts[:i] + [tok] + parts[i:])
                entry = vqa_index.get(candidate)
                if entry:
                    break
            if entry:
                break

    if entry is None:
        m = re.search(r'(\d+)$', sid_stem)
        if m:
            entry = vqa_index.get(m.group(1).zfill(4))

    return _subset_from_entry(entry) if entry is not None else "unknown"


def load_triples_from_file(
    outputs_path: Path,
    vqa_index: Dict[str, dict],
) -> List[Tuple[str, str, str]]:
    """Return [(subset, ref, pred), ...] for list-of-samples JSON."""
    with open(outputs_path) as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("outputs JSON must be a top-level list.")
    print(f"  {len(samples)} sample(s).")

    triples: List[Tuple[str, str, str]] = []
    for s in samples:
        ref  = _ref_text_sample(s)
        pred = (
            _pred_text_sample(s)
            + " have anomalies or not: {}".format(
                s["prediction"]["json_report"]["is_anomalous"]
            )
        )
        if not ref or not pred:
            continue
        sub = _get_subset_sample(s, vqa_index)
        triples.append((sub, ref, pred))
    return triples


# ---------------------------------------------------------------------------
# Mode B – directory of per-sample JSON files
# ---------------------------------------------------------------------------

def _parse_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """Parse a filename stem into (object_slug, zero_padded_id, ctx_suffix).

    The zero-padded numeric ID (3+ consecutive digits) is the anchor.
      Everything before it -> object_slug  (e.g. "ball", "rubber_band")
      Everything after  it -> ctx_suffix   (e.g. "ball_leak", "ball_insu")

    Examples
    --------
    "ball_0000_ball_leak"        -> ("ball",        "0000", "ball_leak")
    "ball_0000_ball_insu"        -> ("ball",        "0000", "ball_insu")
    "rubber_band_0001_rb_broken" -> ("rubber_band", "0001", "rb_broken")
    "ball_0000"                  -> ("ball",        "0000", "")
    """
    # Find the FIRST run of >= 3 digits, possibly surrounded by underscores
    m = re.search(r'(?:(?<=_)|(?<=^))(\d{3,})(?=_|$)', stem)
    if not m:
        return None

    id_str = m.group(1).zfill(4)

    # object slug: everything strictly before the digit run (strip trailing _)
    obj_slug = stem[: m.start()].rstrip("_")

    # context suffix: everything strictly after the digit run (strip leading _)
    ctx_suffix = stem[m.end():].lstrip("_")

    return obj_slug, id_str, ctx_suffix


def _token_overlap_score(ctx_vqa: str, ctx_suffix: str) -> int:
    """Count how many suffix tokens (len > 2) are substrings of ctx_vqa.

    Both strings are lowercased before comparison.

    Examples
    --------
    ctx_vqa="leakage"          ctx_suffix="ball_leak"
        tokens = ["ball", "leak"]  -> "leak" in "leakage" -> score 1

    ctx_vqa="insufficient_gas" ctx_suffix="ball_insu"
        tokens = ["ball", "insu"]  -> "insu" in "insufficient_gas" -> score 1

    ctx_vqa="norm"             ctx_suffix="ball_norm"
        tokens = ["ball", "norm"]  -> "norm" in "norm" -> score 1
    """
    tokens    = [t for t in re.split(r'[_\-\s]+', ctx_suffix.lower()) if len(t) > 2]
    vqa_lower = ctx_vqa.lower()
    return sum(1 for t in tokens if t in vqa_lower)


def _find_vqa_entry_for_file(
    json_stem: str,
    vqa_index: Dict[str, dict],
    vqa_entries_by_obj: Dict[str, List[dict]],
) -> Optional[dict]:
    """Locate the VQA entry matching a per-sample JSON filename stem.

    Algorithm
    ---------
    1. Parse stem -> (object_slug, id_str, ctx_suffix)
    2. Direct key lookup via vqa_index (fast path; works when ctx is fully spelled out).
    3. Fuzzy fallback:
       a. Collect all VQA entries for object_slug (by org_subset).
       b. Filter to entries whose video id matches id_str.
       c. Rank by _token_overlap_score(entry["context"], ctx_suffix).
       d. Return highest-scoring entry.
    """
    parsed = _parse_filename(json_stem)
    if parsed is None:
        print(f"    WARN: cannot parse filename '{json_stem}'")
        return None

    obj_slug, id_str, ctx_suffix = parsed

    # ---- Step 2: direct key lookup -------------------------------------------
    for split in ("test", "train"):
        if ctx_suffix:
            key = f"{obj_slug}_{split}_{ctx_suffix}_{id_str}"
            if key in vqa_index:
                return vqa_index[key]
        key = f"{obj_slug}_{split}_{id_str}"
        if key in vqa_index:
            return vqa_index[key]

    # ---- Step 3: fuzzy scan --------------------------------------------------
    # Collect candidates by object
    candidates: List[dict] = vqa_entries_by_obj.get(obj_slug, [])

    if not candidates:
        # Try partial-match on org_subset (e.g. multi-word slugs)
        for obj_key, entries in vqa_entries_by_obj.items():
            if obj_slug in obj_key or obj_key in obj_slug:
                candidates = entries
                break

    # Filter by matching numeric video id
    id_candidates = [
        e for e in candidates
        if Path(e.get("video_path", "")).stem.zfill(4) == id_str
    ]

    if not id_candidates:
        return None

    if not ctx_suffix:
        # No context info in filename -- return first id-match
        return id_candidates[0]

    # Rank by token-overlap between filename ctx_suffix and VQA entry context
    scored = sorted(
        id_candidates,
        key=lambda e: _token_overlap_score(e.get("context", ""), ctx_suffix),
        reverse=True,
    )

    best = scored[0]
    if _token_overlap_score(best.get("context", ""), ctx_suffix) == 0:
        print(
            f"    WARN: no context-token overlap for '{json_stem}' "
            f"(ctx_suffix='{ctx_suffix}'); using first id-match "
            f"(context='{best.get('context', '')}')."
        )
    return best


def _ref_text_dir(entry: dict) -> str:
    """ref = org_subset + context (spaces, no underscores)."""
    org_subset = str(entry.get("org_subset", "")).strip().replace("_", " ")
    context    = str(entry.get("context",    "")).strip().replace("_", " ")
    parts = [p for p in (org_subset, context) if p]
    return " ".join(parts)


def load_triples_from_dir(
    outputs_dir: Path,
    vqa_index: Dict[str, dict],
) -> List[Tuple[str, str, str]]:
    """Return [(subset, ref, pred), ...] for a directory of per-sample JSONs.

    ref  = vqa_entry["org_subset"] + " " + vqa_entry["context"]
    pred = entire JSON file content (as a string)
    """
    json_files = sorted(outputs_dir.glob("*.json"))
    print(f"  {len(json_files)} JSON file(s) in directory.")

    vqa_entries_by_obj = _build_obj_index(vqa_index)

    triples:   List[Tuple[str, str, str]] = []
    unmatched: List[str] = []

    for jf in json_files:
        stem  = jf.stem
        entry = _find_vqa_entry_for_file(stem, vqa_index, vqa_entries_by_obj)
        if entry is None:
            unmatched.append(stem)
            continue

        ref  = _ref_text_dir(entry)
        pred = jf.read_text(encoding="utf-8").strip()

        if not ref or not pred:
            print(f"    WARN: empty ref or pred for '{stem}' -- skipped.")
            continue

        sub = _subset_from_entry(entry)
        triples.append((sub, ref, pred))

    if unmatched:
        print(f"  WARN: {len(unmatched)} file(s) could not be matched to VQA:")
        for u in unmatched[:20]:
            print(f"    * {u}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more.")

    print(f"  Matched {len(triples)} / {len(json_files)} file(s).")
    return triples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(
    triples: List[Tuple[str, str, str]],
    model_choice: str = "sbert",
) -> Optional[Tuple[Dict[str, float], Dict[str, int], str]]:
    """Embed all unique texts once; compute per-subset + overall cosine scores.

    Returns (scores, counts, model_tag) or None on failure.
    """
    if not triples:
        print("  No valid pairs found.")
        return None

    tag, model_obj = _load_model(model_choice)
    if tag is None:
        return None

    import torch

    all_texts = list(dict.fromkeys(t for _, r, p in triples for t in (r, p)))
    label = "SentenceBERT" if tag == "sbert" else "BERT"
    print(f"  {label}: embedding {len(all_texts)} unique text(s) ...", end=" ", flush=True)
    all_embeds = _embed(all_texts, tag, model_obj)
    text2emb   = {t: all_embeds[i] for i, t in enumerate(all_texts)}
    print("done.")

    by_subset: Dict[str, List[float]] = defaultdict(list)
    for sub, ref, pred in triples:
        by_subset[sub].append(_cosine_sim(text2emb[ref], text2emb[pred]))

    scores: Dict[str, float] = {}
    counts: Dict[str, int]   = {}
    all_sim: List[float]     = []
    for sub in sorted(by_subset):
        sims        = by_subset[sub]
        scores[sub] = float(sum(sims) / len(sims))
        counts[sub] = len(sims)
        all_sim.extend(sims)

    scores["__overall__"] = float(sum(all_sim) / len(all_sim))
    counts["__overall__"] = len(all_sim)
    return scores, counts, tag


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def print_results(
    scores: Dict[str, float],
    counts: Dict[str, int],
    model_tag: str = "sbert",
) -> None:
    if model_tag == "sbert":
        model_line  = "Model : all-MiniLM-L6-v2  [sentence cosine]"
        score_label = "SBERT Score"
    else:
        model_line  = "Model : google-bert/bert-base-uncased  [CLS cosine]"
        score_label = "BERT Score"

    w = 56
    print(f"\n{'='*w}")
    print(f"  PhysAD Embedding Score")
    print(f"  {model_line}")
    print(f"{'='*w}")
    print(f"  {'Subset':<28}  {'N':>5}  {score_label:>11}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*11}")
    for sub in sorted(k for k in scores if k != "__overall__"):
        print(f"  {sub:<28}  {counts[sub]:>5}  {scores[sub]:>11.4f}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*11}")
    print(f"  {'OVERALL':<28}  {counts['__overall__']:>5}  {scores['__overall__']:>11.4f}")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-subset Sentence-BERT score for PhysAD outputs."
    )
    p.add_argument(
        "--outputs", required=True,
        help=(
            "Path to outputs JSON (list of samples) OR a directory "
            "containing per-sample JSON files."
        ),
    )
    p.add_argument("--vqa", required=True, help="Path to PhysAD VQA JSONL.")
    p.add_argument(
        "--model", choices=["bert", "sbert"], default="sbert",
        help=(
            "Embedding model to use for scoring. "
            "'sbert' (default): all-MiniLM-L6-v2 via sentence-transformers. "
            "'bert': google-bert/bert-base-uncased CLS token via transformers."
        ),
    )
    return p.parse_args()


def main() -> None:
    args  = _parse_args()
    out_p = Path(args.outputs)
    vqa_p = Path(args.vqa)

    print(f"Loading VQA     : {vqa_p}")
    vqa_index = load_vqa(vqa_p)
    print(f"  {len(vqa_index)} VQA key(s).")

    if out_p.is_dir():
        print(f"Loading outputs (DIR) : {out_p}")
        triples   = load_triples_from_dir(out_p, vqa_index)
        mode_note = "ref=org_subset+context  pred=JSON file content"
    else:
        print(f"Loading outputs (FILE): {out_p}")
        triples   = load_triples_from_file(out_p, vqa_index)
        mode_note = (
            "ref=ground_truth['detail']  "
            "pred=prediction['json_report']['anomaly_type']"
        )

    print(f"  {len(triples)} valid pair(s) to score.")
    print(f"  Mode : {mode_note}")
    print(f"  Model: {args.model}")

    result = compute_scores(triples, model_choice=args.model)
    if result is None:
        print("Scoring failed.")
        return

    scores, counts, tag = result
    print_results(scores, counts, model_tag=tag)


if __name__ == "__main__":
    main()


# Baseline (Qwen3)
# ====================================================
#   PhysAD BERT Score
#   GT   : ground_truth['detail']
#   Pred : prediction['json_report']['anomaly_type']
#   Model: google-bert/bert-base-uncased  [CLS cosine]
# ====================================================
#   Subset                            N  BERT Score
#   ────────────────────────────  ─────  ──────────
#   ball                            135      0.8090
#   button                          300      0.7968
#   car                             600      0.8118
#   caster_wheel                     60      0.8263
#   clip                            360      0.7974
#   clock                           222      0.7976
#   fan                             358      0.7850
#   gear                            450      0.8067
#   hinge                            60      0.8136
#   liquid                           45      0.7538
#   lock                            180      0.7790
#   magnet                           90      0.7744
#   rolling_bearing                  56      0.8151
#   rubber_band                      60      0.7917
#   screw                            45      0.8027
#   servo                           240      0.8003
#   slide                           150      0.7800
#   spherical_bearing                60      0.8170
#   sticky_roller                    45      0.8032
#   toothpaste                       90      0.8211
#   usb                             240      0.7813
#   zipper                          180      0.7877
#   ────────────────────────────  ─────  ──────────
#   OVERALL                        4026      0.7979
# ====================================================


# Baseline (urf)
# ========================================================
#   PhysAD Embedding Score
#   Model : google-bert/bert-base-uncased  [CLS cosine]
# ========================================================
#   Subset                            N   BERT Score
#   ----------------------------  -----  -----------
#   ball                            135       0.6415
#   button                          180       0.6664
#   car                             600       0.6028
#   caster_wheel                     60       0.6261
#   clip                            240       0.6403
#   clock                           222       0.5973
#   fan                             358       0.6295
#   gear                            450       0.6434
#   hinge                            60       0.6433
#   liquid                           45       0.6396
#   lock                            180       0.6278
#   magnet                           90       0.6427
#   rolling_bearing                  56       0.6258
#   rubber_band                      60       0.6386
#   screw                            45       0.6567
#   servo                           240       0.6384
#   slide                           150       0.6394
#   spherical_bearing                60       0.6619
#   sticky_roller                    45       0.6227
#   toothpaste                       90       0.6187
#   usb                             240       0.6496
#   zipper                          180       0.6491
#   ----------------------------  -----  -----------
#   OVERALL                        3786       0.6322
# ========================================================


