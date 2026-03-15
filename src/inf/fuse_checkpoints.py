#!/usr/bin/env python3
"""
Fuse answers from three checkpoint jsonl files using a simple 3-choose-2 majority vote.
Each input line is a JSON object with at least a `solution` field that contains an
`<answer>A|B</answer>` tag. The script outputs a JSONL file with fused_answer and
vote_counts for each item, preserving the original `images` list.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Sequence


CKPT_PATHS = [

]
# 若提供权重则与 CKPT_PATHS 对应；None 或长度不匹配时自动均分。
# CKPT_WEIGHTS: Sequence[float] | None = None
CKPT_WEIGHTS = [
    0,  # 权重示例：checkpoint-750 权重为 0.5
    1,  # checkpoint-960 权重为 0.5
]

# 输出文件路径
OUTPUT_PATH = Path(r"C:\Users\leozx\Downloads\fused8039.jsonl")
# 是否严格要求 images 列表内容一致（开启能更严谨，默认开启）。顺序可选由
# ORDER_INSENSITIVE 控制。
STRICT_IMAGES = True
# 是否在对齐时忽略 images 的顺序（默认忽略顺序，以免不同 checkpoint 顺序不同无法匹配）。
ORDER_INSENSITIVE = True
# 是否移除含有 "crop" 字样的图片路径（与 remove_crops_from_jsonl 相同逻辑）
REMOVE_CROPS = True

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)


def filter_crops(imgs: List[Any]) -> List[str]:
    """Remove paths containing "crop" (case-insensitive)."""
    return [str(p) for p in imgs if "crop" not in str(p).lower()]


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    removed = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if REMOVE_CROPS and isinstance(rec.get("images"), list):
                    original = rec["images"]
                    filtered = filter_crops(original)
                    removed += len(original) - len(filtered)
                    rec["images"] = filtered
                records.append(rec)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON at {path}:{lineno}: {exc}") from exc
    return records, removed


def extract_answer(solution: str) -> str:
    match = ANSWER_RE.search(solution)
    if not match:
        raise ValueError(f"No <answer> tag found in solution: {solution[:80]}...")
    return match.group(1).upper()


def replace_answer_tag(solution: str, new_answer: str) -> str:
    if not ANSWER_RE.search(solution):
        raise ValueError("No <answer> tag to replace in solution")
    return ANSWER_RE.sub(f"<answer>{new_answer}</answer>", solution, count=1)


def weighted_vote(answers: List[str], weights: Sequence[float], tie_eps: float = 1e-6) -> str:
    if len(answers) != len(weights):
        raise ValueError("answers and weights length mismatch")
    score = {"A": 0.0, "B": 0.0}
    for ans, w in zip(answers, weights):
        ans = ans.upper()
        if ans not in ("A", "B"):
            raise ValueError(f"Invalid answer: {ans}")
        score[ans] += w
    if abs(score["A"] - score["B"]) <= tie_eps:
        # 平票时回退到权重最高的模型答案
        max_idx = max(range(len(weights)), key=lambda i: weights[i])
        return answers[max_idx].upper()
    return "A" if score["A"] > score["B"] else "B"


def images_key(rec: Dict[str, Any]) -> Tuple[str, ...]:
    imgs = rec.get("images")
    if not isinstance(imgs, list) or not imgs:
        raise ValueError("Record missing non-empty 'images' list for alignment")
    normalized = [str(p) for p in imgs]
    if ORDER_INSENSITIVE:
        return tuple(sorted(normalized))
    return tuple(normalized)


def index_by_images(records: List[Dict[str, Any]], label: str) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    mapping: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for idx, rec in enumerate(records):
        key = images_key(rec)
        if key in mapping:
            raise ValueError(f"Duplicate images key in {label} at record {idx}")
        mapping[key] = rec
    return mapping


def fuse_records(all_records: List[List[Dict[str, Any]]], strict_images: bool, weights: Sequence[float]) -> List[Dict[str, Any]]:
    if not all_records:
        return []

    maps = [index_by_images(rec, f"ckpt{i}") for i, rec in enumerate(all_records)]
    key_set = set(maps[0].keys())
    for i, m in enumerate(maps[1:], 1):
        if key_set != set(m.keys()):
            raise ValueError(f"Image sets do not match across checkpoints (ckpt0 vs ckpt{i})")

    fused: List[Dict[str, Any]] = []
    for key in maps[0].keys():  # 保持首个 checkpoint 的顺序
        recs = [m[key] for m in maps]
        if strict_images:
            norm_keys = [images_key(r) for r in recs]
            if len(set(norm_keys)) != 1:
                raise ValueError(f"Images mismatch for key {key}")

        answers = [extract_answer(str(r.get("solution", ""))) for r in recs]
        fused_answer = weighted_vote(answers, weights)
        vote_counts = Counter(answers)

        fused.append({
            "images": list(key),
            "checkpoint_answers": {f"ckpt{i}": ans for i, ans in enumerate(answers)},
            "vote_counts": dict(vote_counts),
            "fused_answer": fused_answer,
        })
    return fused


def apply_fused_to_ckpt1(ckpt1: List[Dict[str, Any]], fused: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fused_map = {tuple(rec["images"]): rec["fused_answer"] for rec in fused}
    updated: List[Dict[str, Any]] = []
    for idx, rec in enumerate(ckpt1):
        key = images_key(rec)
        if key not in fused_map:
            raise KeyError(f"No fused result found for ckpt1 record at index {idx}")
        new_rec = dict(rec)
        new_rec["solution"] = replace_answer_tag(str(rec.get("solution", "")), fused_map[key])
        updated.append(new_rec)
    return updated


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if len(CKPT_PATHS) < 1:
        raise ValueError("CKPT_PATHS must contain at least one path")

    for p in CKPT_PATHS:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    loaded = [load_jsonl(p) for p in CKPT_PATHS]
    records = [r for r, _ in loaded]
    removed_total = sum(rem for _, rem in loaded)

    # 处理权重：未提供或长度不符则均分
    if CKPT_WEIGHTS is None or len(CKPT_WEIGHTS) != len(records):
        weights = [1.0 / len(records)] * len(records)
    else:
        weights = list(float(w) for w in CKPT_WEIGHTS)

    fused = fuse_records(records, strict_images=STRICT_IMAGES, weights=weights)
    updated_ckpt1 = apply_fused_to_ckpt1(records[0], fused)
    save_jsonl(updated_ckpt1, OUTPUT_PATH)
    msg = f"Fused {len(fused)} records and wrote updated ckpt1 -> {OUTPUT_PATH}"
    if REMOVE_CROPS:
        msg += f" (removed {removed_total} cropped paths)"
    print(msg)


if __name__ == "__main__":
    main()
