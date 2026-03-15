from __future__ import annotations

import json
import re
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PRESETS = {
    "validation": {
        "ckpt_paths": [Path("src/inf/ForValidation.jsonl"), Path("checkpoint-960_validation.jsonl")],
        "ckpt_weights": [0, 1],
        "out_fused": Path("json/outputs/fused_intermediate.jsonl"),
        "out_fixed": Path("json/outputs/fused_and_fixed_validation.jsonl"),
    },
    "test": {
        "ckpt_paths": [Path("src/inf/ForTest.jsonl"), Path("checkpoint-960_test.jsonl")],
        "ckpt_weights": [0, 1],
        "out_fused": Path("json/outputs/fused_intermediate.jsonl"),
        "out_fixed": Path("json/outputs/fused_and_fixed_test.jsonl"),
    },
}

# ==================== DEFAULT CONFIG ====================
DEFAULT_CKPT_PATHS = [

]
DEFAULT_CKPT_WEIGHTS: Sequence[float] | None = [0, 1]
DEFAULT_OUTPUT_FUSED = Path("json/outputs/fused_intermediate.jsonl")
DEFAULT_OUTPUT_FIXED = Path("json/outputs/fused_and_fixed_validation.jsonl")
DEFAULT_STRICT_IMAGES = True
DEFAULT_ORDER_INSENSITIVE = True
DEFAULT_REMOVE_CROPS = True
# =========================================================

def parse_float_list(val: str | None) -> Sequence[float] | None:
    if val is None:
        return None
    parts = [p.strip() for p in val.split(',') if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse checkpoint answers then fix <thinking> A/B consistency.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Quick preset for validation/test phases",
    )
    parser.add_argument(
        "--ckpt_paths",
        nargs='+',
        type=Path,
        default=DEFAULT_CKPT_PATHS,
        help="One or more checkpoint jsonl files (first provides base solutions)",
    )
    parser.add_argument(
        "--ckpt_weights",
        type=str,
        default=None,
        help="Comma-separated weights matching ckpt_paths; omitted -> equal weights",
    )
    parser.add_argument(
        "--out_fused",
        type=Path,
        default=DEFAULT_OUTPUT_FUSED,
        help="Intermediate fused output path",
    )
    parser.add_argument(
        "--out_fixed",
        type=Path,
        default=DEFAULT_OUTPUT_FIXED,
        help="Final output after thinking fix",
    )
    parser.add_argument(
        "--strict_images",
        action="store_true",
        default=DEFAULT_STRICT_IMAGES,
        help="Require identical image sets per record",
    )
    parser.add_argument(
        "--no_strict_images",
        action="store_false",
        dest="strict_images",
        help="Disable strict image match",
    )
    parser.add_argument(
        "--order_insensitive",
        action="store_true",
        default=DEFAULT_ORDER_INSENSITIVE,
        help="Ignore image order when aligning records",
    )
    parser.add_argument(
        "--order_sensitive",
        action="store_false",
        dest="order_insensitive",
        help="Require same image order when aligning",
    )
    parser.add_argument(
        "--remove_crops",
        action="store_true",
        default=DEFAULT_REMOVE_CROPS,
        help="Drop image paths containing 'crop'",
    )
    parser.add_argument(
        "--keep_crops",
        action="store_false",
        dest="remove_crops",
        help="Keep crop paths",
    )
    return parser.parse_args()

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)


# ----------------- Fuse logic -----------------
def filter_crops(imgs: List[Any]) -> List[str]:
    """Remove paths containing "crop" (case-insensitive)."""
    return [str(p) for p in imgs if "crop" not in str(p).lower()]


def load_jsonl(path: Path, remove_crops: bool) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    removed = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if remove_crops and isinstance(rec.get("images"), list):
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
        # Tie: fall back to highest-weight model's answer
        max_idx = max(range(len(weights)), key=lambda i: weights[i])
        return answers[max_idx].upper()
    return "A" if score["A"] > score["B"] else "B"


def images_key(rec: Dict[str, Any], order_insensitive: bool) -> Tuple[str, ...]:
    imgs = rec.get("images")
    if not isinstance(imgs, list) or not imgs:
        raise ValueError("Record missing non-empty 'images' list for alignment")
    normalized = [str(p) for p in imgs]
    if order_insensitive:
        return tuple(sorted(normalized))
    return tuple(normalized)


def index_by_images(records: List[Dict[str, Any]], label: str, order_insensitive: bool) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    mapping: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for idx, rec in enumerate(records):
        key = images_key(rec, order_insensitive)
        if key in mapping:
            raise ValueError(f"Duplicate images key in {label} at record {idx}")
        mapping[key] = rec
    return mapping


def fuse_records(
    all_records: List[List[Dict[str, Any]]],
    strict_images: bool,
    order_insensitive: bool,
    weights: Sequence[float],
) -> List[Dict[str, Any]]:
    if not all_records:
        return []

    maps = [index_by_images(rec, f"ckpt{i}", order_insensitive) for i, rec in enumerate(all_records)]
    key_set = set(maps[0].keys())
    for i, m in enumerate(maps[1:], 1):
        if key_set != set(m.keys()):
            raise ValueError(f"Image sets do not match across checkpoints (ckpt0 vs ckpt{i})")

    fused: List[Dict[str, Any]] = []
    for key in maps[0].keys():  # keep ckpt0 order
        recs = [m[key] for m in maps]
        if strict_images:
            norm_keys = [images_key(r, order_insensitive) for r in recs]
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


def apply_fused_to_ckpt1(
    ckpt1: List[Dict[str, Any]],
    fused: List[Dict[str, Any]],
    order_insensitive: bool,
) -> List[Dict[str, Any]]:
    fused_map = {tuple(rec["images"]): rec["fused_answer"] for rec in fused}
    updated: List[Dict[str, Any]] = []
    for idx, rec in enumerate(ckpt1):
        key = images_key(rec, order_insensitive)
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


# ------------- Thinking fix logic -------------
THINKING_RE = re.compile(r"(<thinking>)(.*?)(</thinking>)", re.DOTALL | re.IGNORECASE)
POSITIVE_WORDS = re.compile(
    r"\b(?:superior|richer|sharper|better|cleaner|more\s+defined|"
    r"more\s+detailed|more\s+vibrant|crisper|more\s+pronounced|more\s+refined|"
    r"polished|well[-\s]defined|more\s+balanced|more\s+dynamic|"
    r"more\s+crisp|higher\s+sharpness|more\s+finely|more\s+distinct|"
    r"more\s+accurate|more\s+harmonious|better\s+color|higher\s+quality|"
    r"more\s+visually|technically\s+superior|more\s+controlled|"
    r"higher\s+resolution|higher\s+degree|higher\s+overall|"
    r"higher|clearer|more\s+natural|more\s+realistic|more\s+authentic|"
    r"more\s+uniform)\b",
    re.IGNORECASE,
)
NEGATIVE_WORDS = re.compile(
    r"\b(?:softer|blurrier|noisier|less\s+defined|less\s+detailed|"
    r"less\s+visible|less\s+sharp|less\s+natural|less\s+noticeable|"
    r"less\s+refined|less\s+accurate|over[-\s]smooth(?:ed|ing)?|"
    r"over[-\s]process(?:ed|ing)?|loss\s+of\s+detail|artifacts|"
    r"appears\s+softer|appears\s+slightly\s+softer|less\s+definition|"
    r"less\s+fine|less\s+authentic|less\s+vibrant|less\s+dynamic|"
    r"less\s+favorable|less\s+detailed\s+rendering|"
    r"slightly\s+softer|slightly\s+blurr|slightly\s+less|"
    r"reduced\s+detail|reduced\s+clarity|lower\s+\w+|"
    r"excessive\s+smooth|significant\s+smearing|abundant\s+\w*\s*noise|"
    r"visible\s+noise|noticeable\s+noise|unnatural|artificial)\b",
    re.IGNORECASE,
)
IMG_A_RE = re.compile(r"Image\s+A(?:\s*[\(/][^)]*[\)/])?", re.IGNORECASE)
IMG_B_RE = re.compile(r"Image\s+B(?:\s*[\(/][^)]*[\)/])?", re.IGNORECASE)
COMPARED_TO_RE = re.compile(r"compared\s+to\s+(Image\s+[AB](?:\s*\([^)]*\))?)", re.IGNORECASE)
WHICH_LOSER_RE = re.compile(
    r"(Image\s+[AB](?:\s*\([^)]*\))?)\s*,\s*which\s+(?:shows|appears|"
    r"exhibits|demonstrates|has|displays|presents)", re.IGNORECASE,
)
CONTRAST_LOSER_RE = re.compile(
    r"(?:whereas|while|in\s+contrast[,\s]+)(Image\s+[AB](?:\s*(?:\([^)]*\)|/\w+))?)"
    r"\s+(?:shows?|appears?|has|exhibits?|displays?|presents?|suffers?|is\b|may\b)",
    re.IGNORECASE,
)
NEG_EXHIBITS_RE = re.compile(
    r"(Image\s+[AB](?:\s*(?:\([^)]*\)|/\w+))?)\s+(?:exhibits|shows|displays|has)\s+"
    r"(?:noticeable\s+|significant\s+|excessive\s+|abundant\s+(?:and\s+\w+\s+)?|visible\s+|clear\s+)?"
    r"(?:smearing|artifacts?|loss\s+of|halos?|over[-\s]process|"
    r"over[-\s]smooth|artificial|distortion|degraded?|reduced\s+sharpness|"
    r"lower\s+overall\s+sharpness|lower\s+sharpness|"
    r"noise|smoothing|blurring|blurriness)",
    re.IGNORECASE,
)


def _letter(token: str) -> str:
    """Extract 'A' or 'B' from an 'Image A...' token."""
    m = re.search(r"\bImage\s+([AB])\b", token, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([AB])\b", token, re.IGNORECASE)
    return m2.group(1).upper() if m2 else ""


def infer_thinking_winner(thinking: str) -> str | None:
    """
    Return 'A' or 'B' — whichever letter the thinking block credits
    with more positive-quality statements. Returns None if tied/unclear.
    """
    score = {"A": 0.0, "B": 0.0}

    sentences = re.split(r"(?<=[.!?;])\s+|\n|;\s*", thinking)

    pos_subj_pattern = re.compile(
        r"(Image\s+[AB](?:\s*(?:\([^)]*\)|/\w+))?)\s+"
        r"(?:exhibits|demonstrates|has\s+superior|presents|retains|"
        r"shows\s+richer|shows\s+more|shows\s+superior|shows\s+higher|"
        r"shows\s+a\s+more\s+natural|preserves|delivers|"
        r"displays\s+richer|displays\s+more|displays\s+superior|"
        r"is\s+perceived\s+as\s+more|avoids\s+such\s+heavy|"
        r"has\s+a\s+more\s+balanced|has\s+more\s+natural|has\s+higher)",
        re.IGNORECASE,
    )
    neg_subj_pattern = re.compile(
        r"(Image\s+[AB](?:\s*(?:\([^)]*\)|/\w+))?)\s+"
        r"(?:appears\s+softer|appears\s+slightly|appears\s+less|appears\s+artificial|"
        r"appears\s+unnatural|shows\s+signs|shows\s+less|shows\s+no\s+noticeable|"
        r"exhibits\s+less|has\s+lower|has\s+less|has\s+average|"
        r"is\s+softer|is\s+less|displays\s+less|displays\s+significant\s+smearing|"
        r"suffers\s+from|may\s+exhibit\s+less\s+favorable|"
        r"presents\s+a\s+softer|presents\s+a\s+less|presents\s+less|"
        r"exhibits\s+abundant)",
        re.IGNORECASE,
    )

    def apply_sentence(sent: str) -> None:
        has_a = bool(IMG_A_RE.search(sent))
        has_b = bool(IMG_B_RE.search(sent))

        # Strategy 1: single-image sentences
        if has_a and not has_b:
            pos = len(POSITIVE_WORDS.findall(sent))
            neg = len(NEGATIVE_WORDS.findall(sent))
            neg += len(NEG_EXHIBITS_RE.findall(sent))
            score["A"] += pos - neg
        elif has_b and not has_a:
            pos = len(POSITIVE_WORDS.findall(sent))
            neg = len(NEGATIVE_WORDS.findall(sent))
            neg += len(NEG_EXHIBITS_RE.findall(sent))
            score["B"] += pos - neg

        # Strategy 2: dual-image sentences
        elif has_a and has_b:
            for m in COMPARED_TO_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 2
                    score[loser] -= 2

            for m in WHICH_LOSER_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 1
                    score[loser] -= 1

            for m in CONTRAST_LOSER_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 1.5
                    score[loser] -= 1.5

            neg_exhibit_letters = set()
            for m in NEG_EXHIBITS_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    neg_exhibit_letters.add(loser)
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 1.5
                    score[loser] -= 1.5

            neg_subj_letters = set()
            for m in neg_subj_pattern.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    neg_subj_letters.add(loser)
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 0.5
                    score[loser] -= 1.5

            for m in pos_subj_pattern.finditer(sent):
                winner = _letter(m.group(1))
                if winner and winner not in neg_exhibit_letters and winner not in neg_subj_letters:
                    loser = "B" if winner == "A" else "A"
                    score[winner] += 1.5
                    score[loser] -= 0.5

    for sent in sentences:
        apply_sentence(sent)

    if score["A"] == score["B"]:
        for m in pos_subj_pattern.finditer(thinking):
            winner = _letter(m.group(1))
            if winner:
                score[winner] += 1

        for m in neg_subj_pattern.finditer(thinking):
            loser = _letter(m.group(1))
            if loser:
                score[loser] -= 1

        for m in COMPARED_TO_RE.finditer(thinking):
            loser = _letter(m.group(1))
            if loser:
                winner = "B" if loser == "A" else "A"
                score[winner] += 1

        for m in NEG_EXHIBITS_RE.finditer(thinking):
            loser = _letter(m.group(1))
            if loser:
                score[loser] -= 1

        for m in CONTRAST_LOSER_RE.finditer(thinking):
            loser = _letter(m.group(1))
            if loser:
                winner = "B" if loser == "A" else "A"
                score[winner] += 1

    if score["A"] > score["B"]:
        return "A"
    if score["B"] > score["A"]:
        return "B"
    return None


def swap_ab_in_thinking(thinking: str) -> str:
    """Swap A↔B (and Left/Right) references inside a thinking string."""
    result = thinking
    result = re.sub(r"Image\s+A\s*(?:\(\s*(?:Left|Right)\s*\)|/\s*(?:left|right))?",
                    "__IMA__", result, flags=re.IGNORECASE)
    result = re.sub(r"Image\s+B\s*(?:\(\s*(?:Left|Right)\s*\)|/\s*(?:left|right))?",
                    "__IMB__", result, flags=re.IGNORECASE)
    result = result.replace("__IMA__", "Image B (Right)")
    result = result.replace("__IMB__", "Image A (Left)")
    result = re.sub(r"Image\s+[AB]\s*(?:\([^)]*\))?", lambda m: m.group(0).replace(" ", "\x00"), result)

    def swap_lone(m: re.Match) -> str:
        pre, letter, post = m.group(1), m.group(2), m.group(3)
        return pre + ("B" if letter.upper() == "A" else "A") + post

    result = re.sub(r"(\s)([AB])(\s)", swap_lone, result, flags=re.IGNORECASE)
    result = result.replace("\x00", " ")
    return result


def fix_solution(solution: str) -> tuple[str, str]:
    """Return (fixed_solution, status)."""
    answer_match = ANSWER_RE.search(solution)
    if not answer_match:
        return solution, "no_answer"

    correct = answer_match.group(1).upper()
    thinking_match = THINKING_RE.search(solution)
    if not thinking_match:
        return solution, "no_thinking"

    thinking_content = thinking_match.group(2)
    winner = infer_thinking_winner(thinking_content)

    if winner is None:
        return solution, "unclear"
    if winner == correct:
        return solution, "ok"

    new_thinking = swap_ab_in_thinking(thinking_content)
    new_solution = (
        solution[: thinking_match.start()]
        + thinking_match.group(1)
        + new_thinking
        + thinking_match.group(3)
        + solution[thinking_match.end() :]
    )
    return new_solution, "swapped"


def fix_thinking_records(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {"swapped": 0, "ok": 0, "unclear": 0, "no_answer": 0, "no_thinking": 0}
    fixed: List[Dict[str, Any]] = []
    for rec in records:
        sol = str(rec.get("solution", ""))
        new_sol, status = fix_solution(sol)
        stats[status] += 1
        new_rec = dict(rec)
        new_rec["solution"] = new_sol
        fixed.append(new_rec)
    return fixed, stats


# ----------------- Main flow -----------------
def main() -> None:
    args = parse_args()

    if args.preset:
        preset_cfg = PRESETS[args.preset]
        args.ckpt_paths = preset_cfg["ckpt_paths"]
        args.ckpt_weights = ",".join(str(w) for w in preset_cfg["ckpt_weights"]) if preset_cfg.get("ckpt_weights") else None
        args.out_fused = preset_cfg["out_fused"]
        args.out_fixed = preset_cfg["out_fixed"]

    ckpt_paths = args.ckpt_paths or DEFAULT_CKPT_PATHS
    if len(ckpt_paths) < 1:
        raise ValueError("--ckpt_paths must contain at least one path")

    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    loaded = [load_jsonl(p, args.remove_crops) for p in ckpt_paths]
    records = [r for r, _ in loaded]
    removed_total = sum(rem for _, rem in loaded)

    parsed_weights = parse_float_list(args.ckpt_weights)
    if parsed_weights is None:
        weights = [1.0 / len(records)] * len(records)
    elif len(parsed_weights) != len(records):
        raise ValueError("ckpt_weights length must match ckpt_paths")
    else:
        weights = list(parsed_weights)

    fused = fuse_records(
        records,
        strict_images=args.strict_images,
        order_insensitive=args.order_insensitive,
        weights=weights,
    )
    updated_ckpt1 = apply_fused_to_ckpt1(records[0], fused, args.order_insensitive)
    save_jsonl(updated_ckpt1, args.out_fused)

    fixed_records, stats = fix_thinking_records(updated_ckpt1)
    save_jsonl(fixed_records, args.out_fixed)

    msg = (
        f"Fused {len(fused)} records and wrote intermediate -> {args.out_fused}\n"
        f"Fixed thinking and wrote -> {args.out_fixed}"
    )
    if args.remove_crops:
        msg += f" (removed {removed_total} cropped paths)"
    print(msg)
    print(
        "Stats: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
    )


if __name__ == "__main__":
    main()
