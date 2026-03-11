#!/usr/bin/env python3
"""
Fix thinking A/B consistency in a JSONL file.

The <answer> tag is assumed correct. Inside the <thinking> block, A and B
references may be swapped. This script detects whether the thinking block
concludes the *wrong* letter as better, and if so, swaps all A/B (and
Left/Right) references within that thinking block only.

Detection heuristic
-------------------
Two complementary scoring strategies are combined:

1. **Single-image sentences**: sentences that mention only A or only B get
   scored by the positive-quality keywords they contain.

2. **Dual-image sentences** (both A and B present): the common structure is
   "Image X <positive-verb> … compared to Image Y, which <negative-verb>".
   We find which image is the grammatical subject of a positive verb
   (exhibits, demonstrates, has superior, …) and award it a point; the
   image described after "compared to … which" is the loser and loses a point.
   We also award negative points when an image is the subject of a negative
   predicate (appears softer, shows less detail, …).

The letter with the higher net score is what the thinking claims is better.
If that disagrees with <answer>, we swap A↔B (and Left↔Right) in the block.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
INPUT_PATH  = Path(r"C:\Users\leozx\Downloads\fused8039.jsonl")
OUTPUT_PATH = Path(r"C:\Users\leozx\Downloads\8039_fixed.jsonl")
# ───────────────────────────────────────────────────────────────────────────

THINKING_RE = re.compile(r"(<thinking>)(.*?)(</thinking>)", re.DOTALL | re.IGNORECASE)
ANSWER_RE   = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)

# Positive-quality keywords
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

# Negative-quality keywords
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

# Positive subject verbs: "Image X <verb>" where X is the good one
POS_SUBJECT_VERBS = re.compile(
    r"\b(?:exhibits|demonstrates|has\s+superior|presents|retains|shows\s+richer|"
    r"shows\s+more|shows\s+superior|preserves|delivers)\b",
    re.IGNORECASE,
)

# Pattern to find "Image A" / "Image B" (with optional parenthetical or slash notation)
IMG_A_RE = re.compile(r"Image\s+A(?:\s*[\(/][^)]*[\)/])?", re.IGNORECASE)
IMG_B_RE = re.compile(r"Image\s+B(?:\s*[\(/][^)]*[\)/])?", re.IGNORECASE)

# "compared to Image X" — the X is the loser
COMPARED_TO_RE = re.compile(
    r"compared\s+to\s+(Image\s+[AB](?:\s*\([^)]*\))?)", re.IGNORECASE
)

# "Image X, which shows/appears …" — X is the loser side
WHICH_LOSER_RE = re.compile(
    r"(Image\s+[AB](?:\s*\([^)]*\))?)\s*,\s*which\s+(?:shows|appears|"
    r"exhibits|demonstrates|has|displays|presents)",
    re.IGNORECASE,
)

# "whereas/while/in contrast … Image X" introduces the loser after a contrast
CONTRAST_LOSER_RE = re.compile(
    r"(?:whereas|while|in\s+contrast[,\s]+)(Image\s+[AB](?:\s*(?:\([^)]*\)|/\w+))?)"
    r"\s+(?:shows?|appears?|has|exhibits?|displays?|presents?|suffers?|is\b|may\b)",
    re.IGNORECASE,
)

# Negative exhibits: "Image X exhibits smearing/artifacts/loss/..."
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
    # Must look for the letter AFTER the word "Image" to avoid matching
    # the 'a' in "Image" itself when using case-insensitive search.
    m = re.search(r"\bImage\s+([AB])\b", token, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: bare standalone letter token
    m2 = re.search(r"\b([AB])\b", token, re.IGNORECASE)
    return m2.group(1).upper() if m2 else ""


def infer_thinking_winner(thinking: str) -> str | None:
    """
    Return 'A' or 'B' — whichever letter the thinking block credits
    with more positive-quality statements. Returns None if truly tied.
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

        # ── Strategy 1: single-image sentences ──────────────────────────
        if has_a and not has_b:
            pos = len(POSITIVE_WORDS.findall(sent))
            neg = len(NEGATIVE_WORDS.findall(sent))
            # negative exhibits override positive detection
            neg += len(NEG_EXHIBITS_RE.findall(sent))
            score["A"] += pos - neg
        elif has_b and not has_a:
            pos = len(POSITIVE_WORDS.findall(sent))
            neg = len(NEGATIVE_WORDS.findall(sent))
            neg += len(NEG_EXHIBITS_RE.findall(sent))
            score["B"] += pos - neg

        # ── Strategy 2: dual-image sentences ────────────────────────────
        elif has_a and has_b:
            # "compared to Image X" → X is the loser
            for m in COMPARED_TO_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 2
                    score[loser]  -= 2

            # "Image X, which shows/appears …" → X is the loser side
            for m in WHICH_LOSER_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 1
                    score[loser]  -= 1

            # "whereas/in contrast, Image X shows/appears" → X is the loser
            for m in CONTRAST_LOSER_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 1.5
                    score[loser]  -= 1.5

            # "Image X exhibits smearing/artifacts/noise" → X is bad
            # Must be checked BEFORE pos_subj to prevent false positives
            neg_exhibit_letters = set()
            for m in NEG_EXHIBITS_RE.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    neg_exhibit_letters.add(loser)
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 1.5
                    score[loser]  -= 1.5

            # Subject of a negative verb is the loser
            neg_subj_letters = set()
            for m in neg_subj_pattern.finditer(sent):
                loser = _letter(m.group(1))
                if loser:
                    neg_subj_letters.add(loser)
                    winner = "B" if loser == "A" else "A"
                    score[winner] += 0.5
                    score[loser]  -= 1.5

            # Subject of a positive verb is the winner
            # BUT skip if that same image was already identified as a
            # negative exhibit or negative subject (avoids double-counting
            # "exhibits noticeable noise" as both positive and negative)
            for m in pos_subj_pattern.finditer(sent):
                winner = _letter(m.group(1))
                if winner and winner not in neg_exhibit_letters and winner not in neg_subj_letters:
                    loser = "B" if winner == "A" else "A"
                    score[winner] += 1.5
                    score[loser]  -= 0.5

    for sent in sentences:
        apply_sentence(sent)

    # ── Strategy 3: global fallback — count total hits across entire block ──
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
    return None  # truly tied


def swap_ab_in_thinking(thinking: str) -> str:
    """
    Swap every A↔B reference inside a thinking string.
    Convention: A is ALWAYS Left, B is ALWAYS Right.
    So after swapping letters, the side labels must also be corrected
    to match the new letter (A→Left, B→Right).
    """
    result = thinking

    # Step 1: replace all "Image A ..." variants with __IMA__ placeholder
    # and all "Image B ..." variants with __IMB__ placeholder.
    # Handle explicit side labels first (longer match), then bare Image A/B.
    result = re.sub(r"Image\s+A\s*(?:\(\s*(?:Left|Right)\s*\)|/\s*(?:left|right))?",
                    "__IMA__", result, flags=re.IGNORECASE)
    result = re.sub(r"Image\s+B\s*(?:\(\s*(?:Left|Right)\s*\)|/\s*(?:left|right))?",
                    "__IMB__", result, flags=re.IGNORECASE)

    # Step 2: swap placeholders — A becomes B (Right), B becomes A (Left)
    result = result.replace("__IMA__", "Image B (Right)")
    result = result.replace("__IMB__", "Image A (Left)")

    # Step 3: fix any remaining standalone "(Left)" / "(Right)" that are
    # not already attached to an Image label (rare, but keep consistent)
    # These are left as-is since their meaning without context is ambiguous.

    # Step 4: swap plain standalone letters " A " / " B " (e.g. "In A," "whereas A ")
    # that were NOT part of "Image A/B" phrases already handled.
    # Must re-protect "Image A/B (Left/Right)" so the lone-letter pass can't corrupt them.
    result = re.sub(r"Image\s+[AB]\s*(?:\([^)]*\))?", lambda m: m.group(0).replace(" ", "\x00"), result)

    def swap_lone(m: re.Match) -> str:
        pre, letter, post = m.group(1), m.group(2), m.group(3)
        return pre + ("B" if letter.upper() == "A" else "A") + post

    result = re.sub(r"(\s)([AB])(\s)", swap_lone, result, flags=re.IGNORECASE)

    # Restore protected spaces
    result = result.replace("\x00", " ")

    return result


def fix_solution(solution: str) -> tuple[str, str]:
    """
    Returns (fixed_solution, status) where status is one of:
      'swapped'    - thinking was wrong, swapped
      'ok'         - thinking already consistent
      'unclear'    - could not determine thinking winner; left unchanged
      'no_answer'  - no <answer> tag found
      'no_thinking'- no <thinking> tag found
    """
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

    # Need to swap
    new_thinking = swap_ab_in_thinking(thinking_content)
    new_solution = (
        solution[: thinking_match.start()]
        + thinking_match.group(1)
        + new_thinking
        + thinking_match.group(3)
        + solution[thinking_match.end() :]
    )
    return new_solution, "swapped"


def main() -> None:
    stats = {"swapped": 0, "ok": 0, "unclear": 0, "no_answer": 0, "no_thinking": 0}
    out_lines: list[str] = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sol = str(rec.get("solution", ""))
            new_sol, status = fix_solution(sol)
            stats[status] += 1
            rec["solution"] = new_sol
            out_lines.append(json.dumps(rec, ensure_ascii=False))

    OUTPUT_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"Done. Output -> {OUTPUT_PATH}")
    print(f"  swapped   : {stats['swapped']}")
    print(f"  ok        : {stats['ok']}")
    print(f"  unclear   : {stats['unclear']}")
    print(f"  no_answer : {stats['no_answer']}")
    print(f"  no_thinking: {stats['no_thinking']}")


if __name__ == "__main__":
    main()
