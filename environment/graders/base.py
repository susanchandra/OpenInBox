"""
Shared utilities used by all three graders.

Keeping these in one place so the grader formulas stay identical
and any change only needs to happen once.
"""

import re


def exact_match(a: str, b: str) -> float:
    """1.0 if strings are identical, 0.0 otherwise."""
    return 1.0 if a == b else 0.0


def token_f1(predicted: dict[str, str], ground: dict[str, str]) -> float:
    """
    Token-level F1 between predicted and ground-truth field dicts.

    For each field in ground truth, tokenizes both values and computes
    precision/recall/F1. Returns the mean F1 across all ground-truth fields.
    Fields the agent omitted count as pure recall misses (F1 = 0).
    Returns 0.0 if ground truth has no fields.
    """
    if not ground:
        # No fields to extract — agent correctly extracted nothing → perfect score.
        # If agent hallucinated fields when none were expected, still 0.0.
        return 1.0 if not predicted else 0.0

    scores = []
    for field, gt_value in ground.items():
        gt_tokens = _tokenize(gt_value)
        pred_tokens = _tokenize(predicted.get(field, ""))

        if not gt_tokens and not pred_tokens:
            scores.append(1.0)
            continue
        if not gt_tokens or not pred_tokens:
            scores.append(0.0)
            continue

        common = set(pred_tokens) & set(gt_tokens)
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)

        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters."""
    return re.findall(r"\w+", text.lower())
