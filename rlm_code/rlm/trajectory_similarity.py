"""Root-trajectory similarity metrics for harness generalization checks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]")


@dataclass(frozen=True, slots=True)
class TrajectorySimilarity:
    """Similarity measures used to compare two root-model trajectories."""

    normalized_levenshtein: float
    trigram_containment: float
    trigram_jaccard: float
    weighted_trigram_jaccard: float
    length_ratio: float

    @property
    def mean(self) -> float:
        return sum(asdict(self).values()) / 5.0

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["mean"] = self.mean
        return result


def trajectory_tokens(trajectory: str | Iterable[str]) -> tuple[str, ...]:
    """Tokenize text or a sequence of structural actions deterministically."""
    text = trajectory if isinstance(trajectory, str) else "\n<STEP>\n".join(trajectory)
    return tuple(_TOKEN_PATTERN.findall(str(text)))


def _levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_token in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_token in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _ngrams(tokens: Sequence[str], size: int = 3) -> list[tuple[str, ...]]:
    if not tokens:
        return []
    if len(tokens) < size:
        return [tuple(tokens)]
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def compare_trajectory_similarity(
    left: str | Iterable[str],
    right: str | Iterable[str],
) -> TrajectorySimilarity:
    """Compare two root trajectories using the July harness-post metrics."""
    left_tokens = trajectory_tokens(left)
    right_tokens = trajectory_tokens(right)
    longest = max(len(left_tokens), len(right_tokens))
    levenshtein = (
        1.0 if longest == 0 else 1.0 - (_levenshtein_distance(left_tokens, right_tokens) / longest)
    )

    left_ngrams = _ngrams(left_tokens)
    right_ngrams = _ngrams(right_tokens)
    left_set = set(left_ngrams)
    right_set = set(right_ngrams)
    intersection = len(left_set & right_set)
    smaller = min(len(left_set), len(right_set))
    union = len(left_set | right_set)
    containment = 1.0 if smaller == 0 and union == 0 else intersection / max(1, smaller)
    jaccard = 1.0 if union == 0 else intersection / union

    left_counts = Counter(left_ngrams)
    right_counts = Counter(right_ngrams)
    all_ngrams = set(left_counts) | set(right_counts)
    weighted_union = sum(max(left_counts[item], right_counts[item]) for item in all_ngrams)
    weighted_intersection = sum(min(left_counts[item], right_counts[item]) for item in all_ngrams)
    weighted_jaccard = 1.0 if weighted_union == 0 else weighted_intersection / weighted_union
    length_ratio = 1.0 if longest == 0 else min(len(left_tokens), len(right_tokens)) / longest

    return TrajectorySimilarity(
        normalized_levenshtein=round(levenshtein, 6),
        trigram_containment=round(containment, 6),
        trigram_jaccard=round(jaccard, 6),
        weighted_trigram_jaccard=round(weighted_jaccard, 6),
        length_ratio=round(length_ratio, 6),
    )


def nearest_training_trajectories(
    training: Iterable[str | Iterable[str]],
    evaluation: Iterable[str | Iterable[str]],
) -> list[dict[str, object]]:
    """Find the most structurally similar training trajectory for each eval item."""
    train_items = list(training)
    if not train_items:
        raise ValueError("At least one training trajectory is required.")
    rows: list[dict[str, object]] = []
    for eval_index, eval_item in enumerate(evaluation):
        candidates = [compare_trajectory_similarity(item, eval_item) for item in train_items]
        best_index = max(range(len(candidates)), key=lambda index: candidates[index].mean)
        rows.append(
            {
                "evaluation_index": eval_index,
                "training_index": best_index,
                "similarity": candidates[best_index].to_dict(),
            }
        )
    return rows
