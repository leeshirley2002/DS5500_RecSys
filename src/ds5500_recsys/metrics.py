"""Ranking metrics for next-purchase evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch


def hit_rate_at_k(ranks: np.ndarray, k: int) -> float:
    """ranks: 1-based rank of the ground-truth item for each example."""
    return float((ranks <= k).mean()) if len(ranks) else 0.0


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """NDCG@k with binary relevance; misses contribute 0 (mean over all examples)."""
    if len(ranks) == 0:
        return 0.0
    gains = np.where(ranks <= k, 1.0 / np.log2(ranks.astype(np.float64) + 1.0), 0.0)
    return float(gains.mean())


def mrr(ranks: np.ndarray) -> float:
    return float((1.0 / ranks).mean()) if len(ranks) else 0.0


@torch.no_grad()
def rank_targets(
    scores: torch.Tensor,
    target_items: torch.Tensor,
) -> np.ndarray:
    """Compute 1-based ranks of targets given score matrix [B, n_items+1]."""
    # Higher score = better. Rank among all non-padding items (index >= 1).
    target_scores = scores.gather(1, target_items.view(-1, 1)).squeeze(1)
    # Count how many items (excluding pad 0) score strictly higher.
    better = scores[:, 1:] > target_scores.unsqueeze(1)
    ranks = better.sum(dim=1) + 1
    return ranks.cpu().numpy().astype(np.int64)


@torch.no_grad()
def rank_targets_among_candidates(
    cand_scores: torch.Tensor,
    pos_index: torch.Tensor,
) -> np.ndarray:
    """cand_scores: [B, C], pos_index: [B] index of positive inside candidates."""
    pos_scores = cand_scores.gather(1, pos_index.view(-1, 1)).squeeze(1)
    better = cand_scores > pos_scores.unsqueeze(1)
    ranks = better.sum(dim=1) + 1
    return ranks.cpu().numpy().astype(np.int64)


def summarize_ranks(ranks: np.ndarray, ks: Iterable[int] = (5, 10)) -> Dict[str, float]:
    out: Dict[str, float] = {"MRR": mrr(ranks), "n": float(len(ranks))}
    for k in ks:
        out[f"HR@{k}"] = hit_rate_at_k(ranks, k)
        out[f"NDCG@{k}"] = ndcg_at_k(ranks, k)
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    parts: List[str] = []
    for key in sorted(metrics.keys()):
        if key == "n":
            parts.append(f"n={int(metrics[key])}")
        else:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)
