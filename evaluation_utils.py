
"""
Shared recommendation evaluation utilities.

Used by the popularity baseline, purchase-only SASRec,
and behavior-aware SASRec.

Metrics:
- HitRate@K
- NDCG@K
- Mean Reciprocal Rank (MRR)

All candidates must use original product IDs.
Prediction scores must be aligned with candidate order.
"""

import math

import numpy as np
import pandas as pd


def hit_rate_at_k(ranked_candidates, target_item, k=10):
    """Return 1 if the target ranks within the top K."""
    return int(target_item in ranked_candidates[:k])


def ndcg_at_k(ranked_candidates, target_item, k=10):
    """Calculate NDCG@K for one relevant target."""
    top_k = ranked_candidates[:k]

    if target_item not in top_k:
        return 0.0

    rank = top_k.index(target_item) + 1
    return 1.0 / math.log2(rank + 1)


def reciprocal_rank(ranked_candidates, target_item):
    """Calculate reciprocal rank for one target."""
    if target_item not in ranked_candidates:
        return 0.0

    rank = ranked_candidates.index(target_item) + 1
    return 1.0 / rank


def evaluate_user(ranked_candidates, target_item, k=10):
    """Calculate ranking metrics for one user."""
    return {
        f"HitRate@{k}": hit_rate_at_k(
            ranked_candidates, target_item, k
        ),
        f"NDCG@{k}": ndcg_at_k(
            ranked_candidates, target_item, k
        ),
        "RR": reciprocal_rank(
            ranked_candidates, target_item
        )
    }


def evaluate_model_predictions(candidates, scores, target_item, k=10):
    """
    Evaluate candidate scores from any recommendation model.

    Candidates are original product IDs.
    Scores must correspond to candidates in the same order.
    Higher scores indicate stronger recommendations.
    """
    if len(candidates) != len(scores):
        raise ValueError("Candidates and scores must have the same length.")

    if len(candidates) != len(set(candidates)):
        raise ValueError("Candidate products must be unique.")

    if target_item not in candidates:
        raise ValueError("Target item is missing from candidates.")

    scores = np.asarray(scores, dtype=float)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Prediction scores must be finite.")

    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda i: (-scores[i], candidates[i])
    )

    ranked_candidates = [
        candidates[i] for i in ranked_indices
    ]

    return evaluate_user(ranked_candidates, target_item, k)


def evaluate_model_dataset(candidate_sets, model_scores, targets, k=10):
    """
    Evaluate model predictions across an evaluation population.

    candidate_sets: user ID -> candidate original product IDs
    model_scores: user ID -> scores aligned with candidates
    targets: user ID -> actual next-purchase product ID

    Returns aggregate metrics and per-user results.
    """
    if set(candidate_sets) != set(model_scores):
        raise ValueError(
            "Candidate sets and model scores must contain the same users."
        )

    results = []

    for user, candidates in candidate_sets.items():
        if user not in targets:
            raise ValueError(f"Missing target for user {user}")

        metrics = evaluate_model_predictions(
            candidates=candidates,
            scores=model_scores[user],
            target_item=targets[user],
            k=k
        )

        results.append({
            "user_id": user,
            "target_item": targets[user],
            **metrics
        })

    user_results = pd.DataFrame(results)

    if user_results.empty:
        raise ValueError("No users were provided for evaluation.")

    summary = {
        "n_users": len(user_results),
        f"HitRate@{k}": user_results[f"HitRate@{k}"].mean(),
        f"NDCG@{k}": user_results[f"NDCG@{k}"].mean(),
        "MRR": user_results["RR"].mean()
    }

    return summary, user_results
