"""PyTorch datasets for SASRec / behavior-aware SASRec."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .sequences import (
    BEHAVIOR_BUY,
    BEHAVIOR_PAD,
    BehaviorEvent,
    filter_purchase_only,
)


def _pad_or_truncate(
    item_ids: List[int],
    behavior_ids: List[int],
    max_len: int,
) -> tuple[List[int], List[int]]:
    if len(item_ids) > max_len:
        item_ids = item_ids[-max_len:]
        behavior_ids = behavior_ids[-max_len:]
    pad = max_len - len(item_ids)
    if pad > 0:
        item_ids = [0] * pad + item_ids
        behavior_ids = [BEHAVIOR_PAD] * pad + behavior_ids
    return item_ids, behavior_ids


class SASRecDataset(Dataset):
    """Next-purchase prediction examples from per-user histories.

    Each example is (history_items, history_behaviors, target_item).
    When ``purchase_only=True``, cart events are stripped from history so the
    same target purchases can be used for a purchase-only baseline.
    """

    def __init__(
        self,
        split_data: Dict[int, object],
        max_len: int = 50,
        purchase_only: bool = False,
        n_items: Optional[int] = None,
        n_negatives: int = 1,
        seed: int = 42,
    ) -> None:
        self.max_len = max_len
        self.purchase_only = purchase_only
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.rng = np.random.default_rng(seed)
        self.examples: List[dict] = []

        for user_id, payload in split_data.items():
            # Train split stores a list of instances; val/test store one dict.
            instances = payload if isinstance(payload, list) else [payload]
            for inst in instances:
                history: Sequence[BehaviorEvent] = inst["history"]
                if purchase_only:
                    history = filter_purchase_only(history)
                if not history:
                    continue
                self.examples.append(
                    {
                        "user_id": int(user_id),
                        "history": list(history),
                        "target_item": int(inst["target_item"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def _sample_negatives(self, positive: int, banned: set[int]) -> List[int]:
        assert self.n_items is not None
        negs: List[int] = []
        while len(negs) < self.n_negatives:
            neg = int(self.rng.integers(1, self.n_items + 1))
            if neg != positive and neg not in banned:
                negs.append(neg)
        return negs

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        items = [e.item_id for e in ex["history"]]
        behaviors = [e.behavior_id for e in ex["history"]]
        items, behaviors = _pad_or_truncate(items, behaviors, self.max_len)

        out = {
            "user_id": ex["user_id"],
            "item_ids": torch.tensor(items, dtype=torch.long),
            "behavior_ids": torch.tensor(behaviors, dtype=torch.long),
            "target_item": torch.tensor(ex["target_item"], dtype=torch.long),
        }
        if self.n_negatives > 0 and self.n_items is not None:
            banned = {e.item_id for e in ex["history"] if e.behavior_id == BEHAVIOR_BUY}
            banned.add(ex["target_item"])
            negs = self._sample_negatives(ex["target_item"], banned)
            out["neg_items"] = torch.tensor(negs, dtype=torch.long)
        return out


def collate_sasrec(batch: List[dict]) -> dict:
    collated = {
        "user_id": torch.tensor([b["user_id"] for b in batch], dtype=torch.long),
        "item_ids": torch.stack([b["item_ids"] for b in batch], dim=0),
        "behavior_ids": torch.stack([b["behavior_ids"] for b in batch], dim=0),
        "target_item": torch.stack([b["target_item"] for b in batch], dim=0),
    }
    if "neg_items" in batch[0]:
        collated["neg_items"] = torch.stack([b["neg_items"] for b in batch], dim=0)
    return collated
