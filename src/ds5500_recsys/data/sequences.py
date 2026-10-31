"""Build multi-behavior user sequences and temporal splits for SASRec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Behavior ids used in the model (0 = padding).
BEHAVIOR_PAD = 0
BEHAVIOR_BUY = 1
BEHAVIOR_ADD = 2
BEHAVIOR_REMOVE = 3

BEHAVIOR_NAME_TO_ID = {
    "product_buy": BEHAVIOR_BUY,
    "buy": BEHAVIOR_BUY,
    "purchase": BEHAVIOR_BUY,
    "add_to_cart": BEHAVIOR_ADD,
    "add": BEHAVIOR_ADD,
    "remove_from_cart": BEHAVIOR_REMOVE,
    "remove": BEHAVIOR_REMOVE,
}


@dataclass(frozen=True)
class BehaviorEvent:
    item_id: int
    behavior_id: int
    timestamp: np.datetime64


def load_behavior_events(
    data_dir: Path | str,
    include_behaviors: Sequence[str] = ("product_buy", "add_to_cart", "remove_from_cart"),
    file_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load subsampled parquet event files and return a unified event table.

    Expected columns after load: client_id, sku, timestamp, behavior.
    """
    data_dir = Path(data_dir)
    default_files = {
        "product_buy": "product_buy_subsampled.parquet",
        "add_to_cart": "add_to_cart_subsampled.parquet",
        "remove_from_cart": "remove_from_cart_subsampled.parquet",
    }
    files = file_map or default_files

    frames: List[pd.DataFrame] = []
    for behavior in include_behaviors:
        path = data_dir / files[behavior]
        if not path.exists():
            raise FileNotFoundError(f"Missing event file for {behavior}: {path}")
        df = pd.read_parquet(path)
        df = df.drop_duplicates().copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "client_id", "sku"])
        df["behavior"] = behavior
        frames.append(df[["client_id", "sku", "timestamp", "behavior"]])

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["client_id", "timestamp", "behavior"]).reset_index(drop=True)
    return events


def _remap_ids(series: pd.Series) -> Tuple[pd.Series, Dict[int, int], Dict[int, int]]:
    """Map raw ids to contiguous ints starting at 1 (0 reserved for padding)."""
    uniques = pd.Index(sorted(series.unique()))
    raw_to_idx = {int(raw): i + 1 for i, raw in enumerate(uniques)}
    idx_to_raw = {v: k for k, v in raw_to_idx.items()}
    mapped = series.map(raw_to_idx).astype(np.int64)
    return mapped, raw_to_idx, idx_to_raw


def build_user_sequences(
    events: pd.DataFrame,
    include_behaviors: Sequence[str] = ("product_buy", "add_to_cart", "remove_from_cart"),
    min_buys: int = 2,
    max_len: Optional[int] = None,
    top_n_items: Optional[int] = None,
) -> Dict:
    """Build per-user chronological multi-behavior sequences.

    Returns a dict with sequences, id maps, and metadata. Item/user ids are
    remapped to contiguous integers with 0 reserved for padding.
    """
    df = events[events["behavior"].isin(list(include_behaviors))].copy()
    if df.empty:
        raise ValueError("No events left after filtering behaviors.")

    if top_n_items is not None:
        top_items = (
            df.loc[df["behavior"] == "product_buy", "sku"]
            .value_counts()
            .head(top_n_items)
            .index
        )
        # Keep cart events only for items that also appear in the purchase pool,
        # plus all purchases of those items.
        buy_mask = (df["behavior"] == "product_buy") & df["sku"].isin(top_items)
        cart_mask = (df["behavior"] != "product_buy") & df["sku"].isin(top_items)
        df = df.loc[buy_mask | cart_mask].copy()

    buys_per_user = (
        df.loc[df["behavior"] == "product_buy"].groupby("client_id").size()
    )
    keep_users = set(buys_per_user[buys_per_user >= min_buys].index)
    df = df[df["client_id"].isin(keep_users)].copy()
    if df.empty:
        raise ValueError("No users left after min_buys filter.")

    df["item_id"], item_raw_to_idx, item_idx_to_raw = _remap_ids(df["sku"])
    df["user_id"], user_raw_to_idx, user_idx_to_raw = _remap_ids(df["client_id"])
    df["behavior_id"] = df["behavior"].map(BEHAVIOR_NAME_TO_ID).astype(np.int64)

    sequences: Dict[int, List[BehaviorEvent]] = {}
    for user_id, group in df.groupby("user_id", sort=True):
        group = group.sort_values("timestamp")
        events_list = [
            BehaviorEvent(
                item_id=int(row.item_id),
                behavior_id=int(row.behavior_id),
                timestamp=np.datetime64(row.timestamp.to_datetime64()),
            )
            for row in group.itertuples(index=False)
        ]
        if max_len is not None and len(events_list) > max_len:
            events_list = events_list[-max_len:]
        sequences[int(user_id)] = events_list

    return {
        "sequences": sequences,
        "n_items": len(item_raw_to_idx),
        "n_users": len(user_raw_to_idx),
        "item_raw_to_idx": item_raw_to_idx,
        "item_idx_to_raw": item_idx_to_raw,
        "user_raw_to_idx": user_raw_to_idx,
        "user_idx_to_raw": user_idx_to_raw,
        "include_behaviors": list(include_behaviors),
    }


def _purchase_timestamps(seq: Sequence[BehaviorEvent]) -> List[np.datetime64]:
    return [e.timestamp for e in seq if e.behavior_id == BEHAVIOR_BUY]


def temporal_split_sequences(
    sequences: Dict[int, List[BehaviorEvent]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, Dict[int, dict]]:
    """Temporal leave-one-purchase-out style split per user.

    For each user with >=2 purchases:
      - test target = last purchase
      - val target  = second-to-last purchase (if enough buys)
      - history for a split = all events strictly before that target purchase
        (includes buys/adds/removes), so cart signal before the target is usable.

    Users with only 2 purchases get train+test (no val).
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, train+val < 1.")

    splits = {"train": {}, "val": {}, "test": {}}

    for user_id, seq in sequences.items():
        buy_idx = [i for i, e in enumerate(seq) if e.behavior_id == BEHAVIOR_BUY]
        if len(buy_idx) < 2:
            continue

        # Always hold out last purchase as test target.
        test_pos = buy_idx[-1]
        test_hist = seq[:test_pos]
        test_target = seq[test_pos]
        if not test_hist:
            continue
        splits["test"][user_id] = {
            "history": test_hist,
            "target_item": test_target.item_id,
            "target_time": test_target.timestamp,
        }

        # Validation: second-to-last purchase when available.
        if len(buy_idx) >= 3:
            val_pos = buy_idx[-2]
            val_hist = seq[:val_pos]
            val_target = seq[val_pos]
            if val_hist:
                splits["val"][user_id] = {
                    "history": val_hist,
                    "target_item": val_target.item_id,
                    "target_time": val_target.timestamp,
                }
            # Train uses events before the validation purchase.
            train_hist = seq[:val_pos]
        else:
            # Only two buys: train on events before the test purchase.
            train_hist = test_hist

        # Build next-item training instances from purchases inside train_hist.
        train_buy_idx = [i for i, e in enumerate(train_hist) if e.behavior_id == BEHAVIOR_BUY]
        if len(train_buy_idx) < 1:
            # Need at least one purchase prediction point; use prefix before last
            # event only if there is a buy somewhere — otherwise skip train.
            continue

        # For each purchase after the first event, predict that purchase from prior events.
        instances = []
        for bpos in train_buy_idx:
            if bpos == 0:
                continue
            instances.append(
                {
                    "history": train_hist[:bpos],
                    "target_item": train_hist[bpos].item_id,
                    "target_time": train_hist[bpos].timestamp,
                }
            )
        if instances:
            splits["train"][user_id] = instances

    return splits


def filter_purchase_only(seq: Sequence[BehaviorEvent]) -> List[BehaviorEvent]:
    """Keep only purchase events (for purchase-only SASRec baseline)."""
    return [e for e in seq if e.behavior_id == BEHAVIOR_BUY]


def make_synthetic_events(
    n_users: int = 200,
    n_items: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic multi-behavior event log for smoke tests."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2022-06-01")
    rows = []
    for u in range(1, n_users + 1):
        n_buys = int(rng.integers(3, 8))
        t = start + rng.integers(0, 10).astype("timedelta64[D]")
        for _ in range(n_buys):
            item = int(rng.integers(1, n_items + 1))
            # Often cart before buy.
            if rng.random() < 0.8:
                t = t + rng.integers(1, 5).astype("timedelta64[h]")
                rows.append((u, item, pd.Timestamp(t), "add_to_cart"))
                if rng.random() < 0.15:
                    t = t + rng.integers(1, 3).astype("timedelta64[h]")
                    rows.append((u, item, pd.Timestamp(t), "remove_from_cart"))
                    # maybe re-add
                    if rng.random() < 0.5:
                        t = t + rng.integers(1, 5).astype("timedelta64[h]")
                        rows.append((u, item, pd.Timestamp(t), "add_to_cart"))
            t = t + rng.integers(1, 12).astype("timedelta64[h]")
            rows.append((u, item, pd.Timestamp(t), "product_buy"))
            # occasional unrelated cart
            if rng.random() < 0.3:
                other = int(rng.integers(1, n_items + 1))
                t = t + rng.integers(1, 6).astype("timedelta64[h]")
                rows.append((u, other, pd.Timestamp(t), "add_to_cart"))
    return pd.DataFrame(rows, columns=["client_id", "sku", "timestamp", "behavior"])
