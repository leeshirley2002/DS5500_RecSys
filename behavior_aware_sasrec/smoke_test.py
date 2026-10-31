#!/usr/bin/env python3
"""Quick smoke test: synthetic data, both SASRec variants, few epochs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ds5500_recsys.data.sequences import (  # noqa: E402
    build_user_sequences,
    make_synthetic_events,
    temporal_split_sequences,
)
from ds5500_recsys.metrics import format_metrics  # noqa: E402
from ds5500_recsys.train import TrainConfig, train_model  # noqa: E402


def main() -> None:
    events = make_synthetic_events(n_users=250, n_items=120, seed=0)
    payload = build_user_sequences(
        events,
        include_behaviors=("product_buy", "add_to_cart", "remove_from_cart"),
        min_buys=2,
    )
    splits = temporal_split_sequences(payload["sequences"])
    cfg = TrainConfig(
        hidden_size=32,
        max_len=30,
        n_blocks=1,
        n_heads=1,
        batch_size=64,
        epochs=3,
        eval_negatives=50,
        patience=3,
        device="cpu",
        seed=0,
    )
    out = ROOT / "artifacts" / "smoke"
    results = {}
    for mode in ("purchase_only", "behavior_aware"):
        print(f"\n=== smoke {mode} ===")
        result = train_model(mode, payload, splits, cfg, out / mode)
        results[mode] = result["test_metrics"]
        print(format_metrics(result["test_metrics"]))

    summary = out / "smoke_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2))
    print(f"OK -> {summary}")


if __name__ == "__main__":
    main()
