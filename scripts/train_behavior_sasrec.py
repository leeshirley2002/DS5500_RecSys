#!/usr/bin/env python3
"""Train purchase-only and/or behavior-aware SASRec on Synerise subsampled events.

Examples
--------
# Smoke test with synthetic data (no parquet required):
python scripts/train_behavior_sasrec.py --synthetic --epochs 3 --mode both

# Real data (expects data/subsampled/*_subsampled.parquet):
python scripts/train_behavior_sasrec.py \\
  --data-dir data/subsampled \\
  --mode behavior_aware \\
  --top-n-items 20000 \\
  --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds5500_recsys.data.sequences import (  # noqa: E402
    build_user_sequences,
    load_behavior_events,
    make_synthetic_events,
    temporal_split_sequences,
)
from ds5500_recsys.metrics import format_metrics  # noqa: E402
from ds5500_recsys.train import TrainConfig, train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, default="data/subsampled", help="Directory with subsampled parquet files")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic events instead of parquet")
    p.add_argument("--mode", choices=["purchase_only", "behavior_aware", "both"], default="behavior_aware")
    p.add_argument("--top-n-items", type=int, default=None, help="Optional candidate pool from top purchased SKUs")
    p.add_argument("--min-buys", type=int, default=2)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-negatives", type=int, default=1)
    p.add_argument("--eval-negatives", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default="artifacts/models")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("Generating synthetic multi-behavior events...")
        events = make_synthetic_events(n_users=300, n_items=150, seed=args.seed)
        include = ("product_buy", "add_to_cart", "remove_from_cart")
    else:
        print(f"Loading events from {args.data_dir}...")
        include = ("product_buy", "add_to_cart", "remove_from_cart")
        events = load_behavior_events(args.data_dir, include_behaviors=include)

    print(f"Events: {len(events):,} | behaviors={events['behavior'].value_counts().to_dict()}")
    payload = build_user_sequences(
        events,
        include_behaviors=include,
        min_buys=args.min_buys,
        max_len=None,
        top_n_items=args.top_n_items,
    )
    print(
        f"Users={payload['n_users']:,} items={payload['n_items']:,} "
        f"sequences={len(payload['sequences']):,}"
    )

    splits = temporal_split_sequences(payload["sequences"])
    print(
        f"Split sizes: train_users={len(splits['train']):,} "
        f"val={len(splits['val']):,} test={len(splits['test']):,}"
    )

    cfg = TrainConfig(
        hidden_size=args.hidden_size,
        max_len=args.max_len,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        n_negatives=args.n_negatives,
        eval_negatives=args.eval_negatives,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
    )

    modes = ["purchase_only", "behavior_aware"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        print(f"\n=== Training {mode} ===")
        result = train_model(mode, payload, splits, cfg, out_dir / mode)
        results[mode] = {
            "test_metrics": result["test_metrics"],
            "best_val": result["best_val"],
            "best_checkpoint": result["best_checkpoint"],
            "config": result["config"],
        }
        print(f"[{mode}] final test: {format_metrics(result['test_metrics'])}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
