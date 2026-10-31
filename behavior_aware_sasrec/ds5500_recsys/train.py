"""Training and evaluation loops for SASRec / BehaviorAwareSASRec."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import SASRecDataset, collate_sasrec
from .metrics import format_metrics, rank_targets, rank_targets_among_candidates, summarize_ranks
from .models.sasrec import BehaviorAwareSASRec, SASRec, bce_ranking_loss


@dataclass
class TrainConfig:
    hidden_size: int = 64
    max_len: int = 50
    n_blocks: int = 2
    n_heads: int = 1
    dropout: float = 0.2
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 20
    n_negatives: int = 1
    eval_negatives: int = 100
    num_workers: int = 0
    seed: int = 42
    device: str = "cpu"
    patience: int = 5


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(
    mode: str,
    n_items: int,
    cfg: TrainConfig,
) -> SASRec:
    common = dict(
        n_items=n_items,
        hidden_size=cfg.hidden_size,
        max_len=cfg.max_len,
        n_blocks=cfg.n_blocks,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    )
    if mode == "purchase_only":
        return SASRec(**common)
    if mode == "behavior_aware":
        return BehaviorAwareSASRec(n_behaviors=3, **common)
    raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def evaluate(
    model: SASRec,
    split_data: Dict[int, object],
    n_items: int,
    cfg: TrainConfig,
    purchase_only: bool,
    use_full_softmax: bool = False,
) -> Dict[str, float]:
    model.eval()
    dataset = SASRecDataset(
        split_data,
        max_len=cfg.max_len,
        purchase_only=purchase_only,
        n_items=n_items,
        n_negatives=0,
        seed=cfg.seed,
    )
    if len(dataset) == 0:
        return {"MRR": 0.0, "HR@10": 0.0, "NDCG@10": 0.0, "n": 0.0}

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_sasrec,
        num_workers=cfg.num_workers,
    )
    device = torch.device(cfg.device)
    all_ranks = []
    rng = np.random.default_rng(cfg.seed)

    for batch in loader:
        item_ids = batch["item_ids"].to(device)
        behavior_ids = batch["behavior_ids"].to(device)
        targets = batch["target_item"].to(device)

        if use_full_softmax or cfg.eval_negatives <= 0:
            scores = model.predict(item_ids, behavior_ids)  # [B, n_items+1]
            ranks = rank_targets(scores, targets)
        else:
            batch_size = targets.size(0)
            # Sample negatives per row; put positive at index 0.
            cand = torch.zeros(batch_size, cfg.eval_negatives + 1, dtype=torch.long, device=device)
            cand[:, 0] = targets
            for i in range(batch_size):
                banned = set(item_ids[i].tolist()) | {int(targets[i].item()), 0}
                negs = []
                while len(negs) < cfg.eval_negatives:
                    neg = int(rng.integers(1, n_items + 1))
                    if neg not in banned and neg not in negs:
                        negs.append(neg)
                cand[i, 1:] = torch.tensor(negs, device=device)
            scores = model.predict(item_ids, behavior_ids, candidate_items=cand)
            pos_index = torch.zeros(batch_size, dtype=torch.long, device=device)
            ranks = rank_targets_among_candidates(scores, pos_index)
        all_ranks.append(ranks)

    ranks_np = np.concatenate(all_ranks)
    return summarize_ranks(ranks_np, ks=(5, 10))


def train_model(
    mode: str,
    sequences_payload: Dict,
    splits: Dict[str, Dict[int, object]],
    cfg: TrainConfig,
    output_dir: Path | str,
) -> Dict[str, object]:
    """Train purchase-only or behavior-aware SASRec and return metrics + paths."""
    set_seed(cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    n_items = int(sequences_payload["n_items"])
    purchase_only = mode == "purchase_only"
    model = build_model(mode, n_items, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = SASRecDataset(
        splits["train"],
        max_len=cfg.max_len,
        purchase_only=purchase_only,
        n_items=n_items,
        n_negatives=cfg.n_negatives,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_sasrec,
        num_workers=cfg.num_workers,
    )

    history = []
    best_ndcg = -1.0
    best_path = output_dir / f"{mode}_best.pt"
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{mode} epoch {epoch}", leave=False):
            item_ids = batch["item_ids"].to(device)
            behavior_ids = batch["behavior_ids"].to(device)
            pos = batch["target_item"].to(device)
            neg = batch["neg_items"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_ranking_loss(model, item_ids, behavior_ids, pos, neg)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(
            model,
            splits["val"] if splits["val"] else splits["test"],
            n_items=n_items,
            cfg=cfg,
            purchase_only=purchase_only,
        )
        test_metrics = evaluate(
            model,
            splits["test"],
            n_items=n_items,
            cfg=cfg,
            purchase_only=purchase_only,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)
        print(
            f"[{mode}] epoch={epoch} loss={row['loss']:.4f} "
            f"val({format_metrics(val_metrics)}) test({format_metrics(test_metrics)})"
        )

        ndcg = val_metrics.get("NDCG@10", 0.0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            patience_left = cfg.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": mode,
                    "n_items": n_items,
                    "config": asdict(cfg),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{mode}] early stopping at epoch {epoch}")
                break

    # Reload best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_test = evaluate(model, splits["test"], n_items, cfg, purchase_only)
    return {
        "mode": mode,
        "best_checkpoint": str(best_path),
        "history": history,
        "best_val": ckpt.get("val_metrics"),
        "test_metrics": final_test,
        "config": asdict(cfg),
    }
