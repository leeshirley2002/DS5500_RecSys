# Behavior-Aware SASRec

PyTorch implementation of **purchase-only SASRec** and **behavior-aware SASRec** that incorporates `add_to_cart` and `remove_from_cart` events for next-**purchase** prediction.

## Design (matches project hypothesis)

| Variant | History | Target |
|---|---|---|
| `purchase_only` | purchases only | next purchase |
| `behavior_aware` | purchases + add-to-cart + remove-from-cart | next purchase |

Both models share the same SASRec transformer backbone. The behavior-aware model adds a **behavior-type embedding** (buy / add / remove) summed with item and position embeddings. Holding architecture fixed isolates the effect of the cart signal.

## Temporal split

Per user with ≥2 purchases:

- **test** target = last purchase; history = all prior events
- **val** target = second-to-last purchase (if ≥3 buys)
- **train** = next-purchase instances from earlier history

Cart events that occur *before* the target purchase remain in the behavior-aware history (no leakage of the target itself).

## Metrics

`HR@5/10`, `NDCG@5/10`, `MRR` with 100 sampled negatives by default (configurable).

## Quick start

```bash
# from repo root
pip install -r requirements.txt

# smoke test (synthetic data, no parquet needed)
python scripts/smoke_test_sasrec.py

# train both variants on subsampled Synerise cohort
python scripts/train_behavior_sasrec.py \
  --data-dir data/subsampled \
  --mode both \
  --top-n-items 20000 \
  --device cuda \
  --epochs 30
```

## Package layout

```
src/ds5500_recsys/
  data/sequences.py   # load events, build sequences, temporal split
  data/dataset.py     # PyTorch Dataset
  models/sasrec.py    # SASRec + BehaviorAwareSASRec
  metrics.py
  train.py
scripts/
  train_behavior_sasrec.py
  smoke_test_sasrec.py
configs/sasrec_default.yaml
```
