# Behavior-Aware SASRec

Dedicated folder for the cart-aware sequential recommendation modeling task.

Purchase-only SASRec vs behavior-aware SASRec (`add_to_cart` + `remove_from_cart`), same transformer backbone, next-**purchase** targets, temporal split, metrics `HR@K` / `NDCG@K` / `MRR`.

## Layout

```
behavior_aware_sasrec/
├── README.md
├── Behavior_Aware_SASRec.ipynb   # preferred: step-by-step notebook
├── configs/default.yaml
├── train.py                      # optional CLI
├── smoke_test.py                 # quick synthetic CLI check
├── ds5500_recsys/                # Python package used by the notebook
│   ├── data/
│   ├── models/sasrec.py
│   ├── metrics.py
│   └── train.py
└── artifacts/                    # checkpoints (gitignored)
```

## Design

| Variant | History | Target |
|---|---|---|
| `purchase_only` | purchases only | next purchase |
| `behavior_aware` | buy + add-to-cart + remove-from-cart | next purchase |

Behavior-aware model adds a behavior-type embedding (buy / add / remove) to item + position embeddings. Architecture is otherwise identical, so cart signal is isolated from model-capacity changes.

## Run

From the **repo root**:

```bash
pip install -r requirements.txt

# smoke test (no parquet needed)
python behavior_aware_sasrec/smoke_test.py

# train both variants on subsampled cohort
python behavior_aware_sasrec/train.py \
  --data-dir data/subsampled \
  --mode both \
  --top-n-items 20000 \
  --device cuda \
  --epochs 30
```

Or from inside this folder:

```bash
cd behavior_aware_sasrec
python smoke_test.py
python train.py --data-dir ../data/subsampled --mode behavior_aware
```
