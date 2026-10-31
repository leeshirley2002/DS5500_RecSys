# Behavior-Aware SASRec

Dedicated folder for cart-aware sequential recommendation modeling.

## Contents

```
behavior_aware_sasrec/
├── Behavior_Aware_SASRec.ipynb   # self-contained notebook (run this)
└── artifacts/                    # checkpoints (gitignored)
```

## Design

| Variant | History | Target |
|---|---|---|
| `purchase_only` | purchases only | next purchase |
| `behavior_aware` | buy + add-to-cart + remove-from-cart | next purchase |

Same SASRec backbone; cart signal is added via behavior-type embeddings.

## How to run

1. Open `Behavior_Aware_SASRec.ipynb`
2. Set `USE_SYNTHETIC = True` for a dry run, or `False` with `data/subsampled/` for the real cohort
3. Run All

Metrics: `HR@K`, `NDCG@K`, `MRR`.
