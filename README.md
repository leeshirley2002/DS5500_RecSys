# DS5500_RecSys
## Behavior-Aware Sequential Recommendation for Next Purchase Prediction

Online retailers often have limited insight into what customers are likely to buy next because purchases are relatively infrequent and only reflect intent after a transaction occurs. As a result, recommendations based mainly on purchase history may rely on older preferences and miss products a user is currently considering. Cart activity may offer a more immediate signal of changing intent, but retailers need to know whether that added information actually improves recommendations enough to justify the extra engineering effort. It is also important to understand whether the usefulness of cart behavior is consistent across products or varies by category.

This project will test whether incorporating cart history improves next-purchase recommendations compared with using purchase history alone. We will use the Synerise RecSys 2025 dataset, which contains six months of anonymized real-world retail interactions and product information, including purchases, cart actions, searches, visits, category IDs, and price buckets. Because the dataset was released for a recommendation challenge, much of the original business context is anonymized, including the identities of specific products and categories. We will compare purchase-only recommendations with behavior-aware versions that also use cart activity, evaluate whether this additional information helps rank the product a user eventually purchases more highly, and examine whether the effect of cart behavior differs across anonymized product categories.

## Repository Structure

```
DS5500_RecSys/
├── README.md                       # This file (proposal + how to run)
├── .gitignore                      # Ignores data/, *.parquet, *.csv, notebook checkpoints
├── DataIngestion_Subsample.ipynb   # Ingest, dedup, filter to K≥5 active users
├── Step1_SequenceConstruction.ipynb
├── Step2_CandidateSampling_PopularityBaseline.ipynb
├── Step3_BehaviorAware_SASRec.ipynb  # Cart-aware SASRec on shared Step2 eval
├── Behavior_Aware_SASRec.ipynb       # Redirect → Step3
├── evaluation_utils.py               # Shared HitRate@10 / NDCG@10 / MRR
├── artifacts/                        # Shared splits + candidate sets (+ model runs)
├── preliminary_eda.ipynb
├── event_level_eda.ipynb
├── Cart_to_Purchase_EDA.ipynb
├── ItemCategoryEDA.ipynb
├── preliminary_results_combined.png
└── figures/
```

Notebooks are not run in a fixed order beyond Step 1 → EDA; each EDA notebook loads the `*_subsampled.parquet` files (gitignored) produced by `DataIngestion_Subsample.ipynb`, but not all from the same location — see the path caveat below.

## Notebooks

| Notebook | Purpose |
|---|---|
| `DataIngestion_Subsample.ipynb` | Ingest raw events, define active users (K≥5), write `data/subsampled/` |
| `Step1_SequenceConstruction.ipynb` | Shared purchase sequences + 14/14-day temporal split + `item_to_idx` |
| `Step2_CandidateSampling_PopularityBaseline.ipynb` | Shared 100-neg candidate sets + popularity baseline |
| `Step3_BehaviorAware_SASRec.ipynb` | Cart-aware SASRec on Step2 candidates / `evaluation_utils.py` |
| `preliminary_eda.ipynb` | Broad preliminary EDA (sparsity, cart availability) |
| `event_level_eda.ipynb` | Per-user sequence lengths, 7/14/30-day recency windows |
| `Cart_to_Purchase_EDA.ipynb` | Dedicated cart→purchase conversion rate + time-gap EDA (hypothesis signal) |
| `ItemCategoryEDA.ipynb` | Item popularity, category distribution, category clustering |

## How to Run

The raw dataset is not bundled with the repo (see `.gitignore` → `data/`, `*.parquet`). Download the [Synerise RecSys 2025 dataset](https://synerise.com/recsys-2025-challenge/) parquet files before running anything.

> **Path caveat:** the notebooks do not share one consistent data path today. Each notebook has its own hard-coded paths, so you will likely need to adjust them (or your working directory) before the notebook will run. The exact defaults, read from each notebook's loading cell, are:
>
> | Notebook | Data location it expects |
> |---|---|
> | `DataIngestion_Subsample.ipynb` | Reads `Desktop/DS5500/data/raw/`, writes `Desktop/DS5500/data/subsampled/` |
> | `Cart_to_Purchase_EDA.ipynb` | Tries `/content/subsampled`, `Desktop/DS5500/data/subsampled`, `~/Desktop/DS5500/data/subsampled` (first match wins) |
> | `ItemCategoryEDA.ipynb` | `data/*_subsampled.parquet` and `data/product_properties.parquet` (relative to notebook working directory) |
> | `event_level_eda.ipynb` | `*_subsampled.parquet` directly in the current directory (`DATA_DIR = Path(".")`) |
> | `preliminary_eda.ipynb` | Raw `*.parquet` files in the current directory (not subsampled) |
>
> After running `DataIngestion_Subsample.ipynb`, the simplest path is to copy or symlink `Desktop/DS5500/data/subsampled/*_subsampled.parquet` into a `data/` folder next to the notebook you want to run (for `ItemCategoryEDA.ipynb`), or into the notebook's working directory (for `event_level_eda.ipynb`).

1. **Run Step 1** (`DataIngestion_Subsample.ipynb`) to produce the K≥5 active-user subsample. This writes the `*_subsampled.parquet` files the other EDA notebooks load.
2. **Then open any EDA notebook.** Figures are written to `figures/*.png` (or the notebook working directory); the commit-worthy outputs are checked in under `figures/`.

## Requirements

Notebooks use `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `pyarrow` (for parquet I/O). No pinned `requirements.txt` exists yet — see Open Work below.

## Modeling (shared evaluation)

Downstream models should **not** resample candidates. Load:

- `artifacts/sequences_and_splits.pkl` — train sequences, val/test targets, shared `item_to_idx`
- `artifacts/shared_evaluation_candidates.pkl` — 100 negatives + 1 target per eval user (original product IDs)
- `evaluation_utils.py` — `evaluate_model_dataset(...)`

`Step3_BehaviorAware_SASRec.ipynb` builds directly on Step2: same users/targets/candidates/metrics, training **cart-aware SASRec only** (`add_to_cart` / `remove_from_cart` before `val_cutoff`). Use validation for model selection; keep test for the final comparison.

## Status & Open Work

- Ingestion, EDA, shared sequence/candidate artifacts, popularity baseline, and cart-aware SASRec notebook are present.
- Follow-ups: full-cohort SASRec runs, LightFM / other baselines, category-conditioned variant, `pytest` + CI, normalize notebook data paths.
