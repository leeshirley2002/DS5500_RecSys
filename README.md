# DS5500_RecSys
## Behavior-Aware Sequential Recommendation for Next Purchase Prediction

Online retailers often have limited insight into what customers are likely to buy next because purchases are relatively infrequent and only reflect intent after a transaction occurs. As a result, recommendations based mainly on purchase history may rely on older preferences and miss products a user is currently considering. Cart activity may offer a more immediate signal of changing intent, but retailers need to know whether that added information actually improves recommendations enough to justify the extra engineering effort. It is also important to understand whether the usefulness of cart behavior is consistent across products or varies by category.

This project will test whether incorporating cart history improves next-purchase recommendations compared with using purchase history alone. We will use the Synerise RecSys 2025 dataset, which contains six months of anonymized real-world retail interactions and product information, including purchases, cart actions, searches, visits, category IDs, and price buckets. Because the dataset was released for a recommendation challenge, much of the original business context is anonymized, including the identities of specific products and categories. We will compare purchase-only recommendations with behavior-aware versions that also use cart activity, evaluate whether this additional information helps rank the product a user eventually purchases more highly, and examine whether the effect of cart behavior differs across anonymized product categories.

## Repository Structure

```
DS5500_RecSys/
├── README.md                       # This file (proposal + how to run)
├── .gitignore                      # Ignores data/, *.parquet, *.csv, notebook checkpoints
├── DataIngestion_Subsample.ipynb   # Step 1: ingest, dedup, filter to K≥5 active users
├── preliminary_eda.ipynb           # Broad sparsity / cart-availability EDA
├── event_level_eda.ipynb           # Per-user sequence lengths, recency windows
├── Cart_to_Purchase_EDA.ipynb      # Cart→purchase conversion + time-gap EDA
├── ItemCategoryEDA.ipynb           # Item popularity + category clustering
├── preliminary_results_combined.png
└── figures/                        # PNG outputs written by the EDA notebooks
```

Notebooks are not run in a fixed order beyond Step 1 → EDA; each EDA notebook loads the `*_subsampled.parquet` files (gitignored) produced by `DataIngestion_Subsample.ipynb`, but not all from the same location — see the path caveat below.

## Notebooks

| Notebook | Purpose |
|---|---|
| `DataIngestion_Subsample.ipynb` | Step 1: ingest raw events, define active users (K≥5), write `data/subsampled/` |
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

## Status & Open Work

- The ingestion and EDA notebooks for the hypothesis are present (see `figures/` and each notebook's takeaways).
- **Modeling phase** is the next milestone: build a purchase-history-only baseline and a cart-aware model, with temporal train/test splits and ranking metrics (Recall@K, NDCG@K), following the controlled-experiment setup described in `ItemCategoryEDA.ipynb` §4.
- Follow-ups worth tracking: add `pyproject.toml` + `pytest`, a CI workflow, refactor notebook logic into a `ds5500_recsys/` package, and normalize the per-notebook data paths above into one shared config.
