# DS5500_RecSys
## Behavior-Aware Sequential Recommendation for Next Purchase Prediction

Online retailers often have limited insight into what customers are likely to buy next because purchases are relatively infrequent and only reflect intent after a transaction occurs. As a result, recommendations based mainly on purchase history may rely on older preferences and miss products a user is currently considering. Cart activity may offer a more immediate signal of changing intent, but retailers need to know whether that added information actually improves recommendations enough to justify the extra engineering effort. It is also important to understand whether the usefulness of cart behavior is consistent across products or varies by category.

This project will test whether incorporating cart history improves next-purchase recommendations compared with using purchase history alone. We will use the Synerise RecSys 2025 dataset, which contains six months of anonymized real-world retail interactions and product information, including purchases, cart actions, searches, visits, category IDs, and price buckets. Because the dataset was released for a recommendation challenge, much of the original business context is anonymized, including the identities of specific products and categories. We will compare purchase-only recommendations with behavior-aware versions that also use cart activity, evaluate whether this additional information helps rank the product a user eventually purchases more highly, and examine whether the effect of cart behavior differs across anonymized product categories.

## Repository Structure

```
DS5500_RecSys/
├── README.md                       # This file (proposal + how to run)
├── DataIngestion_Subsample.ipynb   # Step 1: ingest, dedup, filter to K≥5 active users
├── preliminary_eda.ipynb           # Broad sparsity / cart-availability EDA
├── Cart_to_Purchase_EDA.ipynb      # Cart→purchase conversion + time-gap EDA
├── ItemCategoryEDA.ipynb           # Item popularity + category clustering
├── event_level_eda.ipynb           # Per-user sequence lengths, recency windows
└── figures/                        # PNG outputs written by the EDA notebooks
```

Notebooks are not run in a fixed order beyond Step 1 → EDA. Each EDA notebook reads `data/subsampled/*_subsampled.parquet` (gitignored) produced by `DataIngestion_Subsample.ipynb`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `DataIngestion_Subsample.ipynb` | Step 1: ingest raw events, define active users (K≥5), write `data/subsampled/` |
| `preliminary_eda.ipynb` | Broad preliminary EDA (sparsity, cart availability) |
| `event_level_eda.ipynb` | Per-user sequence lengths, 7/14/30-day recency windows |
| `Cart_to_Purchase_EDA.ipynb` | Dedicated cart→purchase conversion rate + time-gap EDA (hypothesis signal) |
| `ItemCategoryEDA.ipynb` | Item popularity, category distribution, category clustering |

## How to Run

1. **Data setup.** Clone the [Synerise RecSys 2025 dataset](https://synerise.com/recsys-2025-challenge/) parquet files into `data/raw/` (paths are configured inside `DataIngestion_Subsample.ipynb`; see the `DATA_DIR` cell).
2. **Run Step 1 first** so `data/subsampled/*_subsampled.parquet` exists:
   ```bash
   jupyter execute DataIngestion_Subsample.ipynb   # or open it in Jupyter and run all
   ```
3. **Then open any EDA notebook.** `Cart_to_Purchase_EDA.ipynb`, `event_level_eda.ipynb`, and `ItemCategoryEDA.ipynb` expect `data/subsampled/*_subsampled.parquet` relative to the notebook (or a `data/subsampled/` under the notebook's directory). `preliminary_eda.ipynb` reads raw `*.parquet` files directly (not subsampled) — see the `files` cell for expected paths.
4. Figures are written to `figures/*.png` (or the notebook working directory); commit-worthy outputs are checked in under `figures/`.

## Requirements

Notebooks use `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `pyarrow` (for parquet I/O). No pinned `requirements.txt` exists yet — see Open Work below.

## Status & Open Work

- EDA is complete for the hypothesis (see `figures/` and each notebook's takeaways).
- **Modeling phase** is the next milestone: build a purchase-history-only baseline and a cart-aware model, with temporal train/test splits and ranking metrics (Recall@K, NDCG@K), following the controlled-experiment setup described in `ItemCategoryEDA.ipynb` §4.
- Follow-ups worth tracking: add `pyproject.toml` + `pytest`, a CI workflow, and refactor notebook logic into a `ds5500_recsys/` package.
