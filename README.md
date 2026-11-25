# DS5500_RecSys
## Behavior-Aware Sequential Recommendation for Next Purchase Prediction

## Abstract

Online retailers often have limited insight into what customers are likely to buy next because purchases are relatively infrequent and only reflect intent after a transaction occurs. As a result, recommendations based mainly on purchase history may rely on older preferences and miss products a user is currently considering. Cart activity may offer a more immediate signal of changing intent, but retailers need to know whether that added information actually improves recommendations enough to justify the extra engineering effort. It is also important to understand whether the usefulness of cart behavior is consistent across products or varies by category.

This project will test whether incorporating cart history improves next-purchase recommendations compared with using purchase history alone. We will use the Synerise RecSys 2025 dataset, which contains six months of anonymized real-world retail interactions and product information, including purchases, cart actions, searches, visits, category IDs, and price buckets. Because the dataset was released for a recommendation challenge, much of the original business context is anonymized, including the identities of specific products and categories. We will compare purchase-only recommendations with behavior-aware versions that also use cart activity, evaluate whether this additional information helps rank the product a user eventually purchases more highly, and examine whether the effect of cart behavior differs across anonymized product categories.

## Repository Structure

```
DS5500_RecSys/
├── README.md                       # This file (proposal + how to run)
├── .gitignore                      # Ignores data/, *.parquet, *.csv, notebook checkpoints
├── evaluation_utils.py             # Shared HitRate@10, NDCG@10, and MRR
├── DataIngestion_Subsample.ipynb   # Step 1: ingest, dedup, filter to K≥5 active users
├── Step1_SequenceConstruction.ipynb
├── Step2_CandidateSampling_PopularityBaseline.ipynb
├── Step2b_Popularity_Test_Evaluation.ipynb
├── Step3_LightFM_PurchaseOnlyBaseline.ipynb
├── Step4_SASRec.ipynb
├── preliminary_eda.ipynb           # Broad sparsity / cart-availability EDA
├── event_level_eda.ipynb           # Per-user sequence lengths, recency windows
├── Cart_to_Purchase_EDA.ipynb      # Cart→purchase conversion + time-gap EDA
├── ItemCategoryEDA.ipynb           # Item popularity + category clustering
├── preliminary_results_combined.png
├── artifacts/                      # sequences_and_splits.pkl, shared_evaluation_candidates.pkl
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
| `Step1_SequenceConstruction.ipynb` | Splits, targets, histories, and the item vocabulary. Writes `artifacts/sequences_and_splits.pkl` |
| `Step2_CandidateSampling_PopularityBaseline.ipynb` | Candidate lists, popularity baseline, and the shared metric code in `evaluation_utils.py`. Writes `artifacts/shared_evaluation_candidates.pkl` |
| `Step2b_Popularity_Test_Evaluation.ipynb` | Popularity scores on the held-out test users |
| `Step3_LightFM_PurchaseOnlyBaseline.ipynb` | Purchase-only LightFM baseline |
| `Step4_SASRec.ipynb` | Purchase-only and behavior-aware SASRec |

## How to Run

The raw dataset is not bundled with the repo (see `.gitignore` → `data/`, `*.parquet`). Download the [Synerise RecSys 2025 dataset](https://synerise.com/recsys-2025-challenge/) parquet files before running anything. The same release is available at [https://recsys.synerise.com/](https://recsys.synerise.com/); column definitions are in the organizers’ repo, [https://github.com/Synerise/recsys2025](https://github.com/Synerise/recsys2025). This pipeline uses `product_buy.parquet`, `add_to_cart.parquet`, and `remove_from_cart.parquet`.

> **Path caveat:** the notebooks do not share one consistent data path today. Each notebook has its own hard-coded paths, so you will likely need to adjust them (or your working directory) before the notebook will run. The exact defaults, read from each notebook's loading cell, are:
>
> | Notebook | Data location it expects |
> |---|---|
> | `DataIngestion_Subsample.ipynb` | Reads `/data/raw/`, writes `/data/subsampled/` |
> | `Cart_to_Purchase_EDA.ipynb` | Tries `/content/subsampled`, `/data/subsampled`, `~/data/subsampled` (first match wins) |
> | `ItemCategoryEDA.ipynb` | `data/*_subsampled.parquet` and `data/product_properties.parquet` (relative to notebook working directory) |
> | `event_level_eda.ipynb` | `*_subsampled.parquet` directly in the current directory (`DATA_DIR = Path(".")`) |
> | `preliminary_eda.ipynb` | Raw `*.parquet` files in the current directory (not subsampled) |
>
> After running `DataIngestion_Subsample.ipynb`, the simplest path is to copy or symlink `/data/subsampled/*_subsampled.parquet` into a `data/` folder next to the notebook you want to run (for `ItemCategoryEDA.ipynb`), or into the notebook's working directory (for `event_level_eda.ipynb`).

1. **Run Step 1** (`DataIngestion_Subsample.ipynb`) to produce the K≥5 active-user subsample. This writes the `*_subsampled.parquet` files the other EDA notebooks load.
2. **Then open any EDA notebook.** Figures are written to `figures/*.png` (or the notebook working directory); the commit-worthy outputs are checked in under `figures/`.

### Modeling run order

After the K ≥ 5 cohort exists, run the modeling notebooks from the repository root in this order:

1. **`DataIngestion_Subsample.ipynb`** — cleaning and the K ≥ 5 cohort.
2. **`Step1_SequenceConstruction.ipynb`** — splits, targets, histories, and the item vocabulary.
3. **`Step2_CandidateSampling_PopularityBaseline.ipynb`** — candidate lists, popularity, and the shared metric code in `evaluation_utils.py`.
4. **`Step2b_Popularity_Test_Evaluation.ipynb`**.
5. **`Step3_LightFM_PurchaseOnlyBaseline.ipynb`** and **`Step4_SASRec.ipynb`** — the LightFM and SASRec notebooks.

Steps 1 and 2 write `artifacts/sequences_and_splits.pkl` and `artifacts/shared_evaluation_candidates.pkl`. LightFM and SASRec load those files and score the same candidates with `evaluation_utils.py`. In the LightFM notebook, keep `RUN_FINAL_TEST = False` while tuning, then set it to `True` and rerun only the final test cell.

## Requirements

Notebooks use `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `pyarrow` (for parquet I/O). No pinned `requirements.txt` exists yet — see Open Work below.

Modeling notebooks also use `scipy` and `lightfm` (`Step3_LightFM_PurchaseOnlyBaseline.ipynb`; `lightfm-next` on newer Python, same `lightfm` import), and `torch` plus `tqdm` (`Step4_SASRec.ipynb`). Category EDA also uses `seaborn`. Install LightFM into the same environment as the notebook kernel, then restart the kernel:

```bash
python -m pip install lightfm
```

SASRec uses CUDA when PyTorch can see a GPU, and CPU otherwise.

## Status & Open Work

- The ingestion and EDA notebooks for the hypothesis are present (see `figures/` and each notebook's takeaways).
- **Modeling phase** is the next milestone: build a purchase-history-only baseline and a cart-aware model, with temporal train/test splits and ranking metrics (Recall@K, NDCG@K), following the controlled-experiment setup described in `ItemCategoryEDA.ipynb` §4.
- Follow-ups worth tracking: add `pyproject.toml` + `pytest`, a CI workflow, refactor notebook logic into a `ds5500_recsys/` package, and normalize the per-notebook data paths above into one shared config.
- Modeling notebooks are now in the repo: popularity (`Step2` / `Step2b`), purchase-only LightFM (`Step3`), and purchase-only plus behavior-aware SASRec (`Step4`). They share the Step 1–2 artifacts and `evaluation_utils.py` (HitRate@10, NDCG@10, MRR).
