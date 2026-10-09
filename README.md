# DS5500_RecSys
## Behavior-Aware Sequential Recommendation for Next Purchase Prediction

Online retailers often have limited insight into what customers are likely to buy next because purchases are relatively infrequent and only reflect intent after a transaction occurs. As a result, recommendations based mainly on purchase history may rely on older preferences and miss products a user is currently considering. Cart activity may offer a more immediate signal of changing intent, but retailers need to know whether that added information actually improves recommendations enough to justify the extra engineering effort. It is also important to understand whether the usefulness of cart behavior is consistent across products or varies by category.

This project will test whether incorporating cart history improves next-purchase recommendations compared with using purchase history alone. We will use the Synerise RecSys 2025 dataset, which contains six months of anonymized real-world retail interactions and product information, including purchases, cart actions, searches, visits, category IDs, and price buckets. Because the dataset was released for a recommendation challenge, much of the original business context is anonymized, including the identities of specific products and categories. We will compare purchase-only recommendations with behavior-aware versions that also use cart activity, evaluate whether this additional information helps rank the product a user eventually purchases more highly, and examine whether the effect of cart behavior differs across anonymized product categories.

## Notebooks

| Notebook | Purpose |
|---|---|
| `DataIngestion_Subsample.ipynb` | Step 1: ingest raw events, define active users (K≥5), write `data/subsampled/` |
| `preliminary_eda.ipynb` | Broad preliminary EDA (sparsity, cart availability) |
| `Cart_to_Purchase_Signal_EDA.ipynb` | Dedicated cart→purchase conversion rate + time-gap EDA (hypothesis signal) |

Run Step 1 first so `data/subsampled/*_subsampled.parquet` exists, then open `Cart_to_Purchase_Signal_EDA.ipynb`.
