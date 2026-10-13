# Milestones & Risks — One Slide (updated after EDA)

**File:** `presentation/Milestones_and_Risks_Slide.pptx` (16:9)

## Correction vs original proposal Table 1

| Original Phase 2 item | Correction |
|---|---|
| “Category clustering” in Phase 2 | **Moved into Phase 1 (done)** as *exploratory* clustering in `ItemCategoryEDA.ipynb` |
| Phase 1 = only preliminary EDA | Phase 1 now includes cart→purchase EDA + item/category EDA |
| Phase 3 = cart then category | Keep both, but category-aware can use **raw category embeddings**; train-only behavioral clusters are optional |

## Current status

### Phase 1 — Done (Wks 1–2): Ingest & EDA
- Shared repo + Pandas ingestion / active-user cohort (**K≥5**, ~520K users)
- Preliminary EDA (sparsity, cart availability)
- Cart→purchase conversion + time-gap EDA (`Cart_to_Purchase_EDA.ipynb`)
- Item popularity + category distribution + exploratory behavioral clustering (`ItemCategoryEDA.ipynb`)

### Phase 2 — Next (Wks 3–4): Purchase-only baselines
- Temporal train / validation / test split (no leakage)
- Popularity, LightFM, SASRec on **purchase-only** sequences
- Fixed candidate item pool + benchmark table (NDCG@10, HitRate@10, MRR)

### Phase 3 — Wks 5–6: Cart- & category-aware
- Cart-augmented SASRec sequences
- Category-conditioned / category-embedding variant
- If using clusters in the model: rebuild **train-period-only** category features/clusters

### Phase 4 — Wks 7–8: Evaluate & deliver
- Novelty / Coverage
- Category-stratified evaluation + ablation
- Final report, slides, public GitHub

## Risks (with EDA updates)
1. **Anonymized categories** — still cannot name product types; EDA shows soft behavioral differences (k=2 cart intensity)
2. **Scale / compute** — mitigated via active cohort (~70% events retained)
3. **Purchase sparsity** — mitigated via K≥5; still need candidate pool + CIs
4. **Negative transfer** — still open until stratified model eval

## One-sentence takeaway
> Phase 1 EDA is complete (including exploratory category clustering); next we set the purchase-only baseline bar, then test whether cart and category signals improve ranking.
