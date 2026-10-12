# Milestones & Risks — One Slide

**File:** `presentation/Milestones_and_Risks_Slide.pptx` (16:9)  
**Source:** DS5500 Project Proposal (Table 1 + Risks & Challenges)

## Title
Milestones & Risks

## Four phases (8 weeks)

| Phase | Weeks | Focus | Key deliverables |
|---|---|---|---|
| **1** | 1–2 | Ingest & EDA | Repo/pipeline, active-user cohort, preliminary EDA |
| **2** | 3–4 | Baselines | Category clustering; Popularity / LightFM / SASRec; eval + temporal split |
| **3** | 5–6 | Cart-aware models | Cart-augmented sequences; retrain SASRec; category-conditioned variant |
| **4** | 7–8 | Evaluate & deliver | Novelty/Coverage; ablation + report; slides + public GitHub |

## Risks & mitigations

1. **Anonymized categories** — no human-readable labels → interpret carefully; report anonymized groups  
2. **Scale / compute** — full logs are large → stratified active-user subsample + Parquet cache  
3. **Purchase sparsity** — noisy rare-category estimates → K≥5 threshold + bootstrap CIs  
4. **Negative transfer** — one category model may hurt some groups → category-stratified evaluation  

## One-sentence takeaway
> We move from clean data → purchase-only baselines → cart-aware / category-aware models → final evaluation, with explicit mitigations for anonymization, scale, sparsity, and negative transfer.

## Speaker notes (~45 sec)
1. Eight weeks, four phases — each has a concrete deliverable.  
2. Phase 1–2 set the data and purchase-only bar.  
3. Phase 3 is the core experiment: does cart (and category) help?  
4. Phase 4 checks popularity bias and packages the story.  
5. Biggest risks are anonymized categories and sparsity; we mitigate with cohort filtering, CIs, and stratified eval.
