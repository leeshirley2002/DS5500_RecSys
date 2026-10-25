# Cart-to-Purchase Signal Exploratory Data Analysis

**Project:** Behavior-Aware Sequential Recommendation for Next Purchase Prediction (DS5500)  
**Source notebook:** `Cart_to_Purchase_EDA.ipynb`  
**Cohort:** Active users with \(K \geq 5\) total purchase/cart events (output of `DataIngestion_Subsample.ipynb`)  
**Observation window:** 2022-06-23 → 2022-12-08  

---

## 1. Introduction

Purchase histories in retail recommendation settings are often short and sparse, so models that rely only on past purchases may miss products a user is currently considering. Add-to-cart events are more frequent and may reflect more recent intent. This report analyzes the **cart-to-purchase signal** on the Synerise RecSys 2025 active-user cohort—the exploratory analysis that most directly motivates the project hypothesis.

We ask two questions:

1. **Conversion:** How often does adding an item to the cart lead to a later purchase of the same item by the same user?
2. **Timing:** When conversion occurs, how soon after the first cart-add does the purchase happen?

If conversion is material (not near zero) and many conversions occur in a short window, cart history is both informative and timely relative to sparse purchase histories, supporting cart-augmented sequential recommenders over purchase-only baselines.

---

## 2. Methods

### 2.1 Data

We use three event files from the Step 1 active-user subsample:

| Event type | Rows | Users | SKUs |
|---|---:|---:|---:|
| Add-to-cart | 4,890,231 | 510,728 | 1,069,877 |
| Product buy | 1,171,501 | 293,044 | 449,958 |
| Remove-from-cart | 2,229,315 | 423,386 | 741,269 |

Records were deduplicated and timestamps parsed. Analyses are restricted to this cohort and therefore describe more active shoppers; they may not generalize to the full Synerise population.

### 2.2 Unit of analysis

The primary unit is the unique **(client_id, sku)** pair.

- **First cart time:** earliest add-to-cart timestamp for that pair  
- **First purchase time:** earliest purchase timestamp for that pair  

A pair is a **conversion** if a purchase of that SKU occurs **strictly after** the first cart-add. This definition matches the project’s preliminary EDA and excludes purchases that precede the cart signal.

### 2.3 Conversion metrics

- **Pair-level conversion rate:** share of carted user–SKU pairs that subsequently convert  
- **Prior-cart share among purchases:** share of purchased user–SKU pairs with a same-SKU cart-add before the purchase  
- **User-level conversion rate:** among users with ≥1 cart event, share with ≥1 converted pair  
- **Remove-from-cart stratification:** intervening remove defined as a remove after the first cart and before any subsequent purchase (or with no purchase)

### 2.4 Time-gap analysis

For converted pairs only, we compute

\[
\text{gap} = t_{\text{first purchase}} - t_{\text{first cart}}
\]

in hours and days. We report percentiles, cumulative conversion within fixed windows (1 hour through 30 days), and coarse buckets (0–1h, 1–24h, 1–7d, 7–30d, >30d).

### 2.5 Scope and limitations of methods

- Pair-level first-cart → first-later-purchase is an aggregate intent measure, not a session-level conversion probability.  
- The Synerise release has no session IDs; timing buckets are calendar elapsed time.  
- Category-stratified conversion is out of scope for this report (covered separately in item/category EDA).

---

## 3. Results

### 3.1 Conversion rates

| Metric | Result |
|---|---|
| Carted user–SKU pairs | 3,537,047 |
| Subsequently purchased pairs | 878,617 |
| **Pair-level conversion rate** | **24.84%** |
| Purchased pairs with prior same-SKU cart | **81.71%** |
| Users with ≥1 cart event | 510,728 |
| Users with ≥1 conversion | 269,271 |
| **User-level conversion (among cart users)** | **52.72%** |

Roughly one in four carted user–item pairs later converts. Among pairs that are purchased, a large majority (81.7%) had a prior same-SKU cart-add. About half of carting users convert at least once.

### 3.2 Remove-from-cart as a competing signal

| Group | Carted pairs | Converted | Conversion rate |
|---|---:|---:|---:|
| No intervening remove | 1,966,261 | 702,449 | **35.73%** |
| Removed before purchase / no buy | 1,570,786 | 176,168 | **11.22%** |

Pairs without an intervening remove convert at more than three times the rate of pairs with a remove before purchase (or with no purchase after remove). Remove-from-cart is associated with substantially weaker purchase follow-through.

### 3.3 Time from cart-add to purchase

Among 878,617 converted pairs with a positive gap:

| Statistic | Hours |
|---|---:|
| P25 | 0.05 |
| **Median** | **0.18 (~11 minutes)** |
| P75 | 1.63 |
| P90 | 47.64 |
| Mean | 31.65 |

The distribution is heavily right-skewed: the mean is much larger than the median because of a long tail of delayed purchases.

### 3.4 Cumulative timing among converters

| Window | Share of eventual converters | Share of all carted pairs |
|---|---:|---:|
| ≤ 1 hour | **72.13%** | 17.92% |
| ≤ 6 hours | 80.02% | 19.88% |
| ≤ 24 hours | **86.52%** | 21.49% |
| ≤ 7 days | **95.49%** | 23.72% |
| ≤ 30 days | 99.07% | 24.61% |
| Ever | 100% | 24.84% |

Bucket shares among converters: **72.1%** in 0–1h, 14.4% in 1–24h, 9.0% in 1–7d, 3.6% in 7–30d, and 0.9% beyond 30 days.

---

## 4. Discussion

### 4.1 Cart is informative, not noise

A pair-level conversion rate of ~25% is far from negligible. Cart history therefore carries predictive information about which items a user may buy next. Treating cart events as pure noise relative to purchase-only baselines would discard a substantial share of eventual purchases that are preceded by cart activity (81.7% of purchased pairs).

### 4.2 Cart is timely relative to sparse purchases

Most conversions happen quickly: median lag ≈ 11 minutes, and 86.5% of converters purchase within 24 hours. In a setting where many users have very short purchase histories, recent cart events are typically **more recent** than older purchases. This supports the project claim that cart activity can supply fresher intent than purchase history alone.

### 4.3 Cart is incomplete as a purchase label

About 75% of carted pairs never convert. An intervening remove is associated with much lower conversion (11.2% vs 35.7%). Modeling implications:

- Do **not** treat every cart-add as a guaranteed future purchase.  
- Keep remove-from-cart and non-conversion as **weak or negative** signals in multi-behavior sequences.  
- Cart-augmented models should encode event type (buy / add / remove), not collapse all interactions into one positive class.

### 4.4 Link to the modeling plan

These results motivate Phase 3 experiments that hold architecture fixed (e.g., SASRec) and compare purchase-only sequences to sequences that also include cart adds and removes. The short lag distribution also suggests that recency-aware or time-aware encodings of cart events may matter. Category heterogeneity is not resolved here; whether cart usefulness varies by anonymized category remains a separate modeling question.

### 4.5 Limitations

- Results apply to the **active-user cohort** (\(K \geq 5\)), not the full population.  
- First-cart / first-purchase pairing ignores repeated carting and multi-purchase dynamics.  
- Elapsed-time gaps are not true “sessions.”  
- Observational associations (e.g., remove vs conversion) are not causal.  
- No category or price stratification in this report.

---

## 5. Conclusion

On the Synerise RecSys 2025 active-user cohort, cart-to-purchase behavior supports the project hypothesis in three ways:

1. **Material conversion:** ~24.8% of carted user–SKU pairs later purchase the same item.  
2. **Strong precursor role:** ~81.7% of purchased pairs had a prior same-SKU cart-add.  
3. **Short lag:** median cart→purchase time ≈ 11 minutes; ~87% of converters buy within 24 hours and ~95% within 7 days.

Together, these findings indicate that cart history is an **informative and timely** signal of purchase intent, while incomplete conversion and remove-from-cart patterns caution against treating cart-adds as hard positives. The appropriate next step is a controlled comparison of purchase-only versus cart-augmented sequential recommenders under a temporal evaluation split.

---

## Appendix: Key figures

Generated by `Cart_to_Purchase_EDA.ipynb` (also under `figures/`):

- `cart_to_purchase_conversion.png` — conversion composition and prior-cart share among purchases  
- `cart_to_purchase_time_gap.png` — gap histogram and empirical CDF  
- `cart_to_purchase_gap_buckets.png` — timing buckets among converters  
