# Cart → Purchase Signal — One Slide

**File:** `presentation/Cart_to_Purchase_Signal_Slide.pptx` (16:9)  
**Source:** `Cart_to_Purchase_EDA.ipynb` on the active-user cohort (K≥5)

## Title
Cart → Purchase Signal: Why Cart History Matters

## Three headline numbers
| Number | Meaning |
|---|---|
| **24.8%** | Pair-level cart→purchase conversion (878K / 3.5M carted user–SKU pairs) |
| **81.7%** | Share of purchases preceded by a same-SKU cart-add |
| **~11 min** | Median time from first cart-add to purchase (0.18 hours) |

## Timing (among eventual converters)
- **72%** within 1 hour
- **87%** within 24 hours
- **95%** within 7 days

## One-sentence takeaway (say this)
> Cart is both common before purchase and fast after cart-add — so it is a stronger *recent intent* signal than sparse purchase history alone, which is why we should test cart-augmented sequential models.

## Speaker notes (30–45 sec)
1. Purchase histories are short for many users; cart is more frequent.
2. About 1 in 4 carted items later get bought — informative, not noise.
3. Most purchases of an item were preceded by carting that same item (81.7%).
4. When conversion happens, it is usually soon (median ~11 minutes; 87% within a day).
5. But 75% of carts never convert, so we should not treat every cart-add as a guaranteed buy — removes / non-buys stay useful as weak or negative signal.

## Optional chart to show alongside
If you want a second visual: `figures/cart_to_purchase_time_gap.png` (CDF of lag).
