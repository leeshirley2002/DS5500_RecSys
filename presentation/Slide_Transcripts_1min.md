# One-Minute Presentation Transcripts

Speaking pace target: ~140 words/minute. Practice once with a timer.

---

## Slide 1 — Cart → Purchase Signal (~1:00)

Good morning. This slide is the EDA that most directly motivates our hypothesis: **cart history is a useful, recent intent signal**.

On the active-user cohort, about **twenty-five percent** of carted user–item pairs later convert to a purchase — so cart is informative, not noise. At the same time, **eighty-two percent** of purchases were preceded by a same-SKU cart-add, meaning cart is usually an observable precursor of buying.

Timing matters just as much. The **median cart-to-purchase lag is only about eleven minutes**. Among pairs that do convert, **seventy-two percent** buy within one hour, **eighty-seven percent** within a day, and **ninety-five percent** within a week.

So cart events are both common before purchase and very fresh compared with sparse purchase histories. That is why Phase 3 will test cart-augmented sequential models against purchase-only baselines. One caveat: most carts still never convert, so we will treat removes and non-purchases as weak or negative signals, not assume every cart-add becomes a buy.

---

## Slide 2 — Milestones & Risks (~1:00)

Here is our eight-week plan, updated for what we have already finished.

**Phase 1 is done.** We built the shared repo and active-user cohort, completed cart-to-purchase EDA, and finished item and category EDA — including exploratory behavioral clustering. That clustering showed categories differ in cart intensity, but because labels are anonymized we cannot name product types.

**Phase 2 is next:** lock a temporal train–validation–test split, then run purchase-only baselines — Popularity, LightFM, and SASRec — so we have a clear bar to beat.

**Phase 3** tests the core question: retrain SASRec with cart events, then add a category-aware variant. **Phase 4** covers novelty and coverage, stratified evaluation, and final delivery.

On risks: anonymization still limits interpretation — we report anonymized groups only. Scale and sparsity are partly mitigated by the K-greater-than-or-equal-to-five cohort. The open risk is **negative transfer**, which we will check with category-stratified evaluation and confidence intervals.

---

## Quick cue cards (if you forget wording)

**Cart slide ending line:**  
“Cart is informative and timely — so we should test cart-augmented models next.”

**Milestones ending line:**  
“EDA is done; next we set the purchase-only baseline, then test cart and category.”
