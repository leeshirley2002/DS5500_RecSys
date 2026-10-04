# DS5500_RecSys
## Behavior-Aware Sequential Recommendation for Next Purchase Prediction

Online retailers often have limited insight into what customers are likely to buy next because purchases are relatively infrequent and only reflect intent after a transaction occurs. As a result, recommendations based mainly on purchase history may rely on older preferences and miss products a user is currently considering. Cart activity may offer a more immediate signal of changing intent, but retailers need to know whether that added information actually improves recommendations enough to justify the extra engineering effort. 

This project tests whether incorporating cart history improves next-purchase recommendations compared with using purchase history alone. We use the Synerise RecSys 2025 dataset, which contains six months of real-world retail interactions and product information. We compare purchase-only recommendations with behavior-aware versions that also use cart activity and evaluate whether this additional information helps rank the product a user eventually purchases more highly.
