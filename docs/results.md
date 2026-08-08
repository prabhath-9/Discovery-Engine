| Model | Recall@20 | NDCG@20 | Coverage | Latency p50 |
| --- | --- | --- | --- | --- |
| Two-tower + ranker | 0.0276 | 0.0171 | 0.4533 | 24.02ms |
| Multi-intent | 0.0185 | 0.0105 | 0.8629 | 6.55ms |
| Two-tower | 0.0154 | 0.0089 | 0.8130 | 0.80ms |
| Baseline | 0.0259 | 0.0172 | 0.5010 | 0.45ms |

## Ranker weak-label simulation

`trainer/train_ranker.py` trains 4 LightGBM binary boosters (click, cart, purchase, wishlist)
on top of the two-tower + session-encoder features. **H&M transaction data has no click,
cart, or wishlist events — only purchases.** The following labels are simulated, not
measured, and should not be read as real user engagement rates:

- `purchase = 1` for the actual transacted item; `0` for every sampled negative candidate.
  This is the only label backed by real data.
- `click = 1` for the purchased item (a purchase implies a click) and for any negative
  candidate whose `category_l1` matches a category the user already interacted with earlier
  in the same session — a proxy for "browsed the aisle," not an observed click.
- `cart = 1` for a purchased item with probability 0.4, drawn independently per example;
  always `0` for negatives. This simulates a funnel drop-off and does not reflect any
  real add-to-cart signal.
- `wishlist = 1` for a purchased item with probability 0.2, drawn independently per
  example; always `0` for negatives. Same caveat as `cart`.

The `item_ctr` input feature is also a fabricated proxy — `n_interactions / (n_interactions
+ 20)`, a smoothed popularity transform — since no impression/click log exists to compute a
real CTR.

These simulated labels exist to demonstrate a multi-objective ranking architecture end to
end on purchase-only historical data. Treat any click/cart/wishlist metric derived from this
ranker as illustrative of the pipeline, not as evidence about real user behavior.
