# Run 2 Results — BiLSTM + Attention + Focal Loss + Mixup

**Date:** 2026-02-22
**Split:** 0 (train=vp1-10, val=vp11-12, test=vp13-15)
**Config:** BiLSTM, attention pooling, LayerNorm, focal loss (gamma=2.0), mixup (alpha=0.2), noise_std=0.1, dropout=0.5, weight_decay=0.0005, lr=0.001, epochs=80 (early stopped at epoch 10)

---

## Summary Metrics (Test Set)

| Metric | Baseline | Run 2 | Change |
|--------|----------|-------|--------|
| Overall Accuracy | 40.6% | 34.6% | -6.0% |
| Mean Per-Class Accuracy (MPCA) | 29.4% | 30.4% | +1.0% |
| Macro F1 | 0.30 | 0.28 | -0.02 |

---

## Key Observations

### Early Stopping at Epoch 10
Despite 80 epochs and patience=15 configured, the model stopped at epoch 10. Focal loss + mixup combination likely hindered convergence.

### sitting_still Recall Crisis
The largest class in the test set (704 samples, 32% of test data) had only 34% recall:
- Confused with: fastening_seat_belt (89), eating (75), using_multimedia_display (65)
- Root cause: WeightedRandomSampler + FocalLoss double-balancing was too aggressive toward minority classes, destroying dominant class recall.

### Attention Not Converged
Attention distributions were nearly uniform across all samples — 10 epochs were insufficient for the attention mechanism to learn meaningful temporal patterns.

### Zero-Support Classes Penalizing MPCA
8 classes had 0 test samples but were counted as 0% accuracy in MPCA, artificially penalizing the metric by ~13%.

---

## Bugs Identified from Run 2 Analysis

| # | Bug | Impact |
|---|-----|--------|
| 1 | ImageNet mean/std applied to IR images | Poor feature quality |
| 2 | MPCA counts 0-support classes as 0% | ~13% MPCA penalty |
| 3 | WeightedSampler + FocalLoss together | Double balancing, sitting_still 34% recall |
| 4 | Mixup train accuracy uses original labels | Misleading training logs |
| 5 | FocalLoss with soft targets semantically wrong | Incorrect loss computation |
| 6 | num_classes = max(labels)+1 | Fragile class counting |
| 7 | Config num_classes: 34 but actual: 39 | Misleading config |

---

## Next Steps (Run 3)

All 7 bugs fixed in pipeline. Re-extract features with:
- Correct IR normalization (computed from training data)
- float32 precision
- CrossEntropyLoss only (single imbalance strategy via WeightedSampler)
- Fixed MPCA metric (NaN for 0-support classes)

**Expected improvement: 30.4% MPCA -> 47-60% MPCA**
