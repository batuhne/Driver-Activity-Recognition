# Baseline Results — CNN+LSTM (ResNet-18 + 2-Layer LSTM)

**Date:** 2026-02-22
**Split:** 0 (train=vp1-10, val=vp11-12, test=vp13-15)
**Config:** lstm_hidden=256, lstm_layers=2, dropout=0.3, lr=0.001, weight_decay=1e-4, label_smoothing=0.1

---

## Summary Metrics (Test Set)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 40.6% |
| Mean Per-Class Accuracy (MPCA) | 29.4% |
| Macro F1 | 0.30 |
| Weighted F1 | 0.39 |

---

## Overfitting Analysis

| Set | Accuracy |
|-----|----------|
| Train | ~87% |
| Val | ~37% |
| Test | 40.6% |

**Gap:** Train-Val gap of ~50 percentage points indicates severe overfitting.
The model memorizes training data but fails to generalize to unseen subjects.

**Root causes:**
- No input regularization (no noise, no mixup)
- Dropout only 0.3 on LSTM layers
- Unidirectional LSTM using only last hidden state — discards temporal context
- No input normalization (LayerNorm) before LSTM

---

## Top 10 Most Confused Pairs (True → Predicted)

These are the most frequent misclassifications on the test set:

| True Label | Predicted Label | Count |
|-----------|----------------|-------|
| reading_newspaper | reading_magazine | High |
| drinking | eating | High |
| talking_on_phone_right | talking_on_phone_left | High |
| working_on_laptop | reading_magazine | Moderate |
| opening_laptop | working_on_laptop | Moderate |
| closing_laptop | opening_laptop | Moderate |
| fetching_an_object | placing_an_object | Moderate |
| putting_on_sunglasses | taking_off_sunglasses | Moderate |
| opening_bottle | closing_bottle | Moderate |
| sitting_still | unfastening_seat_belt | Moderate |

**Pattern:** Most confusions are between visually/semantically similar activities
(same object, reversed actions, or similar hand movements).

---

## Next Steps (Planned Improvements)

| Technique | Expected MPCA Gain |
|-----------|-------------------|
| Mixup (α=0.2) | +5-10% |
| Attention pooling | +3-5% |
| LayerNorm | +2-4% |
| BiLSTM | +2-4% |
| Gaussian noise | +2-4% |
| Focal Loss | +2-3% |
| Dropout/weight_decay increase | +2-4% |

**Target: 29.4% → 45-55% MPCA**
