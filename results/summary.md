## Performance and race-level fairness

Positive class: 0 (no recidivism). SP = share predicted favourable; TPR/FPR computed with 0 as the positive class.

| Classifier | Accuracy | Precision | Recall | F1 | SP AA | SP CC | SP diff (CC - AA) | TPR AA | TPR CC | FPR AA | FPR CC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. Logistic regression | 0.675 | 0.668 | 0.774 | 0.717 | 0.485 | 0.799 | 0.314 | 0.688 | 0.862 | 0.312 | 0.691 |
| 2. Without race | 0.677 | 0.671 | 0.768 | 0.717 | 0.492 | 0.772 | 0.280 | 0.695 | 0.844 | 0.318 | 0.648 |
| 3. Reweighing | 0.669 | 0.662 | 0.770 | 0.712 | 0.581 | 0.671 | 0.090 | 0.779 | 0.761 | 0.411 | 0.519 |
| 4. Equalized-odds post-processing | 0.563 | 0.564 | 0.783 | 0.656 | 0.720 | 0.760 | 0.040 | 0.779 | 0.786 | 0.670 | 0.716 |
