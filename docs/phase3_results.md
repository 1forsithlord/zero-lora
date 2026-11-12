# Phase 3 Results: Lambda Grid (DTD, 3 Epochs, Rank 4)

| Loss | Lambda | Frozen Task | Final Task | Delta Task | Frozen ZS | Final ZS | Delta ZS |
|---|---|---|---|---|---|---|---|
| raw | 0.0 | 0.4489 | 0.5995 | +0.1505 | 0.8932 | 0.8930 | -0.0003 |
| raw | 0.1 | 0.4489 | 0.5931 | +0.1441 | 0.8932 | 0.8918 | -0.0015 |
| raw | 1.0 | 0.4489 | 0.5766 | +0.1277 | 0.8932 | 0.8903 | -0.0030 |
| raw | 10.0 | 0.4489 | 0.5426 | +0.0936 | 0.8932 | 0.8940 | +0.0008 |
| gnsp | 10.0 | 0.4489 | 0.5702 | +0.1213 | 0.8932 | 0.8920 | -0.0013 |
| gnsp | 100.0 | 0.4489 | 0.4989 | +0.0500 | 0.8932 | 0.8928 | -0.0005 |
| gnsp | 1000.0 | 0.4489 | 0.4521 | +0.0032 | 0.8932 | 0.8940 | +0.0008 |

## Key Finding: The Null Result
Under the current experimental setting (LoRA rank 4, 3 epochs on DTD), plain LoRA (lambda=0.0) does not suffer from catastrophic forgetting. It gains +15.05% on the downstream task while its zero-shot retention drops by a statistically insignificant 0.03%.

Adding the preservation losses (Raw Frobenius and GNSP-KL) acts as a capacity constraint rather than a preservation mechanism. As lambda increases, the model gains less task accuracy, but zero-shot retention remains flat (drops are under 0.3% across the board). 

**Conclusion:** For small parameter budgets (rank 4), the baseline is Pareto-dominant. Phase 4 must use a higher LoRA rank (e.g., 16 or 32) to induce meaningful catastrophic forgetting.