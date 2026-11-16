# Phase 4 Results: Inducing Forgetting (Rank 16, 10 Epochs)

| Loss | Lambda | Frozen Task | Final Task | Delta Task | Frozen ZS | Final ZS | Delta ZS |
|---|---|---|---|---|---|---|---|
| gnsp | 10 | 0.4489 | 0.7367 | +0.2878 | 0.8932 | 0.8905 | -0.0027 |
| raw | 0 | 0.4489 | 0.7596 | +0.3106 | 0.8932 | 0.8702 | -0.0230 |
| raw | 0.1 | 0.4489 | 0.7590 | +0.3101 | 0.8932 | 0.8860 | -0.0073 |
| raw | 1 | 0.4489 | 0.7463 | +0.2973 | 0.8932 | 0.8912 | -0.0020 |
| raw | 10 | 0.4489 | 0.7202 | +0.2713 | 0.8932 | 0.8920 | -0.0012 |

## Key Finding: Raw Frobenius is Pareto-Dominant
By increasing LoRA rank to 16 and training for 10 epochs, we successfully induced catastrophic forgetting (Baseline ZS drop = -2.30%). 

Adding Raw Frobenius regularization acts as a Pareto improvement:
- **`raw λ=0.1`** achieves the same task accuracy (+31.01%) while cutting forgetting by 3x (-0.73%).
- **`raw λ=1.0`** practically eliminates forgetting (-0.20%) while still maintaining a +29.73% task gain.
- **`gnsp λ=10`** is Pareto-dominated by `raw λ=1.0` (less task accuracy, more forgetting).

*Conclusion:* Raw similarity-geometry matching successfully anchors the model, preventing drift without bottlenecking task performance.