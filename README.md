# CS771 — Introduction to Machine Learning

Minor assignment submissions for **CS771 (2025–26 Sem II)**, IIT Kanpur.
Instructor: **Prof. Purushottam Kar**.

Suvid Goel · 231065 · Department of Computer Science and Engineering

| # | Topic | Score |
|---|---|---|
| [1](minor_assignment_1) | Linear separability & LinearSVC hyperparameter limits | 50 / 50 |
| [2](minor_assignment_2) | Generative models for classification and image inpainting with missing pixels | 50 / 50 |
| [3](minor_assignment_3) | EM mixed regression for ozone forecasting | 60 / 60 |

---

## Minor Assignment 1 — Testing the Limits

Studying how SVM hyperparameters behave on 2D synthetic data of varying
separation difficulty.

- **Linear separability proof.** Showed the generated data is linearly separable
  for *every* challenge level `l > 0`, by proving the convex hulls of the two
  classes are disjoint. The positive hull is a capsule spanning two unit circles
  whose centres lie on the line `x + ky = 0`; the distance from the negative
  centroid to that line is `2vk / sqrt(1 + k²) > 2`, so the minimum separation
  `δ ≥ 2 − 1 − 1 > 0`. Separating Hyperplane Theorem then applies.
- **Hyperparameter limits.** Grid-searched `C`, `max_iter` and `tol` at an easy
  level (`l = 0.3`) and a hard one (`l = 250`) to find the cheapest settings
  still giving 100% accuracy — `C = 0.01 / 1 iteration / tol = 2.5` when the
  margin is wide, versus `C = 5000 / 20 iterations / tol = 1e-4` when it is narrow.
- **Penalty and loss.** Compared the valid `penalty × loss` combinations and
  explained the timing order (`l2 + squared_hinge` fastest at 1.25 ms, `l1 +
  squared_hinge` slowest at 28.77 ms) via the smoothness of each objective and
  the soft-thresholding cost of the `l1` active-set search.
- **Scaling.** Confirmed the `O(n)` behaviour of liblinear's dual coordinate
  descent up to `n = 5000`, including the fixed-overhead plateau for small `n`.

**Contents** — [report](minor_assignment_1/CS771_minor_assignment_1_solution.pdf) ·
[question](<minor_assignment_1/Minor Assignment 1 - CS771 2025-26-II.pdf>) ·
[notebook](<minor_assignment_1/Minor Assignment 1 - CS771 2025-26-II.ipynb>)

---

## Minor Assignment 2 — Generative Models with Missing Pixels

One Gaussian `N(μᶜ, Σᶜ)` per digit class fitted on all 60,000 MNIST training
images. At test time a centre block of pixels is censored, and the model must
recover **both** the digit label and the blanked pixels.

**The derivation.** The lecture code infers sequentially — classify from the
observed pixels, then inpaint using the winning class. This submission solves
the joint problem instead:

```
{ŷ, x̂ᵐ} = argmax_{c,v} P[y = c, xᵐ = v | xᵒ]
```

Maximising over `v` analytically (a Gaussian's mode is its mean, with peak
density `(2π)^(−dₘ/2)·|Σ̄ᶜ|^(−1/2)`) leaves a score that differs from the
two-step score by one extra, class-dependent term:

```
score(c) = log P[xᵒ | y=c] + log P[y=c]  −  ½ log|Σ̄ᶜ|
           \________ two-step score ________/   \_ penalty _/

Σ̄ᶜ = Σ^{mm,c} − Σ^{mo,c} (Σ^{oo,c})⁻¹ Σ^{om,c}      (Schur complement)
```

So the two methods are **not** equivalent. The penalty rewards classes that
predict the missing region confidently and punishes those whose conditional
distribution is diffuse; they agree only when `|Σ̄ᶜ|` is the same for all classes.

**The experiment.** Both methods evaluated on all 10,000 test images at five
censoring levels (≈2%, 8%, 18%, 32%, 51% missing), measuring accuracy,
end-to-end inference time and reconstruction quality.

**The result — the interesting part.** At mild censoring the two methods are
indistinguishable. At 51% censoring the *two-step* method wins decisively
(≈0.42 vs ≈0.27 accuracy), and single-step collapses to predicting class **0**
for almost every input. With only ≈384 observed pixels `Σ^{oo,c}` is small and
near-singular, so `(Σ^{oo,c})⁺` is a poor approximation, `Σ̄ᶜ` is numerically
unreliable, and the resulting `log|Σ̄ᶜ|` — large in magnitude because `dₘ` is
large — overrides the trustworthy observed-pixel likelihood. Digit 0 is
geometrically simple and symmetric, so it carries the smallest conditional
log-determinant and the broken penalty systematically selects it. The two-step
method, which ignores that term entirely, is more robust here.

A correct joint derivation does not automatically give a better estimator when
the quantities it depends on are badly conditioned.

**Contents** — [report](minor_assignment_2/CS771_minor_assignment_2_solution.pdf) ·
[`predict_single_step.py`](minor_assignment_2/predict_single_step.py) ·
[lecture base code](minor_assignment_2/lecture_code)

---

## Minor Assignment 3 — EM Mixed Regression

Predicting ozone concentration `Ref. O3 (ppb)` from an air-quality dataset —
10,458 valid timestamps, five features (hour-of-day, temperature, relative
humidity, and two raw ozone sensor voltages). Chronological 80/20 split
(8,366 train / 2,092 test); no shuffling, since it is a time series.
Standardisation uses training statistics only.

**Why one model is not enough.** Ozone follows a daily photochemical cycle —
near-zero overnight, peaking in the early afternoon. A single linear model
averages across every regime and is systematically wrong at both extremes:
plain Ridge gives **15.55 ppb** test MAE.

**The model.** *k* Ridge regressors trained with EM:

- **E-step** — assign each training point to the component whose model has the
  smallest absolute residual (hard, winner-takes-all).
- **M-step** — refit each Ridge (`alpha = 0.01`) on the points assigned to it.
- **Initialisation** — sort training data by O3 and split into *k* equal
  buckets, one Ridge per bucket, so EM starts already roughly separated.

**The routing problem.** At test time the true O3 is unknown, so residuals — and
therefore the E-step — are unavailable. After EM converges, an **RBF-kernel SVM**
(`C=1`, `gamma='scale'`) is trained on the final training assignments and used
at test time to route each point to a component. Every reported MAE uses this
router rather than oracle assignments.

**Findings.**

| Study | Result |
|---|---|
| `n_iter` sweep (1 → 100) | Knee at 30–40 (MAE 9.21 ppb); degrades past 60. Non-monotone at small counts, as hard assignments oscillate across component boundaries before stabilising. |
| `n_components` sweep (1 → 32) | 15.55 ppb at *k*=1 → **9.21 ppb at *k*=4** → 11.12 ppb at *k*=32. Beyond 4, each expert sees fewer samples *and* the router faces finer, more ambiguous boundaries. |
| Component analysis | At *k*=4 the components form an ordered O3 ladder — mean 18.2 / 28.9 / 48.7 / 74.8 ppb — and the highest-O3 component has the latest mean hour (13.4 h), i.e. the afternoon photochemical peak. |
| Feature ablation | Dropping hour-of-day hurts most. Dropping relative humidity actually *improves* MAE for *k* ≥ 4 — it was contributing noise. |
| Decision-tree baseline | Best `DecisionTreeRegressor` scores 10.43 ppb (`max_depth=10`) — worse than 9.21 — but runs inference in 100–180 µs against 102–540 ms, a 600–4500× speed advantage. The mixture's bottleneck is the RBF-SVM router, `O(N_SV · d)` per test point. |

**41% MAE reduction** (15.55 → 9.21 ppb) bought with three orders of magnitude
of inference latency — the accuracy/latency trade-off is the real conclusion.

**Contents** — [report](minor_assignment_3/CS771_minor_assignment_3_solution.pdf) ·
[lecture base code](minor_assignment_3/lecture_code)

---

### A note on attribution

`minor_assignment_*/lecture_code/` and the starter files in `minor_assignment_1/`
are course-provided material by Prof. Purushottam Kar, included so the
submissions are reproducible. The reports and `predict_single_step.py` are my own
work.
