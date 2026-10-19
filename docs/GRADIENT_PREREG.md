# Gate R2 preregistration — brightness or gradients at the soma?

Committed before `gradients.py` was run on any test image.

## The claim being tested

Gemini's reading of the morning note ("dendrites break the signal into gradients, the soma summates
gradients"): SihtiImageModel3 and this repo failed because their coarse level was a 4x4 **brightness
average**, which cannot say which way an edge runs. Replace it with **summed gradients** and the
top-down message will carry the angle and polarity the hole needs.

Two parts of that claim are checked before testing:

* *"The layout is orthogonal to the patch states, so it does nothing by construction."* Not so here:
  a 10x10 patch spans several 4x4 blocks, so differences between block means **are** patch detail.
  The layout cut the 20-state model's error by 30% on the synthetic test (G1b). On natural images it
  did nothing because validation chose an almost-zero layout weight, not because of orthogonality.
* *"Summed gradients carry orientation."* Only partly. Summing a forward difference over a block
  telescopes: Σ gx over a 4x4 block = Σ over its 4 rows of (pixel just right of the block − first
  pixel of the row). It is a statement about the block's **boundary**, close to a difference of
  neighbouring block means. It carries the polarity of a step edge, but a **line** that enters and
  leaves the block sums to about zero. Orientation of lines lives in second-order statistics — the
  per-block **structure tensor** (mean gx², gy², gx·gy), which is nonlinear in the image.

So R2 tests four read-outs: `mean`, `gradient`, `both`, and `tensor` (orientation), each as an
**oracle** (the true read-out of the target pinned in — a ceiling, never a competitor) and, for the
linear ones, **predicted** (a Gaussian prior on the read-out, filled in from the visible surround).
Base medium: the one-state linear medium with no layout of its own, which won on your THINGS folders.

## Validation observations made while building this (before any test run)

Synthetic validation (24 images), linear medium with no layout = 0.0804:

| read-out | oracle | predicted (best weight) |
|---|---:|---:|
| mean | 0.0442 | 0.0726 |
| gradient | 0.0523 | 0.0743 |
| both | 0.0362 | 0.0662 |
| tensor (orientation) | 0.0329 | — |

Natural validation (60 crops), no layout = 0.00585:

| read-out | oracle | predicted (best weight) |
|---|---:|---:|
| mean | 0.00452 | 0.00585 |
| gradient | 0.00538 | 0.00585 |
| both | 0.00429 | 0.00585 |
| tensor | 0.00503 (L-BFGS, nonconvex, erratic in its stiffness) | — |

So on validation: true gradients helped **less** than true brightness; brightness and gradients together
helped most among linear read-outs; true orientation helped most on strokes; and on natural crops nothing
*predicted* from the visible surround helped at all, even though the truth would have.

## Gates (fixed now)

| Gate | Pass condition |
|---|---|
| R2-O1 | `oracle_gradient` ≥ 10% below `oracle_mean` |
| R2-O2 | `oracle_both` ≥ 5% below `oracle_mean` |
| R2-O3 | `oracle_tensor` ≥ 10% below `oracle_mean` |
| R2-P1 | `pred_both` ≥ 5% below `pred_mean` |
| R2-P2 | `pred_gradient` ≥ 5% below `pred_mean` |

Run separately on the synthetic test (R1's 32 images) and the natural test (64 crops), with weights
chosen inside `gradients.py` on validation only. Failed gates stay failed.
