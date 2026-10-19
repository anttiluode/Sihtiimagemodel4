# Sihtiimagemodel4 — coarse and fine settling into one image

**One energy. Local patch states below, a global layout above, and a ping that travels down through the medium.**
CPU only. No transformer, no pretrained network, no labels; hidden pixels are never seen by inference.

This is the rebuild of [SihtiImageModel3](https://github.com/anttiluode/SihtiImageModel3)'s Gate R1 after its diagnosis:
R1's coarse level only echoed the fine level, its ping could never be accepted (it searched from a fixed point
and was 25x weaker than the energy gaps), and its scoreboard (MSE) rewards blur. Here the up-and-down loop is
built so that each part can actually matter — and then measured against strong simple methods on
**preregistered gates** ([docs/PREREG.md](docs/PREREG.md), committed before any test image was touched).

## Results (test sets, after freezing everything on validation)

Mean squared error on the hidden pixels — lower is better. **Bold** = best on that set.

| Method | Synthetic contours (R1's 32 images) | Natural crops (64, three unseen photos) |
|---|---:|---:|
| Diffusion fill | 0.1309 | 0.00803 |
| 1 nearest training image | 0.1163 | 0.02835 |
| Average of 4 nearest | 0.0802 | 0.02214 |
| Wiener (full-image Gaussian, tuned) | **0.0677** | 0.00733 |
| *R1 relational model (CI receipt)* | *0.1314* | — |
| *SihtiImageModel3 (CI receipt)* | *0.0818* | — |
| Linear medium (1 state) + layout | 0.0770 | 0.00539 |
| Linear medium, no layout | 0.0854 | 0.00539 |
| 20 states + layout, settled | 0.0890 | 0.00365 |
| 20 states, no layout | 0.1279 | 0.00363 |
| 20 states + ping | 0.0883 | 0.00363 |
| 20 states + sham ping | 0.0886 | 0.00363 |
| 20 states, Gibbs posterior mean | 0.0865 | **0.00362** |

Diffusion and 4-nearest reproduce R1's CI numbers exactly on the synthetic set (same images, seeds and masks).

| Preregistered gate | Synthetic | Natural |
|---|---|---|
| G1a layout helps the linear medium (≥5%) | **PASS** +9.8% | FAIL −0.1% |
| G1b layout helps the 20-state medium (≥5%) | **PASS** +30.4% | FAIL −0.5% |
| G2 switching local states beat the linear medium (≥5%) | FAIL −15.5% | **PASS** +32.3% |
| G3 ping beats sham, best-of-7 (≥2%, ≥2 distinct) | FAIL +0.4% | FAIL +0.2% |
| G4 declared model (linear medium) beats min(4-nearest, Wiener) | FAIL −13.8% | **PASS** +26.5% |

Receipts: [`results/synthetic`](results/synthetic) and [`results/natural`](results/natural) — `metrics.json`
(every image, every arm), `comparison.png`, `loop.gif`.

### What this says

**Each half of the up-and-down loop pays — in a different world, never together.**

* **Top-down (the layout) pays where pictures are layouts.** In the contour world the layout prior cuts error
  30% for the 20-state medium and 10% for the linear one. In natural crops it does nothing: the best layout
  weight on validation was almost zero, and on test it changes nothing.
* **Local states pay where pictures are made of edges.** On natural crops the 20-state medium beats the
  linear medium by 32% and Wiener by 50%; you can see it continue edges cleanly in `results/natural/comparison.png`.
  In the contour world the same states hurt: the commonest state is "empty background", and a hard choice
  erases strokes the linear medium would have half-drawn. That is EPLL's known strength on natural images,
  reproduced, not discovered.
* **The ping is a good search and a useless one.** On the contours it finds much deeper minima than the sham
  (energy drop 18.4 vs 0.8 per image, ~3 distinct settled states each), yet the pictures are no better
  (0.0883 vs 0.0886). A deeper minimum of this energy is not a truer image. On natural crops the states are
  rarely stuck (1.6 distinct states), so there is little to search. Same verdict as SihtiBrain V3 and R1: the
  probe works mechanically; the advantage is not there.
* **Averaging over states (Gibbs) is the best use of them** — best on natural, best of the 20-state variants on
  synthetic — but only by a hair over committing to one.
* On the contours a plain tuned Gaussian over the whole image (Wiener) beats everything here. R1 never ran it.

### Caveats that matter

* The natural test is 64 crops from **three** photographs. On validation (crops from the training photos'
  bottom quarter: grass, gravel, brick, text, …) the 20-state and linear media **tied** (0.00609 vs 0.00605);
  the 32% win appeared on the test photos, which have cleaner edges. Treat the size of that win as
  photo-dependent until it is reproduced on more images — e.g. your own folders.
* The natural parameter grids were extended twice on validation because the choices sat on grid edges
  (recorded in `params_natural.json`); the synthetic noise level sits on its grid edge and was not extended,
  because its test had already been run.
* MSE still rewards blur. All model arms are blurrier than the truth (sharpness 0.4–0.65); retrieval is
  sharper and worse.

## Gate R2 — brightness, summed gradients, or orientation at the soma?

Follow-up to the reading of the 5 a.m. note as "the soma summates **gradients**, not brightness"
([preregistration](docs/GRADIENT_PREREG.md), `python gradients.py --data synthetic|natural`, or
`--train-folder train --eval-folder unseen` for your own pictures).
The one-state linear medium fills the hole; a coarse read-out of the image is either handed in
**true** (an oracle ceiling — what a perfect top-down message of that kind could do) or **predicted**
from the visible surround by a learned Gaussian prior.

| Read-out at 4x4 blocks | Synthetic: true | Synthetic: predicted | Natural: true | Natural: predicted |
|---|---:|---:|---:|---:|
| none | 0.0854 | — | 0.00539 | — |
| brightness (block means) | 0.0451 | 0.0770 | 0.00401 | 0.00539 |
| summed gradients (gx, gy) | 0.0567 | 0.0780 | 0.00489 | 0.00539 |
| brightness + gradients | 0.0367 | **0.0723** | 0.00378 | 0.00539 |
| orientation (structure tensor) | **0.0272** | — | **0.00344** | — |

| Gate | Synthetic | Natural |
|---|---|---|
| R2-O1 true summed gradients beat true brightness (≥10%) | FAIL −25.9% | FAIL −21.7% |
| R2-O2 true gradients add to true brightness (≥5%) | PASS +18.5% | PASS +5.8% |
| R2-O3 true orientation beats true brightness (≥10%) | **PASS +39.6%** | **PASS +14.3%** |
| R2-P1 predicted brightness+gradients beat predicted brightness (≥5%) | PASS +6.2% | FAIL 0.0% |
| R2-P2 predicted gradients alone beat predicted brightness (≥5%) | FAIL −1.2% | FAIL 0.0% |

**What it says.**

* **Linear "summed gradients" are not the missing currency.** Summing a difference over a block telescopes
  to its boundary: it knows the polarity of a step edge, but a line that enters and leaves a block sums to
  about zero. True summed gradients help *less* than true brightness, on both sets.
* **Orientation is.** The per-block structure tensor (gx², gy², gx·gy) — gradient *energy* by direction,
  which is nonlinear and sign-blind — is the best possible top-down message on both sets, and in
  `results/gradients_synthetic/comparison.png` the "TRUE orient." column visibly reconnects the strokes that
  every other column leaves blurred. So the "an average can't say which way the edge runs" point is right;
  the fix is second-order, not a summed first derivative.
* **The truth is not the bottleneck; predicting it is.** On natural crops the true brightness layout would
  cut the error by 26% and true orientation by 36%, yet nothing predicted from the visible surround helps
  at all. On synthetic contours, brightness+gradients predicted together give a real 6% (still behind the
  tuned Wiener filter, 0.0677). The open problem is a predictor for the hidden blocks' **orientation** from
  the visible ring — the oracle says that is where the value is.
* Caveat: the orientation oracle is solved by L-BFGS on a nonconvex energy; its best stiffness sat at the
  edge of its grid on synthetic validation (1e8), so its ceiling may be higher still.

## The model

A 32x32 grey image `x`; visible pixels are clamped, hidden pixels are unknowns.

```
E(x, k) =  sum_p  e_{k_p}( Q P_p x )                       fine:   every 10x10 patch's residue around its own mean,
                                                                    scored by state k_p of a learned 20-state Gaussian mixture
         + gamma/2 (D x - mu)^T S^-1 (D x - mu)              coarse: the 8x8 layout of 4x4 block means, learned Gaussian
```

* `Q P_p x` is the patch minus its own mean — the **local Sihti split**: the fine term only sees residue,
  the coarse term only sees layout.
* `k_p` is a discrete **local state**: which mixture component the patch currently belongs to. It is the
  medium's nonlinearity, the image-model version of SihtiBrain's bistable branch. With one component
  (`K = 1`) the whole model is **linear-Gaussian** — the "linear medium".

### Settle (up and down, exactly)
Block-coordinate descent on that one energy, so `E` can never rise (tested):

1. **local step** — every patch picks its best state for the current image;
2. **global step** — with the states fixed, `E` is quadratic in the hidden pixels; one linear solve
   `A x_h = b` minimises it exactly. The layout prior sits inside `A`, so the coarse level **talks down**
   (hidden pixels are pulled toward a layout consistent with the visible blocks) while the layout it ends up
   with is whatever the fine states make cheap (**talks up**).

### Ping (top-down, through the medium)
A settled state is a local minimum: the discrete states can be stuck. From it, the uncertainty of the coarse
layout *given the medium* is `Cc = L A^-1 L^T` (`L` maps hidden pixels to block means). The ping takes its most
uncertain directions `w_j` and sends them **down through the medium as Green's functions**, `A^-1 L^T w_j`,
at ±s posterior standard deviations. Each patch re-chooses its state under the kicked image, and the system
settles again. A state flips only where the kick carries it across a boundary — state-dependent, as in SihtiBrain.
In the linear medium every kick relaxes straight back: **the ping is provably blind there** (tested).

Controls: a **sham** ping (random layout directions through the same medium, same pixel norm) and a
**Gibbs posterior** (sample states and pixels, average the conditional means) — the honest way to average
over local states instead of committing to one.

## Run it

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q                         # energy never rises, exact global step, no target leak, blind linear ping
python run.py --data synthetic              # R1's exact 32 test images, seeds and masks
python run.py --data natural                # 64 crops of three photographs never used in training
python run.py --train-folder train --eval-folder unseen --eval-limit 32    # your own pictures (e.g. THINGS)

# fairer on your own pictures: tune first on a held-out slice of YOUR train folder
python tune.py --train-folder train          # writes params_folders.json (eval folder never opened)
python run.py --train-folder train --eval-folder unseen --eval-limit 32    # now uses params_folders.json
```

Each run writes `runs/<name>/metrics.json` (every image, every arm), `comparison.png` and `loop.gif`
(one image settling, then each ping and re-settle, with the coarse layout and the local-state map beside it).
Hyperparameters are frozen in `params_synthetic.json` / `params_natural.json` by `tune.py`, which only ever
sees validation images. Folder mode uses `params_folders.json` if `tune.py --train-folder` has made one (15% of the train
folder, at most 32 images, held out as validation), otherwise the natural parameters. Fitting takes a few minutes; evaluation is
several seconds per image because every image runs ~25 settles across the arms.

## Prior art (this is a combination, not an invention)

* Patch-GMM priors with overlapping patches and alternating state choice / exact pixel solve:
  Zoran & Weiss, *From learning models of natural image patches to whole image restoration* (EPLL), ICCV 2011.
* Adding coarser scales to such priors: Papyan & Elad, *Multi-scale patch-based image restoration*, IEEE TIP 2016.
* Patch states with pairwise compatibilities (R1's design): Freeman, Pasztor & Carmichael, *Learning low-level vision*, IJCV 2000.

What is particular to this repository is the framing and the controls: the layout prior and the patch prior
negotiate inside one exactly-minimised energy, the "ping" is the medium's own Green's-function response to its
most uncertain layout direction, and every mechanism is put against its linear / sham / retrieval twin.
