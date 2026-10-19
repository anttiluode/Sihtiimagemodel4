# Sihtiimagemodel4 — coarse and fine settling into one image

**One energy. Local patch states below, a global layout above, and a ping that travels down through the medium.**
CPU only. No transformer, no pretrained network, no labels; hidden pixels are never seen by inference.

This is the rebuild of [SihtiImageModel3](https://github.com/anttiluode/SihtiImageModel3)'s Gate R1 after its diagnosis:
R1's coarse level only echoed the fine level, its ping could never be accepted (it searched from a fixed point
and was 25x weaker than the energy gaps), and its scoreboard (MSE) rewards blur. Here the up-and-down loop is
built so that each part can actually matter — and then measured against strong simple methods on
**preregistered gates** ([docs/PREREG.md](docs/PREREG.md), committed before any test image was touched).

RESULTS_PLACEHOLDER

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
python run.py --train-folder train --eval-folder unseen --eval-limit 32    # your own pictures
```

Each run writes `runs/<name>/metrics.json` (every image, every arm), `comparison.png` and `loop.gif`
(one image settling, then each ping and re-settle, with the coarse layout and the local-state map beside it).
Hyperparameters are frozen in `params_synthetic.json` / `params_natural.json` by `tune.py`, which only ever
sees validation images. Folder mode uses the natural parameters. Fitting takes a few minutes; evaluation is
several seconds per image because every image runs ~25 settles across the arms.

## Prior art (this is a combination, not an invention)

* Patch-GMM priors with overlapping patches and alternating state choice / exact pixel solve:
  Zoran & Weiss, *From learning models of natural image patches to whole image restoration* (EPLL), ICCV 2011.
* Adding coarser scales to such priors: Papyan & Elad, *Multi-scale patch-based image restoration*, IEEE TIP 2016.
* Patch states with pairwise compatibilities (R1's design): Freeman, Pasztor & Carmichael, *Learning low-level vision*, IJCV 2000.

What is particular to this repository is the framing and the controls: the layout prior and the patch prior
negotiate inside one exactly-minimised energy, the "ping" is the medium's own Green's-function response to its
most uncertain layout direction, and every mechanism is put against its linear / sham / retrieval twin.
