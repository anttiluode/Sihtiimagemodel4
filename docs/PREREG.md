# Preregistration — Sihtiimagemodel4

Written and committed **before** any test image (synthetic seed 73, or the natural
test photographs camera / coffee / clock) is run through any model in this repository.
Hyperparameters are chosen by `tune.py` on validation images only and frozen to
`params_synthetic.json` / `params_natural.json` before the test run.

## What is being tested

SihtiImageModel3's R1 failed, and the diagnosis was that its coarse level only
echoed the fine level, its ping could never be accepted, and MSE rewards blur.
This repository rebuilds the up-and-down idea as **one energy**:

```
E(x, k) = sum_p  e_{k_p}(Q P_p x)                       local: each 10x10 patch's residue around its own
                                                         mean, under the state k_p of a learned 20-state GMM
        + gamma/2 (D x - mu)^T S^-1 (D x - mu)           global: 8x8 layout of 4x4 block means, learned Gaussian
```

* **Settle** = exact block-coordinate descent: each patch picks its state, then the
  hidden pixels are solved exactly (one linear system that contains the layout prior).
* **Ping** = from a settled state, the most uncertain layout directions of the
  medium (eigenvectors of `L A^-1 L^T`), sent down through the medium as Green's
  functions `A^-1 L^T w`, at ±s posterior SDs; patches re-choose states; settle again.
  Lowest-energy settled state = the ping estimate. All settled states = hypotheses.
* **Sham** = identical, but random layout directions through the same medium with the same pixel norm.
* **Posterior** = Gibbs sampling of (states, pixels); Rao-Blackwellised mean.
* **Linear medium** = the same model with K = 1 (one patch state): linear-Gaussian, ping provably blind.

## Validation observations made before writing this (synthetic validation, seed 11, 24 images)

Recorded so nobody mistakes the gates below for predictions made blind:

* Linear medium with layout prior (gamma 100): MSE 0.0699; four-nearest-retrieval 0.0729.
* 20-state medium settled (gamma 1000): 0.0794; its Gibbs posterior mean 0.0792.
* Ping found ~1–2 distinct settled states per image; best-of-7 barely below the settled state.
* Heating the sampler (dividing the energy by 10–30) made the posterior worse (0.105–0.117).

* Tuning the Wiener baseline fairly (shrinkage grid extended to 0.005–10 after the first grid
  hit its edge) gave **0.0628** on synthetic validation — better than the linear medium (0.0699).

So on validation the **switching local states did not pay**, the **linear medium beat retrieval**,
and the **full-image Gaussian (Wiener) beat everything** on the synthetic contours.
The main model for gate G4 is therefore declared to be the **linear medium** (chosen on validation).

## Gates (thresholds fixed now; lower MSE on hidden pixels is better)

| Gate | Question | Pass condition |
|---|---|---|
| G1a | Does the layout talk down usefully? (linear medium) | `linear_medium` ≥ 5% below `linear_medium_flat` |
| G1b | Same, in the 20-state medium | `states_settle` ≥ 5% below `states_flat` |
| G2 | Do switching local states pay? | `states_settle` ≥ 5% below `linear_medium` |
| G3 | Does the ping beat a sham? | `ping_best_of_7` ≥ 2% below `sham_best_of_7` **and** mean distinct ping hypotheses ≥ 2 |
| G4 | Does the declared model beat the strongest simple method? | `linear_medium` below min(`nearest4`, `wiener`) |

Reported but not gated: `states_ping`, `states_posterior`, best-of-7 against best-of-7 retrieval,
sharpness (gradient energy in the hole relative to the truth), and the R1 numbers for comparison
(R1 CI receipt: linear fill 0.1309, four-nearest 0.0802, SihtiImageModel3 0.0818, relational 0.1314).

Gates are evaluated separately on the synthetic test set (32 images, R1's exact seeds and masks)
and the natural test set (64 crops from three photographs never used for training or validation).
No threshold, arm, or hyperparameter will be changed after the test run. A failed gate stays failed.
