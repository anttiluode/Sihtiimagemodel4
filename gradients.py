"""Gate R2: does the soma read out gradients rather than brightness?

The idea under test (from the 5 a.m. note, sharpened by Gemini): the coarse /
"soma" level should carry SUMMED GRADIENTS, not 4x4 brightness averages,
because an average cannot say which way an edge runs.

Three candidate read-outs C of a 32x32 image, all at 4x4 blocks:
    mean      block means                                   (what the model used)
    gradient  block sums of gx, gy  (linear "summated gradients")
    both      the two together
    tensor    block structure tensor (gx^2, gy^2, gx*gy)    (orientation; nonlinear)

Two questions, kept apart:
    ORACLE    hand the medium the TRUE read-out of the hidden region. How much
              could a perfect top-down message of that kind help? (ceiling)
    PREDICTED learn a Gaussian prior on the read-out, predict it from the
              visible surround, let it talk down. (what is actually usable)

    python gradients.py --data synthetic
    python gradients.py --data natural
    python gradients.py --train-folder train --eval-folder unseen --eval-limit 32

Hyperparameters (layout weight per read-out, tensor stiffness) are chosen on
validation images only, inside this script, before the test images are loaded.
Gates are fixed in docs/GRADIENT_PREREG.md.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from data import synthetic, damaged_set, natural, load_images, check_disjoint
from sihti4 import Sihti4, learn_priors, with_coarse
from bench import mse
from tune import folder_split
import oracle
import visuals

GAMMAS = [0.3, 3., 30., 100., 300., 1000.]
TENSOR_LAMS = [1e5, 1e6, 1e7, 1e8]
ORACLE_LAM = 1e4
NOISE = {"synthetic": 0.01, "natural": 0.0003, "folders": 0.0003}


def mean_mse(fn, cases):
    return float(np.mean([mse(fn(o, m, t), t, m) for t, o, m, _ in cases]))


def choose(train, val, noise, log):
    pri1 = learn_priors(train, K=1, patch=10)
    flat = Sihti4.from_priors(pri1, gamma=0., noise=noise)
    pick = {}
    for kind in ("mean", "gradient", "both"):
        pk = with_coarse(pri1, train, kind)
        scores = {g: mean_mse(lambda o, m, t, md=Sihti4.from_priors(pk, gamma=g, noise=noise):
                              md.infer(o, m, mode="settle")["map"], val) for g in GAMMAS}
        log[f"pred_{kind}"] = scores
        pick[f"gamma_{kind}"] = min(scores, key=scores.get)
        print(f"  validation, predicted {kind:8s}", {k: round(v, 5) for k, v in scores.items()}, flush=True)
    scores = {lam: mean_mse(lambda o, m, t, lam=lam: oracle.tensor_oracle(flat, o, m, t, lam=lam), val)
              for lam in TENSOR_LAMS}
    log["oracle_tensor"] = scores
    pick["tensor_lam"] = min(scores, key=scores.get)
    print("  validation, oracle tensor   ", {f"{k:g}": round(v, 5) for k, v in scores.items()}, flush=True)
    return pri1, pick


def evaluate(train, cases, pri1, pick, noise):
    flat = Sihti4.from_priors(pri1, gamma=0., noise=noise)
    pred = {k: Sihti4.from_priors(with_coarse(pri1, train, k), gamma=pick[f"gamma_{k}"], noise=noise)
            for k in ("mean", "gradient", "both")}
    records, images = [], []
    for i, (t, o, m, kind) in enumerate(cases):
        out = {"flat": flat.infer(o, m, mode="settle")["map"]}
        for k in ("mean", "gradient", "both"):
            out[f"oracle_{k}"] = oracle.linear_oracle(flat, o, m, t, k, lam=ORACLE_LAM)
            out[f"pred_{k}"] = pred[k].infer(o, m, mode="settle")["map"]
        out["oracle_tensor"] = oracle.tensor_oracle(flat, o, m, t, lam=pick["tensor_lam"])
        rec = {k: mse(v, t, m) for k, v in out.items()}
        rec.update(index=i, mask_kind=kind)
        records.append(rec)
        images.append((t, o, m, out))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(cases)} images", flush=True)
    return records, images


def gates(means):
    def g(a, b, margin):
        gain = 1 - means[a] / means[b]
        return {"arm": a, "against": b, "improvement": round(gain, 4), "needed": margin, "pass": bool(gain >= margin)}
    return {
        "R2-O1 true summed gradients beat true brightness (>=10%)": g("oracle_gradient", "oracle_mean", .10),
        "R2-O2 true gradients add to true brightness (>=5%)": g("oracle_both", "oracle_mean", .05),
        "R2-O3 true orientation beats true brightness (>=10%)": g("oracle_tensor", "oracle_mean", .10),
        "R2-P1 predicted brightness+gradients beat predicted brightness (>=5%)": g("pred_both", "pred_mean", .05),
        "R2-P2 predicted gradients alone beat predicted brightness (>=5%)": g("pred_gradient", "pred_mean", .05),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "natural"], default="synthetic")
    ap.add_argument("--train-folder")
    ap.add_argument("--eval-folder")
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--output")
    a = ap.parse_args()
    t0 = time.time()
    if a.train_folder:
        if not a.eval_folder:
            raise SystemExit("give --eval-folder too")
        train, val = folder_split(a.train_folder)
        name, label = "folders", f"folders {a.train_folder} -> {a.eval_folder}"
    elif a.data == "synthetic":
        train, val = synthetic(320, 0), damaged_set(synthetic(24, 11), 11)
        name, label = "synthetic", "synthetic contours (R1 seeds)"
    else:
        train, val_imgs, _ = natural()
        val, name, label = damaged_set(val_imgs, 5), "natural", "natural crops: camera / coffee / clock"
    noise = NOISE[name]
    print(f"{label}: choosing settings on {len(val)} validation images ...", flush=True)
    log = {}
    pri1, pick = choose(train, val, noise, log)

    # test images are loaded only now
    if a.train_folder:
        test = load_images(a.eval_folder, a.eval_limit or None)
        check_disjoint(load_images(a.train_folder), test)
        train = load_images(a.train_folder)
        pri1 = learn_priors(train, K=1, patch=10)
    elif name == "synthetic":
        test = synthetic(32, 73)
    else:
        test = natural()[2]
    cases = damaged_set(test, 73)
    print(f"testing on {len(cases)} images with {json.dumps(pick)}", flush=True)
    records, images = evaluate(train, cases, pri1, pick, noise)
    keys = [k for k in records[0] if k not in ("index", "mask_kind")]
    means = {k: float(np.mean([r[k] for r in records])) for k in keys}
    G = gates(means)

    out = Path(a.output or f"runs/gradients_{name}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"data": label, "settings": pick, "validation": {k: {str(i): v for i, v in d.items()} for k, d in log.items()},
               "means": means, "gates": G, "per_image": records, "minutes": round((time.time() - t0) / 60, 1)},
              open(out / "metrics.json", "w"), indent=1)
    vk = ["flat", "pred_mean", "pred_both", "oracle_mean", "oracle_gradient", "oracle_both", "oracle_tensor"]
    visuals.comparison(images, vk, ["no layout", "pred. mean", "pred. mean+grad", "TRUE mean", "TRUE grad", "TRUE mean+grad", "TRUE orient."],
                       out / "comparison.png")
    print("\nmean missing-pixel MSE (lower is better)")
    for k in ["flat", "pred_mean", "pred_gradient", "pred_both", "oracle_mean", "oracle_gradient", "oracle_both", "oracle_tensor"]:
        print(f"  {k:16s} {means[k]:.5f}")
    print("\ngates")
    for k, v in G.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  ({v['improvement']*100:+.1f}%)")
    print(f"\nwrote {out}/metrics.json, comparison.png")


if __name__ == "__main__":
    main()
