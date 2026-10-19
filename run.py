"""Run every arm on a test set and write metrics.json, comparison.png and loop.gif.

    python run.py --data synthetic                       # R1's exact 32 held-out contour images
    python run.py --data natural                         # 64 crops of three unseen photographs
    python run.py --train-folder train --eval-folder unseen --eval-limit 32   # your own images

Hyperparameters come from params_<data>.json (frozen by tune.py on validation).
Folder mode uses params_natural.json.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from data import synthetic, damaged_set, natural, load_images, check_disjoint
import bench
import visuals

GATES = [
    ("G1a layout talks down (linear medium)", "linear_medium", "linear_medium_flat", 0.05),
    ("G1b layout talks down (20-state medium)", "states_settle", "states_flat", 0.05),
    ("G2 switching local states pay", "states_settle", "linear_medium", 0.05),
]


def gates(means, n_hyp):
    out = {}
    for name, a, b, margin in GATES:
        gain = 1 - means[a] / means[b]
        out[name] = {"arm": a, "against": b, "improvement": round(gain, 4), "needed": margin,
                     "pass": bool(gain >= margin)}
    pb, sb = means[f"ping_best_of_{n_hyp}"], means[f"sham_best_of_{n_hyp}"]
    gain = 1 - pb / sb
    out["G3 ping beats sham"] = {"improvement": round(gain, 4), "needed": 0.02,
                                 "mean_distinct": round(means["ping_distinct"], 2),
                                 "pass": bool(gain >= 0.02 and means["ping_distinct"] >= 2)}
    strongest = min(("nearest4", "wiener"), key=lambda k: means[k])
    out["G4 declared model beats strongest simple method"] = {
        "arm": "linear_medium", "against": strongest,
        "improvement": round(1 - means["linear_medium"] / means[strongest], 4),
        "pass": bool(means["linear_medium"] < means[strongest])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "natural"], default="synthetic")
    ap.add_argument("--train-folder")
    ap.add_argument("--eval-folder")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--params")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.train_folder or a.eval_folder:
        if not (a.train_folder and a.eval_folder):
            raise SystemExit("give both --train-folder and --eval-folder")
        train = load_images(a.train_folder, a.train_limit or None)
        test = load_images(a.eval_folder, a.eval_limit or None)
        check_disjoint(train, test)
        cases, label, pfile = damaged_set(test, 73), f"folders {a.train_folder} -> {a.eval_folder}", "params_natural.json"
    elif a.data == "synthetic":
        train, test = synthetic(320, 0), synthetic(32, 73)
        cases, label, pfile = damaged_set(test, 73), "synthetic contours (R1 seeds)", "params_synthetic.json"
    else:
        train, _, test = natural()
        cases, label, pfile = damaged_set(test, 73), "natural crops: camera / coffee / clock", "params_natural.json"
    pfile = a.params or pfile
    params = json.load(open(pfile))["params"] if Path(pfile).exists() else {}
    if not params:
        print(f"note: {pfile} not found, using built-in defaults", flush=True)
    out = Path(a.output or f"runs/{'folders' if a.train_folder else a.data}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"{label}: {len(train)} training images, {len(cases)} test images", flush=True)
    M = bench.Models(train, params)
    records, sharps, images = bench.run(M, cases)
    means, sharp = bench.summarise(records, sharps)
    n_hyp = 2 * M.p["n_dirs"] + 1
    g = gates(means, n_hyp)

    report = {"data": label, "n_train": len(train), "n_test": len(cases), "params": M.p,
              "means": means, "median_sharpness": sharp, "gates": g, "per_image": records,
              "minutes": round((time.time() - t0) / 60, 1)}
    with open(out / "metrics.json", "w") as f:
        json.dump(report, f, indent=1)

    keys = ["linear", "nearest4", "wiener", "linear_medium", "states_settle", "states_ping", "states_posterior"]
    labels = ["diffusion", "4-nearest", "wiener", "linear medium", "20 states", "20 st.+ping", "posterior"]
    visuals.comparison(images, keys, labels, out / "comparison.png")
    pick = int(np.argmax([r["sham_best_of_7"] - r["ping_best_of_7"] if "ping_best_of_7" in r else 0 for r in records]))
    t, o, m, _ = cases[pick]
    visuals.loop_gif(M.model, t, o, m, out / "loop.gif", M.p["n_dirs"], M.p["size"])

    print("\nmean missing-pixel MSE (lower is better)")
    for k in keys + ["linear_medium_flat", "states_flat", "states_sham", "nearest1"]:
        print(f"  {k:20s} {means[k]:.5f}   sharpness {sharp[k]:.2f}")
    for k in (f"ping_best_of_{n_hyp}", f"sham_best_of_{n_hyp}", f"posterior_best_of_{n_hyp}",
              f"nearest_best_of_{n_hyp}", "ping_distinct", "sham_distinct"):
        print(f"  {k:20s} {means[k]:.5f}")
    print("\ngates")
    for k, v in g.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  ({v['improvement']*100:+.1f}%)")
    print(f"\nwrote {out}/metrics.json, comparison.png, loop.gif  ({report['minutes']} min)")


if __name__ == "__main__":
    main()
