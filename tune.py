"""Choose hyperparameters on VALIDATION images only, then freeze them to a JSON file.

    python tune.py --data synthetic --out params_synthetic.json
    python tune.py --data natural   --out params_natural.json
    python tune.py --train-folder train                    # -> params_folders.json

Folder mode holds out a slice of YOUR train folder (default 15%, at most 32
images) as validation. The eval folder is never opened here.

Selection rule (fixed in docs/PREREG.md): every grid point is scored by mean
missing-pixel MSE on the validation images; the lowest wins. Test images are
never loaded here.
"""
import argparse
import json
import time
import numpy as np
from data import synthetic, damaged_set, natural, load_images
from sihti4 import Sihti4, learn_priors
from baselines import Baselines
from bench import mse, DEFAULTS

GRIDS = {
    # first-pass grid; the Wiener shrinkage was extended once because 0.1 sat on the edge
    "synthetic": {"linear_gamma": [30., 100., 300., 1000.],
                  "gamma": [100., 300., 1000., 3000.],
                  "noise": [0.003, 0.01],
                  "size": [2.5, 5.0],
                  "wiener_shrink": [0.005, 0.02, 0.1, 0.3, 1.0, 3.0, 10.0]},
    # natural photographs wanted much weaker layout and sharper patch states; the grids
    # were extended downward twice (validation only) until the optimum stopped moving by >1%
    "natural": {"linear_gamma": [0.3, 1., 3., 10., 30., 100., 300., 1000.],
                "gamma": [3., 10., 30., 100., 300., 1000., 3000.],
                "noise": [0.0003, 0.001, 0.003, 0.01],
                "size": [2.5, 5.0],
                "wiener_shrink": [0.005, 0.02, 0.1, 0.3, 1.0, 3.0, 10.0]},
    # your own folders: the natural-image range, trimmed of the settings that never won there
    "folders": {"linear_gamma": [0.3, 1., 3., 10., 30., 100., 300.],
                "gamma": [3., 10., 30., 100., 300.],
                "noise": [0.0003, 0.001, 0.003, 0.01],
                "size": [2.5, 5.0],
                "wiener_shrink": [0.005, 0.02, 0.1, 0.3, 1.0, 3.0]},
}


def folder_split(folder, val_fraction=0.15, val_limit=32, train_limit=0, seed=0):
    """Shuffle the train folder once (fixed seed) and hold out a validation slice."""
    imgs = load_images(folder, train_limit or None)
    order = np.random.default_rng(seed).permutation(len(imgs))
    n_val = int(min(max(4, round(val_fraction * len(imgs))), val_limit))
    if len(imgs) - n_val < 20:
        raise SystemExit("need at least 20 training images left after the validation slice")
    return imgs[order[n_val:]], damaged_set(imgs[order[:n_val]], 5)


def splits(name):
    if name == "synthetic":
        return synthetic(320, 0), damaged_set(synthetic(24, 11), 11)
    train, val, _ = natural()
    return train, damaged_set(val, 5)


def score(model, cases, **kw):
    return float(np.mean([mse(model.infer(o, m, **kw)["map"], t, m) for t, o, m, _ in cases]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "natural"], default="synthetic")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-folder", help="tune on a held-out slice of this folder")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--val-limit", type=int, default=32)
    ap.add_argument("--train-limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if a.train_folder:
        train, val = folder_split(a.train_folder, a.val_fraction, a.val_limit, a.train_limit)
        a.data = "folders"
        print(f"{a.train_folder}: fitting on {len(train)}, validating on {len(val)} held-out images", flush=True)
    else:
        train, val = splits(a.data)
    GRID = GRIDS[a.data]
    p = dict(DEFAULTS)
    log = {}
    pri = learn_priors(train, K=p["K"], patch=p["patch"], block=p["block"])
    pri1 = learn_priors(train, K=1, patch=p["patch"], block=p["block"])

    log["linear_medium"] = {g: score(Sihti4.from_priors(pri1, gamma=g, noise=p["noise"]), val, mode="settle")
                            for g in GRID["linear_gamma"]}
    p["linear_gamma"] = min(log["linear_medium"], key=log["linear_medium"].get)
    print("linear medium", log["linear_medium"], flush=True)

    log["settle"] = {}
    for g in GRID["gamma"]:
        for n in GRID["noise"]:
            log["settle"][f"{g}|{n}"] = score(Sihti4.from_priors(pri, gamma=g, noise=n), val, mode="settle")
            print("settle", g, n, log["settle"][f"{g}|{n}"], flush=True)
    best = min(log["settle"], key=log["settle"].get)
    p["gamma"], p["noise"] = map(float, best.split("|"))

    model = Sihti4.from_priors(pri, gamma=p["gamma"], noise=p["noise"])
    log["ping_best_of"] = {}
    for s in GRID["size"]:
        vals = []
        for t, o, m, _ in val:
            r = model.infer(o, m, mode="ping", n_dirs=p["n_dirs"], size=s)
            vals.append(min(mse(h, t, m) for h in r["hypotheses"]))
        log["ping_best_of"][s] = float(np.mean(vals))
        print("ping size", s, log["ping_best_of"][s], flush=True)
    p["size"] = min(log["ping_best_of"], key=log["ping_best_of"].get)

    log["wiener"] = {}
    for sh in GRID["wiener_shrink"]:
        B = Baselines(train, wiener_shrink=sh)
        log["wiener"][sh] = float(np.mean([mse(B.wiener(o, m), t, m) for t, o, m, _ in val]))
    p["wiener_shrink"] = min(log["wiener"], key=log["wiener"].get)
    print("wiener", log["wiener"], flush=True)

    out = {"data": a.data if a.data != "folders" else f"folder {a.train_folder}", "params": p, "validation_log": {k: {str(i): v for i, v in d.items()} for k, d in log.items()},
           "minutes": round((time.time() - t0) / 60, 1)}
    path = a.out or f"params_{a.data}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("frozen ->", path, json.dumps(p))


if __name__ == "__main__":
    main()
