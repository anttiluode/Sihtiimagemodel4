"""Benchmark: every arm on the same damaged images, hidden target used only for scoring."""
import numpy as np
from sihti4 import Sihti4, learn_priors
from baselines import Baselines

DEFAULTS = dict(K=20, patch=10, block=4, noise=0.003, gamma=300., shrink=0.05,
                temp=0.0, n_dirs=3, size=2.5, tau=1.0, linear_gamma=100., wiener_shrink=0.02)


def mse(x, target, mask):
    return float(np.mean((x[~mask] - target[~mask])**2))


def sharpness(x, target, mask):
    """Gradient energy in the hidden region (dilated by 1 px), relative to the truth.
    1.0 = as sharp as the real image; < 1 blurrier; > 1 noisier."""
    from scipy.ndimage import binary_dilation
    region = binary_dilation(~mask, iterations=1)
    g = lambda im: np.hypot(*np.gradient(im))[region]
    return float(np.sum(g(x)**2) / max(np.sum(g(target)**2), 1e-9))


class Models:
    """Fit everything once from the training images."""
    def __init__(self, train, params=None, verbose=True):
        p = dict(DEFAULTS, **(params or {}))
        self.p = p
        if verbose:
            print(f"fitting {p['K']}-state patch mixture ({p['patch']}x{p['patch']}) and layout prior ...", flush=True)
        pri = learn_priors(train, K=p["K"], patch=p["patch"], block=p["block"])
        pri1 = learn_priors(train, K=1, patch=p["patch"], block=p["block"])
        self.model = Sihti4.from_priors(pri, gamma=p["gamma"], noise=p["noise"],
                                        shrink=p["shrink"], temp=p["temp"])
        self.linear_medium = Sihti4.from_priors(pri1, gamma=p["linear_gamma"], noise=p["noise"],
                                                shrink=p["shrink"])
        self.base = Baselines(train, wiener_shrink=p["wiener_shrink"])


def evaluate_case(M, target, obs, mask):
    p, m, B = M.p, M.model, M.base
    n_hyp = 2 * p["n_dirs"] + 1
    out = {"linear": B.linear(obs, mask), "nearest1": B.nearest(obs, mask, 1),
           "nearest4": B.nearest(obs, mask, 4), "wiener": B.wiener(obs, mask)}
    extra = {}
    # linear medium (one patch state): with and without the coarse layout
    out["linear_medium"] = M.linear_medium.infer(obs, mask, mode="settle")["map"]
    out["linear_medium_flat"] = M.linear_medium.infer(obs, mask, mode="settle", gamma=0.)["map"]
    # K-state medium: without layout, settled, pinged, sham-pinged, posterior mean
    out["states_flat"] = m.infer(obs, mask, mode="settle", gamma=0.)["map"]
    ping = m.infer(obs, mask, mode="ping", n_dirs=p["n_dirs"], size=p["size"], seed=1)
    sham = m.infer(obs, mask, mode="sham", n_dirs=p["n_dirs"], size=p["size"], seed=1)
    post = m.infer(obs, mask, mode="posterior", seed=1)
    out["states_settle"] = ping["base"]["x"]
    out["states_ping"] = ping["map"]
    out["states_sham"] = sham["map"]
    out["states_posterior"] = post["map"]
    for name, r in (("ping", ping), ("sham", sham), ("posterior", post)):
        hyps = r["hypotheses"][:n_hyp]
        extra[f"{name}_best_of_{n_hyp}"] = min(mse(h, target, mask) for h in hyps)
        extra[f"{name}_distinct"] = len({np.round(h, 4).tobytes() for h in hyps})
    extra[f"nearest_best_of_{n_hyp}"] = min(mse(h, target, mask) for h in B.nearest_set(obs, mask, n_hyp))
    extra["ping_energy_drop"] = float(ping["base"]["energy"] - ping["energies"].min())
    extra["sham_energy_drop"] = float(sham["base"]["energy"] - sham["energies"].min())
    extra["settle_iterations"] = ping["base"]["iterations"]
    metrics = {k: mse(v, target, mask) for k, v in out.items()}
    metrics.update(extra)
    sharp = {k: sharpness(v, target, mask) for k, v in out.items()}
    return metrics, sharp, out, ping


def run(M, cases, log_every=8):
    records, sharps, images = [], [], []
    for i, (target, obs, mask, kind) in enumerate(cases):
        m, s, imgs, ping = evaluate_case(M, target, obs, mask)
        m.update(index=i, mask_kind=kind)
        records.append(m)
        sharps.append(s)
        images.append((target, obs, mask, imgs, ping))
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i+1}/{len(cases)} images", flush=True)
    return records, sharps, images


def summarise(records, sharps):
    keys = [k for k in records[0] if k not in ("index", "mask_kind")]
    means = {k: float(np.mean([r[k] for r in records])) for k in keys}
    sharp = {k: float(np.median([s[k] for s in sharps])) for k in sharps[0]}
    return means, sharp
