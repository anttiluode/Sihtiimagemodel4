"""Sihti Image Model 4 — one joint energy, local states, a global layout, and a ping.

The image x (32x32, visible pixels clamped) is scored by

    E(x, k) =  sum_p  [ 1/2 (Q P_p x - m_k)^T C_k^-1 (Q P_p x - m_k) + 1/2 log|C_k| - log pi_k ]   (fine)
             + gamma/2 (D x - mu)^T S^-1 (D x - mu)                                                     (coarse)

* P_p cuts the 6x6 patch p; Q removes its mean. So the fine term only sees the
  *residue* of each patch around its own mean (the local Sihti split).
* k_p in {1..K} is the discrete LOCAL STATE of patch p: which Gaussian
  component of a learned mixture (GMM) it currently belongs to. This choice is
  the nonlinearity of the medium. With K = 1 the whole model is linear-Gaussian.
* D takes 4x4 block means: the 8x8 coarse LAYOUT. (mu, S) is a Gaussian layout
  prior learned from training images. It couples distant blocks.

Inference ("settle") is exact block-coordinate descent on that ONE energy:
    local step : every patch picks its best state given the current image
    global step: given all states, the energy is quadratic in the hidden pixels
                 and is minimised exactly (one linear solve, coarse prior included)
The coarse prior therefore talks DOWN (it pulls hidden pixels toward a layout
consistent with the visible blocks) and the fine states talk UP (the layout the
solve settles on is whatever the patch states make cheap). E never increases.

The PING. A settled state is only a local minimum: the discrete states can be
stuck. The ping is a top-down probe. From the settled state we take the coarse
layout's posterior covariance given the medium, Cc = L A^-1 L^T (A = Hessian of
E in the hidden pixels at the current states, L = hidden->coarse map), and pick
its most uncertain directions w_j. The ping travels down through the medium
itself: the pixel kick is the medium's Green-function response A^-1 L^T w_j,
scaled to move the layout by s posterior standard deviations. Each patch then
re-chooses its state under the kicked image (a state flips only where the kick
pushes it across a boundary), and the system settles again. In a linear medium
(K = 1) every kick relaxes back to the same point: the ping is blind, exactly as
in SihtiBrain's passive tree. The set of settled states is a set of HYPOTHESES.

No transformer, labels, pretrained network, or hidden target pixels are used.
"""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture

SIDE = 32


# ---------------------------------------------------------------- utilities
def gaussian_fill(observation, mask, sigma=1.8, cycles=60):
    """Ordinary linear diffusion fill with visible pixels clamped (baseline/init)."""
    x = np.asarray(observation, float).copy()
    visible = np.asarray(mask, bool)
    x[~visible] = float(np.mean(x[visible])) if visible.any() else .5
    for _ in range(cycles):
        x[~visible] = gaussian_filter(x, sigma=sigma)[~visible]
        x[visible] = observation[visible]
    return x


def block_mean_matrix(side=SIDE, block=4):
    """D: (side/block)^2 x side^2 matrix of block averages."""
    g = side // block
    D = np.zeros((g * g, side * side))
    for i in range(g):
        for j in range(g):
            cell = np.zeros((side, side))
            cell[i*block:(i+1)*block, j*block:(j+1)*block] = 1. / block**2
            D[i*g + j] = cell.ravel()
    return D


def patch_index(side=SIDE, patch=6):
    """(n_patches, patch*patch) flat pixel indices of every valid patch, stride 1."""
    grid = np.arange(side * side).reshape(side, side)
    return sliding_window_view(grid, (patch, patch)).reshape(-1, patch * patch)


def centering(d):
    return np.eye(d) - np.ones((d, d)) / d


# ---------------------------------------------------------------- learning
def learn_priors(train, K=20, patch=6, block=4, n_patches=40000, seed=0):
    """Fit the raw priors once: a K-component GMM on mean-removed 6x6 patches,
    and the mean/covariance of the 8x8 block-mean layouts."""
    imgs = np.asarray(train, float)
    if imgs.ndim != 3 or imgs.shape[1:] != (SIDE, SIDE) or len(imgs) < 20:
        raise ValueError("need >= 20 grayscale 32x32 training images")
    rng = np.random.default_rng(seed)
    idx = patch_index(SIDE, patch)
    flat = imgs.reshape(len(imgs), -1)
    P = flat[rng.integers(0, len(imgs), n_patches)[:, None], idx[rng.integers(0, len(idx), n_patches)]]
    P = P - P.mean(1, keepdims=True)
    gmm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=1e-4,
                          max_iter=200, random_state=seed).fit(P)
    coarse = flat @ block_mean_matrix(SIDE, block).T
    return {"gmm_means": gmm.means_ - gmm.means_.mean(1, keepdims=True),
            "gmm_cov": gmm.covariances_, "gmm_weights": gmm.weights_,
            "coarse_mu": coarse.mean(0), "coarse_cov": np.cov(coarse, rowvar=False),
            "patch": patch, "block": block}


# ---------------------------------------------------------------- the model
@dataclass
class Sihti4:
    means: np.ndarray      # (K, d) zero-mean component means
    W: np.ndarray          # (K, d, d) Q C_k^-1 Q
    a: np.ndarray          # (K, d)    Q C_k^-1 m_k
    const: np.ndarray      # (K,)      1/2 log|C_k| - log pi_k (+ 1/2 m^T C^-1 m)
    prec: np.ndarray       # (K, d, d) C_k^-1 (for state choice)
    logc: np.ndarray       # (K,)      1/2 log|C_k| - log pi_k
    coarse_mu: np.ndarray  # (G,)
    coarse_prec: np.ndarray  # (G, G)
    patch: int = 6
    block: int = 4
    gamma: float = 1.0
    temp: float = 0.0

    # .......................................................... learning
    @classmethod
    def fit(cls, train, K=20, patch=10, block=4, gamma=300., noise=0.003, shrink=0.05,
            temp=0.0, n_patches=40000, seed=0):
        """Learn the patch mixture and the coarse layout prior from training images.

        noise is the variance added to every component covariance (C_k + noise*I):
        the precision with which a local state is held. Without it the tightest
        (flat) component wins every uncertain patch.
        """
        return cls.from_priors(learn_priors(train, K, patch, block, n_patches, seed),
                               gamma=gamma, noise=noise, shrink=shrink, temp=temp)

    @classmethod
    def from_priors(cls, pri, gamma=1.0, noise=0.01, shrink=0.05, temp=0.0):
        d = pri["patch"] ** 2
        Q = centering(d)
        cov = pri["gmm_cov"] + noise * np.eye(d)[None]
        means = pri["gmm_means"]
        prec = np.linalg.inv(cov)
        prec = (prec + prec.transpose(0, 2, 1)) / 2
        logc = .5 * np.linalg.slogdet(cov)[1] - np.log(pri["gmm_weights"])
        W = np.einsum("ij,kjl,lm->kim", Q, prec, Q)
        a = np.einsum("ij,kjl,kl->ki", Q, prec, means)
        const = logc + .5 * np.einsum("ki,kij,kj->k", means, prec, means)
        c = pri["coarse_cov"]
        c = c + (shrink * np.trace(c) / len(c) + 1e-4) * np.eye(len(c))
        return cls(means, W, a, const, prec, logc, pri["coarse_mu"], np.linalg.inv(c),
                   pri["patch"], pri["block"], gamma, temp)

    @property
    def Pm(self):
        if not hasattr(self, "_Pm"):
            self._Pm = np.einsum("kde,ke->kd", self.prec, self.means)
        return self._Pm

    @property
    def K(self):
        return len(self.means)

    # .......................................................... per-image setup
    def prepare(self, observation, mask):
        """Precompute which patches touch hidden pixels and the scatter bookkeeping."""
        obs = np.asarray(observation, float)
        mask = np.asarray(mask, bool)
        if obs.shape != (SIDE, SIDE) or mask.shape != (SIDE, SIDE):
            raise ValueError("32x32 observation and mask required")
        if mask.all() or not mask.any():
            raise ValueError("mask must hide some pixels and show some pixels")
        hidden = np.flatnonzero(~mask.ravel())
        H = len(hidden)
        hid_of = -np.ones(SIDE * SIDE, int)
        hid_of[hidden] = np.arange(H)
        idx = patch_index(SIDE, self.patch)
        active = np.flatnonzero((hid_of[idx] >= 0).any(1))
        pidx = idx[active]
        d = pidx.shape[1]
        # flat scatter indices for A (pairs of hidden pixels sharing a patch) and b
        pa, la, ga = [], [], []
        pb, lb, gb = [], [], []
        for p, row in enumerate(pidx):
            loc = np.flatnonzero(hid_of[row] >= 0)
            gid = hid_of[row][loc]
            li, lj = np.meshgrid(loc, loc, indexing="ij")
            gi, gj = np.meshgrid(gid, gid, indexing="ij")
            pa.append(np.full(li.size, p)); la.append((li * d + lj).ravel()); ga.append((gi * H + gj).ravel())
            pb.append(np.full(len(loc), p)); lb.append(loc); gb.append(gid)
        base = obs.ravel().copy()
        base[hidden] = 0.
        D = block_mean_matrix(SIDE, self.block)
        cat = np.concatenate
        return {"obs": obs, "mask": mask, "hidden": hidden, "pidx": pidx,
                "A_idx": (cat(pa), cat(la), cat(ga)), "b_idx": (cat(pb), cat(lb), cat(gb)),
                "cvis": base[pidx], "Dh": D[:, hidden], "Dv_xv": D @ base}

    def compose(self, prep, xh):
        x = prep["obs"].ravel().copy()
        x[prep["hidden"]] = xh
        return x.reshape(SIDE, SIDE)

    # .......................................................... the two steps
    def state_costs(self, prep, x):
        """e[p, k]: energy of patch p if it were in local state k."""
        v = x.ravel()[prep["pidx"]]
        r = v - v.mean(1, keepdims=True)
        K, d = self.means.shape
        Pr = (r @ self.prec.transpose(1, 0, 2).reshape(d, K * d)).reshape(len(r), K, d)
        quad = np.einsum("pkd,pd->pk", Pr, r)
        return .5 * quad - r @ self.Pm.T + self.const[None]

    def choose_states(self, prep, x):
        """Local step. temp = 0: each patch takes its single best state (hard,
        binary-like). temp > 0: each patch holds a graded state, a distribution
        r_pk proportional to exp(-e_pk / temp) (soft, like a graded branch)."""
        e = self.state_costs(prep, x)
        if self.temp <= 0:
            R = np.zeros_like(e)
            R[np.arange(len(e)), np.argmin(e, 1)] = 1.
            return R
        z = -(e - e.min(1, keepdims=True)) / self.temp
        R = np.exp(z)
        return R / R.sum(1, keepdims=True)

    def system(self, prep, R, gamma=None):
        """Global step's quadratic: A xh = b, exact given the local states R."""
        g = self.gamma if gamma is None else gamma
        H = len(prep["hidden"])
        K, d = self.a.shape
        Wp = R @ self.W.reshape(K, d * d)                    # state-dependent "conductances"
        ap = R @ self.a
        p, l, gflat = prep["A_idx"]
        A = np.bincount(gflat, Wp[p, l], minlength=H * H).reshape(H, H)
        u = ap - np.einsum("pde,pe->pd", Wp.reshape(-1, d, d), prep["cvis"])
        p, l, gid = prep["b_idx"]
        b = np.bincount(gid, u[p, l], minlength=H)
        if g > 0:
            Dh = prep["Dh"]
            A += g * Dh.T @ self.coarse_prec @ Dh
            b += g * Dh.T @ self.coarse_prec @ (self.coarse_mu - prep["Dv_xv"])
        A += 1e-8 * np.eye(H)
        return A, b

    def energy(self, prep, x, R, gamma=None):
        """Free energy F = sum_pk R e_pk + temp * sum R log R + coarse term (= E for hard states)."""
        g = self.gamma if gamma is None else gamma
        e = float(np.sum(R * self.state_costs(prep, x)))
        if self.temp > 0:
            e += self.temp * float(np.sum(R * np.log(np.maximum(R, 1e-300))))
        if g > 0:
            dc = prep["Dh"] @ x.ravel()[prep["hidden"]] + prep["Dv_xv"] - self.coarse_mu
            e += .5 * g * dc @ self.coarse_prec @ dc
        return float(e)

    # .......................................................... settle
    def settle(self, prep, x0, gamma=None, max_iter=40, tol=1e-3, record=False):
        """Alternate the local step and the exact global solve until the local
        states stop changing. Each step lowers the same (free) energy."""
        x = np.asarray(x0, float)
        R = self.choose_states(prep, x)
        frames, energies, change = [], [], []
        for it in range(max_iter):
            A, b = self.system(prep, R, gamma)
            x = self.compose(prep, np.linalg.solve(A, b))
            new = self.choose_states(prep, x)
            change.append(float(np.abs(new - R).max()))
            if record:
                frames.append((x.copy(), R.argmax(1)))
                energies.append(self.energy(prep, x, R, gamma))
            R = new
            if change[-1] < tol:
                break
        out = {"x": x, "R": R, "states": R.argmax(1), "energy": self.energy(prep, x, R, gamma),
               "iterations": it + 1, "change": change, "A": A}
        if record:
            out["frames"], out["energies"] = frames, energies
        return out

    # .......................................................... posterior
    def posterior(self, prep, x0, n_burn=8, n_samples=24, seed=0, gamma=None, heat=1.0):
        """Gibbs sampling over the joint (local states, hidden pixels).

        Each sweep: draw every patch's state k_p ~ exp(-e_pk), then draw the hidden
        pixels from the exact Gaussian they have given those states. The returned
        estimate averages the conditional means A^-1 b over sweeps (Rao-Blackwell),
        i.e. it approximates the posterior mean under the full nonlinear model.
        The per-sweep conditional means (one per sampled state configuration) are
        kept as hypotheses.

        heat divides the whole energy. Overlapping patches count every pixel about
        patch^2 times, so the raw energy is far too confident; heat ~ patch^2
        makes the sampled uncertainty roughly calibrated. heat does not move the
        minimum, only how widely the sampler explores around it.
        """
        rng = np.random.default_rng(seed)
        x = np.asarray(x0, float)
        means, draws = [], []
        for it in range(n_burn + n_samples):
            e = self.state_costs(prep, x)
            pr = np.exp(-(e - e.min(1, keepdims=True)) / heat)
            pr /= pr.sum(1, keepdims=True)
            k = (pr.cumsum(1) > rng.random((len(pr), 1))).argmax(1)
            R = np.zeros_like(pr)
            R[np.arange(len(k)), k] = 1.
            A, b = self.system(prep, R, gamma)
            L = np.linalg.cholesky(A)
            mean = np.linalg.solve(A, b)
            xh = mean + np.sqrt(heat) * np.linalg.solve(L.T, rng.standard_normal(len(b)))
            x = self.compose(prep, xh)
            if it >= n_burn:
                means.append(mean)
                draws.append(self.compose(prep, mean))
        return self.compose(prep, np.mean(means, 0)), draws

    # .......................................................... ping
    def ping_directions(self, prep, A, n_dirs=3):
        """Most uncertain coarse-layout directions of the settled medium."""
        L = prep["Dh"]
        G = np.linalg.solve(A, L.T)            # Green's functions: A^-1 L^T
        Cc = L @ G
        lam, U = np.linalg.eigh((Cc + Cc.T) / 2)
        order = np.argsort(lam)[::-1][:n_dirs]
        return lam[order], U[:, order], G

    def kicks(self, prep, settled, n_dirs=3, size=2.5, kind="ping", seed=0):
        """Pixel kicks for the hypothesis search.

        kind="ping"      : top-down along the most uncertain coarse directions,
                           propagated through the medium (A^-1 L^T w).
        kind="sham"      : random coarse direction through the same medium, same pixel norm.
        kind="pixel"     : random pixel noise in the hole, same pixel norm.
        """
        lam, U, G = self.ping_directions(prep, settled["A"], n_dirs)
        rng = np.random.default_rng(seed)
        out = []
        for j in range(len(lam)):
            ping = G @ U[:, j] * (size / np.sqrt(max(lam[j], 1e-12)))
            norm = np.linalg.norm(ping)
            if kind == "ping":
                d = ping
            elif kind == "sham":
                w = rng.standard_normal(len(U))
                d = G @ w
                d *= norm / max(np.linalg.norm(d), 1e-12)
            elif kind == "pixel":
                d = rng.standard_normal(len(ping))
                d *= norm / max(np.linalg.norm(d), 1e-12)
            else:
                raise ValueError(kind)
            out += [d, -d]
        return out

    def hypotheses(self, prep, settled, kind="ping", n_dirs=3, size=2.5, seed=0, gamma=None):
        """Settled states reached after each kick (plus the original)."""
        xh0 = settled["x"].ravel()[prep["hidden"]]
        found = [settled]
        for d in self.kicks(prep, settled, n_dirs, size, kind, seed):
            kicked = self.compose(prep, xh0 + d)
            found.append(self.settle(prep, kicked, gamma))
        return found

    # .......................................................... full inference
    def infer(self, observation, mask, mode="ping", n_dirs=3, size=2.5, tau=1.0,
              gamma=None, seed=0, record=False, heat=1.0):
        """mode: "settle" (greedy only), "ping", "sham", "pixel" or "posterior"."""
        prep = self.prepare(observation, mask)
        start = gaussian_fill(prep["obs"], prep["mask"])
        base = self.settle(prep, start, gamma, record=record)
        if mode == "posterior":
            est, draws = self.posterior(prep, base["x"], seed=seed, gamma=gamma, heat=heat)
            return {"map": est, "bma": est, "hypotheses": draws, "base": base, "prep": prep,
                    "energies": np.array([base["energy"]]), "distinct": len(draws)}
        if mode == "settle":
            hyps = [base]
        else:
            hyps = self.hypotheses(prep, base, mode, n_dirs, size, seed, gamma)
        E = np.array([h["energy"] for h in hyps])
        best = int(np.argmin(E))
        w = np.exp(-(E - E.min()) / tau)
        w /= w.sum()
        bma = np.tensordot(w, np.array([h["x"] for h in hyps]), axes=1)
        return {"map": hyps[best]["x"], "bma": bma, "hypotheses": [h["x"] for h in hyps],
                "energies": E, "weights": w, "base": base, "prep": prep,
                "distinct": len({h["states"].tobytes() for h in hyps})}
