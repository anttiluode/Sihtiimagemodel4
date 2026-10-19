"""Oracle ceilings: how much could a PERFECT top-down message of a given kind help?

The true value of a coarse read-out, computed from the hidden target, is handed
to the medium as a stiff extra term. If even the truth of a read-out cannot fix
the hole, no prediction of that read-out can. This is a diagnostic: it uses the
target, so it is never a competitor, only a ceiling.

    linear read-outs  (mean / gradient / both): lam/2 || C_h x_h - (C x_true - C_v x_v) ||^2
    orientation       (structure tensor per 4x4 block, nonlinear in x):
        T_b(x) = (mean gx^2, mean gy^2, mean gx*gy)  over the block
        lam/2 sum_b |T_b(x) - T_b(x_true)|^2   (xy counted twice), minimised by L-BFGS
"""
import numpy as np
from scipy.optimize import minimize
from sihti4 import coarse_matrix, block_mean_matrix, difference_matrices, gaussian_fill, SIDE

_AX, _AY = difference_matrices()
_D4 = block_mean_matrix(SIDE, 4)


def linear_oracle(model, observation, mask, target, kind, lam=1e4, gamma=0.):
    """Settle the medium with the TRUE read-out of `kind` pinned in (layout prior off)."""
    prep = model.prepare(observation, mask)
    C = coarse_matrix(kind, SIDE, model.block)
    hidden = prep["hidden"]
    base = prep["obs"].ravel().copy()
    base[hidden] = 0.
    Ch = C[:, hidden]
    resid = C @ target.ravel() - C @ base
    prep["extra"] = (lam * Ch.T @ Ch, lam * Ch.T @ resid)
    return model.settle(prep, gaussian_fill(observation, mask), gamma=gamma)["x"]


def tensor(x):
    gx, gy = _AX @ x.ravel(), _AY @ x.ravel()
    return _D4 @ (gx * gx), _D4 @ (gy * gy), _D4 @ (gx * gy), gx, gy


def tensor_oracle(model, observation, mask, target, lam=3e4, x0=None):
    """Linear medium (no layout prior) + the TRUE per-block structure tensor, by L-BFGS."""
    prep = model.prepare(observation, mask)
    R = model.choose_states(prep, observation)
    A, b = model.system(prep, R, gamma=0.)
    hidden = prep["hidden"]
    txx, tyy, txy, _, _ = tensor(target)
    if x0 is None:
        x0 = model.compose(prep, np.linalg.solve(A, b))

    def f(xh):
        x = model.compose(prep, xh).ravel()
        sxx, syy, sxy, gx, gy = tensor(x)
        rxx, ryy, rxy = sxx - txx, syy - tyy, sxy - txy
        e = .5 * xh @ A @ xh - b @ xh + .5 * lam * (rxx @ rxx + ryy @ ryy + 2 * rxy @ rxy)
        dgx = lam * (2 * (_D4.T @ rxx) * gx + 2 * (_D4.T @ rxy) * gy)
        dgy = lam * (2 * (_D4.T @ ryy) * gy + 2 * (_D4.T @ rxy) * gx)
        grad = A @ xh - b + (_AX.T @ dgx + _AY.T @ dgy)[hidden]
        return e, grad

    res = minimize(f, x0.ravel()[hidden], jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return model.compose(prep, res.x)
