import numpy as np
import pytest
from data import synthetic, damage, check_disjoint
from sihti4 import Sihti4, learn_priors, block_mean_matrix


@pytest.fixture(scope="module")
def priors():
    train = synthetic(60, 0)
    return learn_priors(train, K=4, patch=10, n_patches=4000), learn_priors(train, K=1, patch=10, n_patches=4000)


@pytest.fixture(scope="module")
def case():
    target = synthetic(1, 11)[0]
    obs, mask = damage(target, seed=3, kind="hole")
    return target, obs, mask


@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_settle_never_raises_energy(priors, case, temp):
    m = Sihti4.from_priors(priors[0], gamma=300., noise=.003, temp=temp)
    _, obs, mask = case
    prep = m.prepare(obs, mask)
    from sihti4 import gaussian_fill
    out = m.settle(prep, gaussian_fill(obs, mask), record=True)
    e = np.array(out["energies"] + [out["energy"]])
    assert np.all(np.diff(e) <= 1e-6 * np.abs(e[:-1]).max())


def test_global_step_is_exact_minimiser(priors, case):
    m = Sihti4.from_priors(priors[0], gamma=300., noise=.003)
    _, obs, mask = case
    prep = m.prepare(obs, mask)
    x0 = obs.copy()
    R = m.choose_states(prep, x0)
    A, b = m.system(prep, R)
    xh = np.linalg.solve(A, b)
    f = lambda v: m.energy(prep, m.compose(prep, v), R)
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.standard_normal(len(xh)) * 1e-3
        assert f(xh + d) >= f(xh) - 1e-9
        assert f(xh - d) >= f(xh) - 1e-9


def test_visible_pixels_kept_and_no_target_leak(priors, case):
    m = Sihti4.from_priors(priors[0], gamma=300., noise=.003)
    target, obs, mask = case
    r1 = m.infer(obs, mask, mode="ping", n_dirs=2)
    other = obs.copy()
    other[~mask] = .9          # different values where the target is hidden
    r2 = m.infer(other, mask, mode="ping", n_dirs=2)
    assert np.array_equal(r1["map"][mask], obs[mask])
    assert np.allclose(r1["map"], r2["map"]) and np.allclose(r1["bma"], r2["bma"])


def test_linear_medium_ping_is_blind(priors, case):
    """K = 1: one state, the medium is linear-Gaussian; every kick relaxes back."""
    m = Sihti4.from_priors(priors[1], gamma=300., noise=.003)
    _, obs, mask = case
    r = m.infer(obs, mask, mode="ping", n_dirs=3, size=4.)
    hyps = np.array(r["hypotheses"])
    assert np.abs(hyps - hyps[0]).max() < 1e-6


def test_ping_is_greens_function(priors, case):
    m = Sihti4.from_priors(priors[0], gamma=300., noise=.003)
    _, obs, mask = case
    prep = m.prepare(obs, mask)
    from sihti4 import gaussian_fill
    s = m.settle(prep, gaussian_fill(obs, mask))
    lam, U, G = m.ping_directions(prep, s["A"], 3)
    assert np.allclose(s["A"] @ G, prep["Dh"].T)
    assert np.all(lam > 0) and np.all(np.diff(lam) <= 0)


def test_block_means():
    D = block_mean_matrix(32, 4)
    x = np.arange(32 * 32, dtype=float)
    assert D.shape == (64, 1024) and np.allclose(D.sum(1), 1.)
    assert np.isclose(D[0] @ x, np.mean(x.reshape(32, 32)[:4, :4]))


def test_disjoint_check():
    a = synthetic(3, 0)
    with pytest.raises(ValueError):
        check_disjoint(a, a[:1])
