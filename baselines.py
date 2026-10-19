"""Strong, simple competitors. Each sees only the visible pixels."""
import numpy as np
from sihti4 import gaussian_fill


class Baselines:
    def __init__(self, train, wiener_shrink=0.02):
        self.train = np.asarray(train, float)
        flat = self.train.reshape(len(self.train), -1)
        self.mu = flat.mean(0)
        cov = np.cov(flat, rowvar=False)
        cov += (wiener_shrink * np.trace(cov) / len(self.mu) + 1e-5) * np.eye(len(self.mu))
        self.cov = cov

    def neighbours(self, obs, mask, n):
        d = ((self.train[:, mask] - obs[mask])**2).mean(1)
        return np.argsort(d)[:n]

    def nearest(self, obs, mask, n=1):
        """Average of the n training images closest on the visible pixels."""
        fill = self.train[self.neighbours(obs, mask, n)].mean(0)
        return np.where(mask, obs, fill)

    def nearest_set(self, obs, mask, n):
        """n separate retrieval hypotheses (for best-of-n comparisons)."""
        return [np.where(mask, obs, self.train[i]) for i in self.neighbours(obs, mask, n)]

    def wiener(self, obs, mask):
        """Linear MMSE: Gaussian conditional mean of hidden pixels given visible."""
        m = mask.ravel()
        x = obs.ravel().copy()
        S_hv = self.cov[np.ix_(~m, m)]
        S_vv = self.cov[np.ix_(m, m)]
        x[~m] = self.mu[~m] + S_hv @ np.linalg.solve(S_vv, x[m] - self.mu[m])
        return x.reshape(obs.shape)

    def linear(self, obs, mask):
        return gaussian_fill(obs, mask)
