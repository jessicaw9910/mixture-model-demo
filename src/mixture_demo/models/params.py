"""Parameter container for a K-component 1-D Gaussian mixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class MixtureParams:
    """Parameters of a 1-D Gaussian mixture.

    Parameters
    ----------
    weights : array_like, shape (K,)
        Mixing proportions, summing to 1.
    means : array_like, shape (K,)
        Component means.
    sds : array_like, shape (K,)
        Component standard deviations.
    """

    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray

    def __post_init__(self) -> None:
        for name in ("weights", "means", "sds"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.weights.ndim != 1 or not (self.weights.shape == self.means.shape == self.sds.shape):
            raise ValueError("weights, means and sds must be 1-D arrays of equal length")

    @property
    def n_components(self) -> int:
        return len(self.weights)

    # -- densities ----------------------------------------------------------
    def component_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Weighted per-component log-densities.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Data.

        Returns
        -------
        ndarray, shape (n, K)
            ``log(w_k) + log N(x_i | mu_k, sd_k)``.
        """
        return np.log(self.weights) + norm.logpdf(x[:, None], self.means, self.sds)

    def log_likelihood(self, x: np.ndarray) -> float:
        """Total mixture log-likelihood of ``x``."""
        return float(logsumexp(self.component_logpdf(x), axis=1).sum())

    def pdf(self, grid: np.ndarray) -> np.ndarray:
        """Mixture density evaluated on ``grid``."""
        return (self.weights * norm.pdf(grid[:, None], self.means, self.sds)).sum(1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points from the mixture.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator
            Random source.

        Returns
        -------
        ndarray, shape (n,)
        """
        z = rng.choice(self.n_components, size=n, p=self.weights)
        return rng.normal(self.means[z], self.sds[z])

    # -- utilities ----------------------------------------------------------
    def sorted(self) -> MixtureParams:
        """Copy with components ordered by mean (resolves label switching)."""
        order = np.argsort(self.means)
        return MixtureParams(self.weights[order], self.means[order], self.sds[order])

    def as_vector(self) -> np.ndarray:
        """Concatenate ``[weights, means, sds]`` into one array of length 3K."""
        return np.concatenate([self.weights, self.means, self.sds])

    def is_collapsed(self, threshold: float) -> bool:
        """True if any component's sd is below ``threshold``."""
        return bool(self.sds.min() < threshold)

    def to_dict(self) -> dict[str, list[float]]:
        return {"weights": self.weights.tolist(), "means": self.means.tolist(), "sds": self.sds.tolist()}
