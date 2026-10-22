"""Conjugate-style prior for a 1-D Gaussian mixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from mixture_demo.models.params import MixtureParams


@dataclass(frozen=True)
class Prior:
    """Dirichlet on weights, Normal on means, Inverse-Gamma on variances.

    Parameters
    ----------
    alpha : float
        Symmetric Dirichlet concentration on the weights.
    m0, tau : float
        Mean and sd of the Normal prior on each component mean.
    a0, b0 : float
        Shape and scale of the Inverse-Gamma prior on each variance.
    """

    alpha: float
    m0: float
    tau: float
    a0: float
    b0: float

    @classmethod
    def weak(
        cls,
        x: np.ndarray,
        n_components: int = 2,
        alpha: float = 2.0,
        a0: float = 2.0,
        tau_scale: float = 3.0,
    ) -> Prior:
        """Weak, data-scaled prior (Fraley & Raftery 2007).

        The prior mode of each variance is ~ ``var(x) / K**2`` and the shape is
        small, so the data easily override it.

        Parameters
        ----------
        x : ndarray
            Data used to set the location and scale.
        n_components : int, default 2
            Number of mixture components, K.
        alpha, a0 : float, default 2.0
            Dirichlet concentration and Inverse-Gamma shape.
        tau_scale : float, default 3.0
            Prior sd of the means, in units of ``std(x)``.

        Returns
        -------
        Prior
        """
        b0 = (a0 + 1) * x.var() / n_components**2
        return cls(alpha=alpha, m0=x.mean(), tau=tau_scale * x.std(), a0=a0, b0=b0)

    def logpdf(self, params: MixtureParams) -> float:
        """Unnormalised log prior density of ``params``."""
        var = params.sds**2
        lp = (self.alpha - 1) * np.log(params.weights).sum()
        lp += norm.logpdf(params.means, self.m0, self.tau).sum()
        lp += (-(self.a0 + 1) * np.log(var) - self.b0 / var).sum()
        return float(lp)
