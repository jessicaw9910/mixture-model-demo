"""Shared EM machinery for 1-D Gaussian mixtures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from scipy.special import logsumexp

from mixture_demo.models.params import MixtureParams


@dataclass(frozen=True)
class FitResult:
    """Outcome of one EM run.

    Attributes
    ----------
    params : MixtureParams
        Fitted parameters.
    objective : float
        Final objective: log L for MLE, log L + log prior for MAP.
    n_iter : int
        Iterations used.
    converged : bool
        Whether the tolerance was reached before ``max_iter``.
    """

    params: MixtureParams
    objective: float
    n_iter: int
    converged: bool

    def sorted(self) -> FitResult:
        """Copy with components ordered by mean."""
        return replace(self, params=self.params.sorted())

    def to_dict(self) -> dict:
        return {
            **self.params.to_dict(),
            "objective": self.objective,
            "n_iter": self.n_iter,
            "converged": self.converged,
        }


class EMGaussianMixture(ABC):
    """Expectation-maximisation for a 1-D Gaussian mixture.

    The E-step, convergence loop and variance floor are shared; subclasses
    implement :meth:`m_step` and may override :meth:`log_prior`.

    Parameters
    ----------
    n_components : int, default 2
        Number of mixture components, K.
    max_iter : int, default 500
        Maximum EM iterations.
    tol : float, default 1e-9
        Stop when the objective changes by less than this.
    sd_floor : float, default 1e-3
        Numerical floor on fitted sds; a fit sitting at ~this has collapsed.
    """

    label: ClassVar[str] = "EM"

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 500,
        tol: float = 1e-9,
        sd_floor: float = 1e-3,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.sd_floor = sd_floor

    @staticmethod
    def responsibilities(x: np.ndarray, params: MixtureParams) -> np.ndarray:
        """E-step: posterior component memberships, shape (n, K)."""
        comp = params.component_logpdf(x)
        return np.exp(comp - logsumexp(comp, axis=1, keepdims=True))

    @staticmethod
    def effective_counts(resp: np.ndarray) -> np.ndarray:
        """Soft count of points per component, shape (K,)."""
        return resp.sum(0) + 1e-12

    @abstractmethod
    def m_step(
        self, x: np.ndarray, resp: np.ndarray, params: MixtureParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update the parameters given responsibilities.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Data.
        resp : ndarray, shape (n, K)
            Responsibilities from the E-step.
        params : MixtureParams
            Current parameters.

        Returns
        -------
        weights, means, variances : ndarray, shape (K,)
        """

    def log_prior(self, params: MixtureParams) -> float:
        """Log prior added to the objective; zero (flat) by default."""
        return 0.0

    def objective(self, x: np.ndarray, params: MixtureParams) -> float:
        """Quantity EM maximises: log-likelihood plus :meth:`log_prior`."""
        return params.log_likelihood(x) + self.log_prior(params)

    def fit(self, x: np.ndarray, init: MixtureParams) -> FitResult:
        """Run EM from one starting point.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Data.
        init : MixtureParams
            Starting parameters.

        Returns
        -------
        FitResult
        """
        params, obj_old = init, -np.inf
        converged = False
        for it in range(1, self.max_iter + 1):
            resp = self.responsibilities(x, params)
            weights, means, var = self.m_step(x, resp, params)
            params = MixtureParams(weights, means, np.maximum(np.sqrt(var), self.sd_floor))
            obj = self.objective(x, params)
            if abs(obj - obj_old) < self.tol:
                converged = True
                break
            obj_old = obj
        return FitResult(params=params, objective=obj, n_iter=it, converged=converged)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_components={self.n_components}, "
            f"max_iter={self.max_iter}, tol={self.tol}, sd_floor={self.sd_floor})"
        )
