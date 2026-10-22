"""Behaviour mixed into :class:`~mixture_demo.models.base.EMGaussianMixture` subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mixture_demo.models.params import MixtureParams

if TYPE_CHECKING:
    from mixture_demo.models.base import FitResult
    from mixture_demo.priors import Prior


class MultiStartMixin:
    """Run EM from many random starts and keep the highest objective.

    That selection rule *is* the estimator, which is exactly how degenerate
    spikes win when the likelihood is unbounded.
    """

    n_components: int

    def random_init(self, x: np.ndarray, rng: np.random.Generator) -> MixtureParams:
        """Random starting point: means drawn from the data, sds a random fraction of ``std(x)``.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Data.
        rng : numpy.random.Generator
            Random source.

        Returns
        -------
        MixtureParams
        """
        K = self.n_components
        means = rng.choice(x, size=K, replace=False)
        sds = np.full(K, x.std()) * rng.uniform(0.2, 1.0, size=K)
        weights = rng.dirichlet(np.ones(K))
        return MixtureParams(weights, means, sds)

    def fit_best(self, x: np.ndarray, rng: np.random.Generator, n_starts: int = 30) -> FitResult:
        """Best of ``n_starts`` EM runs, by objective.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Data.
        rng : numpy.random.Generator
            Random source for the starting points.
        n_starts : int, default 30
            Number of random starts.

        Returns
        -------
        FitResult
            Highest-objective fit, with components sorted by mean.
        """
        best = None
        for _ in range(n_starts):
            res = self.fit(x, self.random_init(x, rng))  # type: ignore[attr-defined]
            if best is None or res.objective > best.objective:
                best = res
        return best.sorted()


class PriorMixin:
    """Attach a prior and add its log-density to the EM objective.

    Parameters
    ----------
    prior : Prior
        Prior over the mixture parameters.
    **kwargs
        Forwarded to the next class in the MRO.
    """

    def __init__(self, prior: Prior, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prior = prior

    def log_prior(self, params: MixtureParams) -> float:
        return self.prior.logpdf(params)
