"""Experiment 1: log L vs log posterior as one component's sd -> 0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixture_demo.config import SingularityConfig
from mixture_demo.experiments.base import Experiment, ExperimentResult
from mixture_demo.models import MixtureParams

SD_MAX = 3.0
SD_REF = 1.0  # both curves are anchored to zero here


@dataclass(frozen=True)
class SingularityResult(ExperimentResult):
    """Objectives along a sigma_2 sweep.

    Attributes
    ----------
    sds : ndarray
        sigma_2 grid, increasing.
    log_likelihood, log_posterior : ndarray
        Objectives on the grid, relative to their values at sigma_2 = 1.
    anchor : float
        Data point that mu_2 is fixed on.
    """

    sds: np.ndarray
    log_likelihood: np.ndarray
    log_posterior: np.ndarray
    anchor: float

    def summary(self) -> dict:
        return {
            "anchor_point": self.anchor,
            "sigma_min": float(self.sds[0]),
            "delta_loglik_at_sigma_min": float(self.log_likelihood[0]),
            "delta_logpost_at_sigma_min": float(self.log_posterior[0]),
            "argmax_logpost_sigma": float(self.sds[np.argmax(self.log_posterior)]),
        }


class SingularityExperiment(Experiment[SingularityConfig, SingularityResult]):
    """Pin mu_2 on one data point and sweep sigma_2 -> 0: log L diverges, log posterior does not."""

    name = "singularity"

    def run(self) -> SingularityResult:
        s = self.settings
        x = self.truth.sample(s.n, np.random.default_rng(s.seed))
        prior = self.models.make_prior(x, n_components=2)
        # Component 1 is a broad Gaussian covering all the data, so every point keeps a
        # finite likelihood; component 2 sits exactly on one data point. Then the only
        # sigma_2-dependent term that matters as sigma_2 -> 0 is -log(sigma_2) -> +inf.
        anchor = x[np.argmin(np.abs(x - self.truth.means[-1]))]
        means = np.array([x.mean(), anchor])
        weights = np.array([0.5, 0.5])
        sds = np.logspace(s.log10_sd_min, np.log10(SD_MAX), s.n_grid)

        ll, lpost = np.empty_like(sds), np.empty_like(sds)
        for i, sd in enumerate(sds):
            params = MixtureParams(weights, means, np.array([x.std(), sd]))
            ll[i] = params.log_likelihood(x)
            lpost[i] = ll[i] + prior.logpdf(params)
        ref = np.argmin(np.abs(sds - SD_REF))
        return SingularityResult(sds=sds, log_likelihood=ll - ll[ref], log_posterior=lpost - lpost[ref],
                                 anchor=float(anchor))
