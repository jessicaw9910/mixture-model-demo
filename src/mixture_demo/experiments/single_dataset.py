"""Experiment 2: one small dataset, best-likelihood MLE fit vs MAP fit."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mixture_demo.config import SingleDatasetConfig
from mixture_demo.experiments.base import Experiment, ExperimentResult
from mixture_demo.models import FitResult, MixtureParams

log = logging.getLogger(__name__)

# Fits use RNG streams offset from the data seed so they are independent of it.
MLE_RNG_OFFSET = 10_000
MAP_RNG_OFFSET = 20_000


@dataclass(frozen=True)
class SingleDatasetResult(ExperimentResult):
    """MLE and MAP fits to one dataset.

    Attributes
    ----------
    seed : int
        Data seed used.
    data : ndarray
        The simulated dataset.
    truth : MixtureParams
        Mixture the data came from.
    mle, map : FitResult
        Best-of-``n_starts`` fits.
    n_starts : int
        Random EM starts per fit.
    found_collapse : bool
        False if no seed in the search produced a collapsed MLE.
    """

    seed: int
    data: np.ndarray
    truth: MixtureParams
    mle: FitResult
    map: FitResult
    n_starts: int
    found_collapse: bool

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "n": len(self.data),
            "found_collapsed_mle": self.found_collapse,
            "mle": self.mle.to_dict(),
            "map": self.map.to_dict(),
        }


class SingleDatasetExperiment(Experiment[SingleDatasetConfig, SingleDatasetResult]):
    """Find a small dataset whose best-likelihood MLE collapses, then fit MAP to the same data."""

    name = "single-dataset"

    def run(self) -> SingleDatasetResult:
        s = self.settings
        mle_model = self.models.mle()
        # Pick a seed where the best-likelihood MLE is degenerate (common at small n).
        for seed in range(s.seed, s.seed + s.max_search):
            x = self.truth.sample(s.n, np.random.default_rng(seed))
            mle = mle_model.fit_best(x, np.random.default_rng(seed + MLE_RNG_OFFSET), n_starts=s.n_starts)
            if mle.params.is_collapsed(self.collapse_sd):
                found = True
                break
        else:
            found = False
            log.warning("no collapsed MLE found in seeds %d..%d; plotting the last one", s.seed, seed)

        map_fit = self.models.map(x).fit_best(x, np.random.default_rng(seed + MAP_RNG_OFFSET), n_starts=s.n_starts)
        log.info("single-dataset seed=%d\n  MLE: %s\n  MAP: %s", seed, mle.params.to_dict(), map_fit.params.to_dict())
        return SingleDatasetResult(seed=seed, data=x, truth=self.truth, mle=mle, map=map_fit,
                                   n_starts=s.n_starts, found_collapse=found)
