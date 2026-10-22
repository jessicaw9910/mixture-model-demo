"""Experiment 3: degeneracy rate and parameter error vs sample size."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mixture_demo.config import MonteCarloConfig
from mixture_demo.experiments.base import Experiment, ExperimentResult

log = logging.getLogger(__name__)

METHODS = ("MLE", "MAP")


@dataclass(frozen=True)
class MonteCarloResult(ExperimentResult):
    """Per-sample-size summaries for each method.

    Attributes
    ----------
    sample_sizes : list of int
        Sample sizes swept.
    reps : int
        Datasets per sample size.
    degenerate_rate : dict of str to list of float
        Fraction of fits with a collapsed component, keyed by "MLE" / "MAP".
    median_rmse : dict of str to list of float
        Median RMSE of ``[weights, means, sds]`` against the truth.
    """

    sample_sizes: list[int]
    reps: int
    degenerate_rate: dict[str, list[float]]
    median_rmse: dict[str, list[float]]

    def summary(self) -> dict:
        return {
            "sample_sizes": self.sample_sizes,
            "reps": self.reps,
            "degenerate_rate": self.degenerate_rate,
            "median_rmse": self.median_rmse,
        }


class MonteCarloExperiment(Experiment[MonteCarloConfig, MonteCarloResult]):
    """Fit MLE and MAP to many simulated datasets at each sample size."""

    name = "monte-carlo"

    def run(self) -> MonteCarloResult:
        s = self.settings
        rng = np.random.default_rng(s.seed)
        true_vec = self.truth.as_vector()
        degen = {m: [] for m in METHODS}
        rmse = {m: [] for m in METHODS}
        for n in s.sample_sizes:
            collapsed = {m: [] for m in METHODS}
            errors = {m: [] for m in METHODS}
            for _ in range(s.reps):
                x = self.truth.sample(n, rng)
                for model in (self.models.mle(), self.models.map(x)):
                    params = model.fit_best(x, rng, n_starts=s.n_starts).params
                    collapsed[model.label].append(params.is_collapsed(self.collapse_sd))
                    errors[model.label].append(np.sqrt(np.mean((params.as_vector() - true_vec) ** 2)))
            for m in METHODS:
                degen[m].append(float(np.mean(collapsed[m])))
                rmse[m].append(float(np.median(errors[m])))
            log.info(
                "n=%5d  degenerate MLE=%.2f MAP=%.2f  median RMSE MLE=%.3f MAP=%.3f",
                n, degen["MLE"][-1], degen["MAP"][-1], rmse["MLE"][-1], rmse["MAP"][-1],
            )
        return MonteCarloResult(sample_sizes=list(s.sample_sizes), reps=s.reps,
                                degenerate_rate=degen, median_rmse=rmse)
