"""Experiment base class and the model factory experiments share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from mixture_demo.config import ModelConfig, PriorConfig
from mixture_demo.models import MAPMixture, MixtureParams, MLEMixture
from mixture_demo.priors import Prior


@dataclass(frozen=True)
class ModelFactory:
    """Builds MLE / MAP estimators with consistent EM and prior settings.

    Parameters
    ----------
    n_components : int
        Number of mixture components to fit.
    model : ModelConfig
        EM settings.
    prior : PriorConfig
        Hyper-parameters for :meth:`Prior.weak <mixture_demo.priors.Prior.weak>`.
    """

    n_components: int
    model: ModelConfig
    prior: PriorConfig

    @property
    def _em_kwargs(self) -> dict[str, Any]:
        return dict(
            n_components=self.n_components,
            max_iter=self.model.max_iter,
            tol=self.model.tol,
            sd_floor=self.model.sd_floor,
        )

    def make_prior(self, x: np.ndarray, n_components: int | None = None) -> Prior:
        """Weak data-scaled prior for ``x``; ``n_components`` defaults to the factory's."""
        return Prior.weak(
            x,
            n_components=n_components or self.n_components,
            alpha=self.prior.alpha,
            a0=self.prior.a0,
            tau_scale=self.prior.tau_scale,
        )

    def mle(self) -> MLEMixture:
        return MLEMixture(**self._em_kwargs)

    def map(self, x: np.ndarray) -> MAPMixture:
        """The prior is data-scaled, so a MAP estimator is built per dataset."""
        return MAPMixture(prior=self.make_prior(x), **self._em_kwargs)


class ExperimentResult(ABC):
    """Everything a renderer needs to draw one figure."""

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """JSON-serialisable digest written to ``summary.json``."""


SettingsT = TypeVar("SettingsT")
ResultT = TypeVar("ResultT", bound=ExperimentResult)


class Experiment(ABC, Generic[SettingsT, ResultT]):
    """Pure computation: simulate data, fit models, return a result object.

    Parameters
    ----------
    settings : SettingsT
        This experiment's config section.
    truth : MixtureParams
        Ground-truth mixture to simulate from.
    models : ModelFactory
        Builds the MLE / MAP estimators.
    collapse_sd : float
        A fitted sd below this counts as a collapsed component.
    """

    name: ClassVar[str]

    def __init__(
        self,
        settings: SettingsT,
        truth: MixtureParams,
        models: ModelFactory,
        collapse_sd: float,
    ) -> None:
        self.settings = settings
        self.truth = truth
        self.models = models
        self.collapse_sd = collapse_sd

    @abstractmethod
    def run(self) -> ResultT:
        """Run the experiment and return its result."""
