"""Computation for each figure, kept separate from plotting."""

from mixture_demo.experiments.base import Experiment, ExperimentResult, ModelFactory
from mixture_demo.experiments.monte_carlo import MonteCarloExperiment, MonteCarloResult
from mixture_demo.experiments.single_dataset import SingleDatasetExperiment, SingleDatasetResult
from mixture_demo.experiments.singularity import SingularityExperiment, SingularityResult

__all__ = [
    "Experiment",
    "ExperimentResult",
    "ModelFactory",
    "MonteCarloExperiment",
    "MonteCarloResult",
    "SingleDatasetExperiment",
    "SingleDatasetResult",
    "SingularityExperiment",
    "SingularityResult",
]
