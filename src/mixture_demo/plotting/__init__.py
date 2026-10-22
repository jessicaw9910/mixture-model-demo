"""Rendering for each experiment result, kept separate from computation."""

from mixture_demo.plotting.base import FigureRenderer
from mixture_demo.plotting.figures import MonteCarloFigure, SingleDatasetFigure, SingularityFigure

__all__ = ["FigureRenderer", "MonteCarloFigure", "SingleDatasetFigure", "SingularityFigure"]
