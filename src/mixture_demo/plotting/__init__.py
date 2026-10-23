"""Rendering for each experiment result, kept separate from computation."""

from mixture_demo.plotting.base import FigureRenderer
from mixture_demo.plotting.figures import MonteCarloFigure, SingleDatasetFigure, SingularityFigure
from mixture_demo.plotting.plate import PlateDiagramFigure

__all__ = ["FigureRenderer", "MonteCarloFigure", "PlateDiagramFigure", "SingleDatasetFigure", "SingularityFigure"]
