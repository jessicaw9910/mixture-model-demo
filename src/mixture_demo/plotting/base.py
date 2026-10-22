"""Renderer base class and shared figure style."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import matplotlib as mpl
from matplotlib import colormaps
from matplotlib.figure import Figure

from mixture_demo.experiments.base import ExperimentResult

# Colour-blind-safe pair from viridis
C_MLE = colormaps["viridis"](0.15)
C_MAP = colormaps["viridis"](0.75)
C_TRUE = "0.25"
METHOD_COLORS = {"MLE": C_MLE, "MAP": C_MAP}

STYLE = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}

ResultT = TypeVar("ResultT", bound=ExperimentResult)


class FigureRenderer(ABC, Generic[ResultT]):
    """Turns an experiment result into a saved figure. No computation lives here.

    Parameters
    ----------
    dpi : int, default 200
        Resolution of the saved image.
    """

    filename: ClassVar[str]

    def __init__(self, dpi: int = 200) -> None:
        self.dpi = dpi

    @abstractmethod
    def draw(self, result: ResultT) -> Figure:
        """Build the figure for ``result`` (called inside the shared style context)."""

    def save(self, result: ResultT, out_dir: Path) -> Path:
        """Draw ``result`` and write it to ``out_dir / filename``.

        Parameters
        ----------
        result : ExperimentResult
            Output of the matching experiment.
        out_dir : Path
            Existing directory to write into.

        Returns
        -------
        Path
            The saved image.
        """
        path = Path(out_dir) / self.filename
        with mpl.rc_context(STYLE):
            fig = self.draw(result)
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        return path
