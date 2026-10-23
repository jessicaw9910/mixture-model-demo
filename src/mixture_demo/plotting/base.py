"""Renderer base class and shared figure style."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Optional, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.figure import Figure

from mixture_demo.experiments.base import ExperimentResult

# Colour-blind-safe pair from viridis
C_MLE = colormaps["viridis"](0.15)
C_MAP = colormaps["viridis"](0.75)
C_TRUE = "0.25"
METHOD_COLORS = {"MLE": C_MLE, "MAP": C_MAP}

STYLE = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}

ResultT = TypeVar("ResultT", bound=Optional[ExperimentResult])


class FigureRenderer(ABC, Generic[ResultT]):
    """Turns an experiment result into a saved figure. No computation lives here.

    Parameters
    ----------
    dpi : int, default 200
        Resolution of saved raster images.
    """

    stem: ClassVar[str]  # output filename without extension
    style: ClassVar[dict[str, Any]] = {}  # rcParams layered on top of STYLE

    def __init__(self, dpi: int = 200) -> None:
        self.dpi = dpi

    @abstractmethod
    def draw(self, result: ResultT) -> Figure:
        """Build the figure for ``result`` (called inside the shared style context)."""

    def save(self, result: ResultT, out_dir: Path, formats: Sequence[str] = ("png",)) -> list[Path]:
        """Draw ``result`` and write it to ``out_dir / stem.<fmt>`` for each format.

        Parameters
        ----------
        result : ExperimentResult or None
            Output of the matching experiment (None for figures that need no computation).
        out_dir : Path
            Existing directory to write into.
        formats : sequence of str, default ("png",)
            File extensions understood by matplotlib, e.g. ``("png", "svg", "pdf")``.

        Returns
        -------
        list of Path
            The saved files, in the order of ``formats``.
        """
        paths = [Path(out_dir) / f"{self.stem}.{fmt}" for fmt in formats]
        with mpl.rc_context({**STYLE, **self.style}):
            fig = self.draw(result)
            for path in paths:
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)  # no-op for figures not created through pyplot
        return paths
