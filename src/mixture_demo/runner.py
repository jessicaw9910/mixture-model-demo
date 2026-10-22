"""Run the configured experiments and write figures + a reproducible record to a run directory."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from mixture_demo.config import DemoConfig
from mixture_demo.experiments import (
    Experiment,
    ModelFactory,
    MonteCarloExperiment,
    SingleDatasetExperiment,
    SingularityExperiment,
)
from mixture_demo.plotting import FigureRenderer, MonteCarloFigure, SingleDatasetFigure, SingularityFigure

log = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"
SUMMARY_FILENAME = "summary.json"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


class DemoRunner:
    """Pairs each configured experiment with its renderer and records the run.

    Parameters
    ----------
    config : DemoConfig
        Full run configuration.

    Examples
    --------
    >>> from mixture_demo import DemoConfig, DemoRunner
    >>> cfg = DemoConfig().with_overrides(["figures=[singularity]"])
    >>> run_dir = DemoRunner(cfg).run()  # doctest: +SKIP
    """

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        truth = config.truth.to_params()
        self._common = dict(
            truth=truth,
            models=ModelFactory(truth.n_components, config.model, config.prior),
            collapse_sd=config.model.collapse_sd,
        )

    def build(self, name: str) -> tuple[Experiment, FigureRenderer]:
        """Return the ``(experiment, renderer)`` pair for a figure name in :data:`FIGURE_NAMES`."""
        c, dpi = self.config, self.config.output.dpi
        if name == "singularity":
            return SingularityExperiment(c.singularity, **self._common), SingularityFigure(dpi)
        if name == "single-dataset":
            return (SingleDatasetExperiment(c.single_dataset, **self._common),
                    SingleDatasetFigure(dpi, bins=c.single_dataset.bins))
        if name == "monte-carlo":
            return MonteCarloExperiment(c.monte_carlo, **self._common), MonteCarloFigure(dpi)
        raise KeyError(name)

    def make_run_dir(self, now: datetime | None = None) -> Path:
        """Create ``<images_dir>/<timestamp>/``, adding a ``_N`` suffix if it already exists."""
        base = Path(self.config.output.images_dir) / (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
        run_dir, i = base, 1
        while run_dir.exists():
            run_dir, i = base.with_name(f"{base.name}_{i}"), i + 1
        run_dir.mkdir(parents=True)
        return run_dir

    def run(self, run_dir: Path | None = None) -> Path:
        """Run every configured figure.

        Writes the figures, ``config.yaml`` (reusable with ``--config``) and
        ``summary.json`` (fitted values and timings).

        Parameters
        ----------
        run_dir : Path, optional
            Output directory. Defaults to a new timestamped folder under
            ``config.output.images_dir``.

        Returns
        -------
        Path
            The run directory.
        """
        from mixture_demo import __version__

        started = datetime.now()
        run_dir = Path(run_dir) if run_dir else self.make_run_dir(started)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Write the config first so the record exists even if a long run is interrupted.
        config_path = self.config.dump(
            run_dir / CONFIG_FILENAME,
            header=f"# mixture-demo {__version__} run config ({started.isoformat(timespec='seconds')})\n"
                   f"# Re-run with:  mixture-demo run --config {run_dir / CONFIG_FILENAME}\n",
        )
        log.info("writing to %s", run_dir)

        summary: dict = {"version": __version__, "started": started.isoformat(timespec="seconds"),
                         "config": config_path.name, "figures": {}}
        for name in self.config.figures:
            experiment, renderer = self.build(name)
            t0 = time.perf_counter()
            result = experiment.run()
            path = renderer.save(result, run_dir)
            elapsed = time.perf_counter() - t0
            log.info("%s -> %s (%.1fs)", name, path.name, elapsed)
            summary["figures"][name] = {"file": path.name, "seconds": round(elapsed, 2), "results": result.summary()}

        summary["finished"] = datetime.now().isoformat(timespec="seconds")
        (run_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, default=_to_json) + "\n")
        return run_dir


def _to_json(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
