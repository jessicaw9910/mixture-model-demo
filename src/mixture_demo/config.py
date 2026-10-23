"""Run configuration.

The dataclasses below are the schema. OmegaConf turns them into structured
configs, which gives type checking, rejection of unknown keys, and merging of
YAML files and ``key=value`` dot-list overrides for free.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from mixture_demo.models.params import MixtureParams

FIGURE_NAMES = ("plate-diagram", "singularity", "single-dataset", "monte-carlo")


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or invalid values in a config."""


@dataclass
class TruthConfig:
    """Ground-truth mixture that data are simulated from.

    Attributes
    ----------
    weights, means, sds : list of float
        Per-component mixture weights (summing to 1), means and standard deviations.
    """

    weights: list[float] = field(default_factory=lambda: [0.6, 0.4])
    means: list[float] = field(default_factory=lambda: [-2.0, 2.0])
    sds: list[float] = field(default_factory=lambda: [1.0, 0.7])

    def validate(self) -> None:
        """Raise :class:`ConfigError` unless this describes a valid mixture of >= 2 components."""
        if not (len(self.weights) == len(self.means) == len(self.sds)) or len(self.weights) < 2:
            raise ConfigError("truth.weights/means/sds must have the same length (>= 2)")
        if not np.isclose(sum(self.weights), 1.0) or min(self.weights) <= 0:
            raise ConfigError(f"truth.weights must be positive and sum to 1, got {self.weights}")
        if min(self.sds) <= 0:
            raise ConfigError(f"truth.sds must be positive, got {self.sds}")

    def to_params(self) -> MixtureParams:
        return MixtureParams(self.weights, self.means, self.sds)


@dataclass
class ModelConfig:
    """EM settings shared by MLE and MAP.

    Attributes
    ----------
    max_iter : int
        Maximum EM iterations per start.
    tol : float
        Stop when the objective changes by less than this.
    sd_floor : float
        Numerical floor on fitted sds; a fit sitting at ~this has collapsed.
    collapse_sd : float
        A fitted sd below this counts as a degenerate (collapsed) component.
    """

    max_iter: int = 500
    tol: float = 1e-9
    sd_floor: float = 1e-3
    collapse_sd: float = 0.05


@dataclass
class PriorConfig:
    """Hyper-parameters passed to :meth:`mixture_demo.models.priors.Prior.weak`.

    Attributes
    ----------
    alpha : float
        Symmetric Dirichlet concentration on the weights.
    a0 : float
        Inverse-Gamma shape on the variances.
    tau_scale : float
        Prior sd of the component means, in units of ``std(x)``.
    """

    alpha: float = 2.0
    a0: float = 2.0
    tau_scale: float = 3.0


@dataclass
class SingularityConfig:
    """Figure 1 settings.

    Attributes
    ----------
    n : int
        Sample size.
    seed : int
        Data seed.
    log10_sd_min : float
        Smallest sigma_2 on the sweep, as log10.
    n_grid : int
        Number of sigma_2 values on the sweep.
    """

    n: int = 40
    seed: int = 0
    log10_sd_min: float = -40.0
    n_grid: int = 800


@dataclass
class SingleDatasetConfig:
    """Figure 2 settings.

    Attributes
    ----------
    n : int
        Sample size.
    seed : int
        First data seed tried when searching for a collapsed MLE.
    max_search : int
        Number of consecutive seeds to try.
    n_starts : int
        Random EM starts per fit.
    bins : int
        Histogram bins.
    """

    n: int = 25
    seed: int = 0
    max_search: int = 500
    n_starts: int = 30
    bins: int = 15


@dataclass
class MonteCarloConfig:
    """Figure 3 settings.

    Attributes
    ----------
    sample_sizes : list of int
        Sample sizes to sweep.
    reps : int
        Simulated datasets per sample size.
    seed : int
        RNG seed for the whole sweep.
    n_starts : int
        Random EM starts per fit.
    """

    sample_sizes: list[int] = field(default_factory=lambda: [15, 25, 50, 100, 250, 1000])
    reps: int = 200
    seed: int = 1
    n_starts: int = 15


@dataclass
class OutputConfig:
    """Where and how figures are written.

    Attributes
    ----------
    images_dir : str
        Parent directory for timestamped run folders.
    dpi : int
        Figure resolution.
    """

    images_dir: str = "images"
    dpi: int = 200


@dataclass
class DemoConfig:
    """Complete, reproducible description of a demo run.

    Attributes
    ----------
    figures : list of str
        Which figures to produce; a subset of :data:`FIGURE_NAMES`.
    truth, model, prior, singularity, single_dataset, monte_carlo, output
        Section configs; see each class.
    """

    figures: list[str] = field(default_factory=lambda: list(FIGURE_NAMES))
    truth: TruthConfig = field(default_factory=TruthConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    singularity: SingularityConfig = field(default_factory=SingularityConfig)
    single_dataset: SingleDatasetConfig = field(default_factory=SingleDatasetConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        unknown = set(self.figures) - set(FIGURE_NAMES)
        if unknown:
            raise ConfigError(f"unknown figure(s) {sorted(unknown)}; choose from {list(FIGURE_NAMES)}")
        self.truth.validate()

    # -- OmegaConf bridge ----------------------------------------------------
    def to_omegaconf(self) -> DictConfig:
        return OmegaConf.structured(self)

    def to_dict(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.to_omegaconf())  # type: ignore[return-value]

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self.to_omegaconf())

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> DemoConfig:
        return _validated(lambda: OmegaConf.to_object(cfg))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DemoConfig:
        """Build a config from a (possibly partial) nested mapping; missing keys keep defaults."""
        return cls().merge(data)

    @classmethod
    def load(cls, path: str | Path) -> DemoConfig:
        """Load a YAML (or JSON, which is valid YAML) file on top of the defaults.

        Parameters
        ----------
        path : str or Path
            Config file, e.g. a previous run's ``config.yaml``.

        Raises
        ------
        ConfigError
            If the file has unknown keys or values of the wrong type.
        """
        return cls().merge(_validated(lambda: OmegaConf.load(path)))

    def dump(self, path: str | Path, header: str = "") -> Path:
        """Write the config as YAML, prefixed by ``header`` (e.g. comment lines). Returns ``path``."""
        path = Path(path)
        path.write_text(header + self.to_yaml())
        return path

    # -- overrides -------------------------------------------------------------
    def merge(self, *others: Mapping[str, Any] | DictConfig) -> DemoConfig:
        """Return a validated copy with ``others`` merged on top (later wins)."""
        return self.from_omegaconf(_validated(lambda: OmegaConf.merge(self.to_omegaconf(), *others)))

    def with_overrides(self, overrides: Mapping[str, Any] | Iterable[str]) -> DemoConfig:
        """Return a validated copy with dotted-key overrides applied.

        Parameters
        ----------
        overrides : mapping or iterable of str
            Either ``{"monte_carlo.reps": 50}`` or ``["monte_carlo.reps=50"]``.

        Raises
        ------
        ConfigError
            On malformed overrides, unknown keys or bad values.
        """
        if isinstance(overrides, Mapping):
            dotlist = [f"{k}={_dotlist_value(v)}" for k, v in overrides.items()]
        else:
            dotlist = list(overrides)
        bad = [item for item in dotlist if "=" not in item]
        if bad:
            raise ConfigError(f"overrides must look like key=value, got {bad}")
        return self.merge(_validated(lambda: OmegaConf.from_dotlist(dotlist)))


def _dotlist_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(map(str, value)) + "]"
    return str(value)


def _validated(fn):
    """Run an OmegaConf operation, re-raising its errors as :class:`ConfigError`."""
    try:
        return fn()
    except OmegaConfBaseException as exc:
        raise ConfigError(str(exc).splitlines()[0] if str(exc) else type(exc).__name__) from None
