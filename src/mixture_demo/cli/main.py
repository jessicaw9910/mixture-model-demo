"""``mixture-demo`` entry point (typer + OmegaConf).

Every parameter lives in :class:`~mixture_demo.config.DemoConfig`. A run starts
from the defaults, merges ``--config FILE`` on top, then any ``key=value``
overrides, so later sources win.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from mixture_demo.config import FIGURE_NAMES, ConfigError, DemoConfig

Figure = Enum("Figure", {name.replace("-", "_"): name for name in FIGURE_NAMES}, type=str)

app = typer.Typer(
    name="mixture-demo",
    help="MLE vs MAP for a 1-D Gaussian mixture: why the likelihood is unbounded.",
    epilog=(
        "Examples:\n\n"
        "  mixture-demo run\n\n"
        "  mixture-demo run -f singularity -f single-dataset single_dataset.n=40\n\n"
        "  mixture-demo run 'monte_carlo.sample_sizes=[20,50,200]' monte_carlo.reps=50\n\n"
        "  mixture-demo run --config images/<run>/config.yaml   # reproduce a run\n\n"
        "  mixture-demo show-config > my_config.yaml            # see every key"
    ),
    no_args_is_help=True,
    rich_markup_mode=None,
    pretty_exceptions_show_locals=False,
)

ConfigOpt = Annotated[
    Optional[Path],
    typer.Option("--config", "-c", exists=True, dir_okay=False,
                 help="YAML/JSON config to start from, e.g. a previous run's config.yaml."),
]
OverridesArg = Annotated[
    Optional[list[str]],
    typer.Argument(metavar="[KEY=VALUE]...", show_default=False,
                   help="Dot-list overrides, e.g. monte_carlo.reps=50 'truth.means=[-1,1]' (quote lists in zsh). "
                        "Run `mixture-demo show-config` to list every key."),
]
FiguresOpt = Annotated[
    Optional[list[Figure]],
    typer.Option("--figure", "-f", help="Figure to produce (repeatable). Default: all."),
]


def resolve_config(
    config: Path | None = None,
    overrides: list[str] | None = None,
    figures: list[Figure] | None = None,
) -> DemoConfig:
    """Merge defaults, then ``config``, then ``figures``, then dot-list ``overrides`` (later wins).

    Parameters
    ----------
    config : Path, optional
        YAML/JSON config file.
    overrides : list of str, optional
        ``key=value`` strings, e.g. ``["monte_carlo.reps=50"]``.
    figures : list of Figure, optional
        Restrict which figures are produced.

    Returns
    -------
    DemoConfig

    Raises
    ------
    typer.BadParameter
        If the resulting config is invalid.
    """
    try:
        cfg = DemoConfig.load(config) if config else DemoConfig()
        if figures:
            cfg = cfg.with_overrides({"figures": [f.value for f in figures]})
        return cfg.with_overrides(overrides or [])
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from None


def _print_version(value: bool) -> None:
    if value:
        from mixture_demo import __version__

        typer.echo(f"mixture-demo {__version__}")
        raise typer.Exit()


@app.callback()
def _setup(
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only log warnings.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_print_version, is_eager=True, help="Show version and exit.")
    ] = False,
) -> None:
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(message)s")


@app.command()
def run(
    overrides: OverridesArg = None,
    config: ConfigOpt = None,
    figures: FiguresOpt = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", file_okay=False,
                     help="Write into this exact directory instead of a new timestamped one."),
    ] = None,
) -> None:
    """Run the demo.

    Outputs go to <output.images_dir>/<timestamp>/: the figures, config.yaml
    (reusable with --config) and summary.json (fitted parameters, timings).
    """
    from mixture_demo.runner import DemoRunner

    cfg = resolve_config(config, overrides, figures)
    typer.echo(DemoRunner(cfg).run(output_dir))


@app.command("show-config")
def show_config(overrides: OverridesArg = None, config: ConfigOpt = None, figures: FiguresOpt = None) -> None:
    """Print the resolved config (what `run` would use with the same arguments) as YAML."""
    typer.echo(resolve_config(config, overrides, figures).to_yaml(), nl=False)


def main() -> None:
    app()
