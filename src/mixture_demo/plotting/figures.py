"""One renderer per experiment."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from mixture_demo.experiments import MonteCarloResult, SingleDatasetResult, SingularityResult
from mixture_demo.plotting.base import C_MAP, C_MLE, C_TRUE, METHOD_COLORS, FigureRenderer


class SingularityFigure(FigureRenderer[SingularityResult]):
    """Fig. 1: log L and log posterior against sigma_2."""

    filename = "fig1_singularity.png"

    def draw(self, result: SingularityResult) -> Figure:
        sds, ll, lpost = result.sds, result.log_likelihood, result.log_posterior
        lo, hi = np.log10(sds[0]), np.log10(sds[-1])

        def at(frac: float) -> float:  # x position a fraction of the way along the log axis
            return 10 ** (lo + frac * (hi - lo))

        fig = Figure(figsize=(7.5, 4.4))
        ax = fig.subplots()
        ax.plot(sds, ll, color=C_MLE, lw=2.5, label="log-likelihood (MLE objective)")
        ax.plot(sds, lpost, color=C_MAP, lw=2.5, label="log-posterior (MAP objective)")
        ax.set_xscale("log")
        ax.set_ylim(-45, 75)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel(r"$\sigma_2$  ($\mu_2$ fixed on a single data point)")
        ax.set_ylabel(r"objective $-$ objective at $\sigma_2=1$")
        ax.set_title("Likelihood grows without bound as $\\sigma_2 \\to 0$; posterior does not")
        x_arrow = at(0.25)
        ax.annotate(r"$\log L \sim -\log\sigma_2 \to +\infty$", xy=(x_arrow, ll[sds.searchsorted(x_arrow)]),
                    xytext=(at(0.4), 60), arrowprops=dict(arrowstyle="->", color=C_MLE), color=C_MLE)
        ax.annotate(r"prior term $-b_0/\sigma_2^2 \to -\infty$", xy=(0.3, -40), xytext=(at(0.57), -35),
                    arrowprops=dict(arrowstyle="->", color=C_MAP), color=C_MAP)
        ax.legend(frameon=False, loc="lower left")
        fig.tight_layout()
        return fig


class SingleDatasetFigure(FigureRenderer[SingleDatasetResult]):
    """Fig. 2: MLE and MAP fits side by side over the data histogram.

    Parameters
    ----------
    dpi : int, default 200
        Resolution of the saved image.
    bins : int, default 15
        Histogram bins.
    """

    filename = "fig2_single_dataset.png"

    def __init__(self, dpi: int = 200, bins: int = 15) -> None:
        super().__init__(dpi)
        self.bins = bins

    def draw(self, result: SingleDatasetResult) -> Figure:
        x, truth = result.data, result.truth
        lo = min(x.min(), (truth.means - 4 * truth.sds).min())
        hi = max(x.max(), (truth.means + 4 * truth.sds).max())
        grid = np.linspace(lo, hi, 2000)
        truth_pdf = truth.pdf(grid)
        heights, _ = np.histogram(x, bins=self.bins, density=True)
        # Cap the y-axis near the data scale so a collapsed spike runs off the top.
        top = 1.3 * max(truth_pdf.max(), heights.max())

        fig = Figure(figsize=(11, 4.2))
        axes = fig.subplots(1, 2, sharey=True)
        panels = [(f"{fit_name} (best of {result.n_starts} EM starts)", fit, METHOD_COLORS[fit_name])
                  for fit_name, fit in (("MLE", result.mle), ("MAP", result.map))]
        for ax, (title, fit, color) in zip(axes, panels):
            ax.hist(x, bins=self.bins, density=True, color="0.85", edgecolor="white")
            ax.plot(grid, truth_pdf, color=C_TRUE, ls="--", lw=1.5, label="truth")
            ax.plot(grid, fit.params.pdf(grid), color=color, lw=2.5, label="fit")
            ax.plot(x, np.full_like(x, -0.025 * top), "|", color="0.3", ms=10)
            p = fit.params
            txt = "\n".join(f"$\\pi$={p.weights[k]:.2f}  $\\mu$={p.means[k]:+.2f}  $\\sigma$={p.sds[k]:.3g}"
                            for k in range(p.n_components))
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10, family="monospace")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylim(-0.05 * top, top)
            ax.legend(frameon=False, loc="upper right")
        axes[0].set_ylabel("density")
        if result.found_collapse:
            headline = "the highest-likelihood solution collapses to a near-zero-variance spike"
        else:
            headline = "no collapsed MLE found within the seed search"
        fig.suptitle(f"n = {len(x)}: {headline}", y=1.0)
        fig.tight_layout()
        return fig


class MonteCarloFigure(FigureRenderer[MonteCarloResult]):
    """Fig. 3: degeneracy rate and median RMSE against sample size."""

    filename = "fig3_monte_carlo.png"

    def draw(self, result: MonteCarloResult) -> Figure:
        ns = result.sample_sizes
        fig = Figure(figsize=(11, 4.2))
        axes = fig.subplots(1, 2)
        for method, color in METHOD_COLORS.items():
            axes[0].plot(ns, result.degenerate_rate[method], "o-", color=color, lw=2.5, label=method)
            axes[1].plot(ns, result.median_rmse[method], "o-", color=color, lw=2.5, label=method)
        axes[0].set_ylabel("fraction of fits with a collapsed component")
        axes[1].set_ylabel("median parameter RMSE")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("sample size n")
            ax.legend(frameon=False)
        axes[0].set_ylim(-0.02, 1.02)
        fig.suptitle(f"{result.reps} simulated datasets per n; prior influence fades as n grows", y=1.0)
        fig.tight_layout()
        return fig
