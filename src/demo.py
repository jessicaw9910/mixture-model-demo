"""
MLE vs MAP for a 1-D two-component Gaussian mixture.

Teaching point
--------------
The mixture log-likelihood is *unbounded*: put one component's mean on a single
data point and let its variance shrink to zero, and log L -> +inf
(Kiefer & Wolfowitz 1956; Day 1969). So the "maximum likelihood estimate" of an
unconstrained heteroscedastic GMM does not exist; what EM returns is a local
optimum, and the higher-likelihood answers are often degenerate spikes.

MAP with a weak conjugate-style prior (Dirichlet on weights, Normal on means,
Inverse-Gamma on variances) makes the objective bounded and gives the same
EM algorithm with one extra pseudo-count term in each M-step
(Fraley & Raftery 2007, J. Classification 24:155-181).

Outputs (in ./figures):
  fig1_singularity.png     log L vs log posterior as sigma_1 -> 0
  fig2_single_dataset.png  one small dataset: best-likelihood MLE fit vs MAP fit
  fig3_monte_carlo.png     degeneracy rate and parameter error vs sample size
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Colour-blind-safe pair from viridis
C_MLE = plt.cm.viridis(0.15)
C_MAP = plt.cm.viridis(0.75)
C_TRUE = "0.25"

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------
# Ground truth: asymmetric bimodal mixture
# ---------------------------------------------------------------------------
TRUE_W = np.array([0.6, 0.4])
TRUE_MU = np.array([-2.0, 2.0])
TRUE_SD = np.array([1.0, 0.7])
SD_FLOOR = 1e-3  # numerical floor; a fit hitting ~this is "collapsed"
COLLAPSE_SD = 0.05  # any sd below this (true min is 0.7) counts as degenerate


def sample(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.choice(2, size=n, p=TRUE_W)
    return rng.normal(TRUE_MU[z], TRUE_SD[z])


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prior:
    alpha: float  # symmetric Dirichlet concentration on weights
    m0: float  # prior mean for component means
    tau: float  # prior sd for component means (weak)
    a0: float  # Inverse-Gamma shape on variances
    b0: float  # Inverse-Gamma scale on variances

    @classmethod
    def weak(cls, x: np.ndarray, K: int = 2) -> "Prior":
        # Fraley & Raftery-style data-scaled defaults: prior mode of each
        # variance ~ var(x)/K^2, shape small so it is easily overridden by data.
        a0 = 2.0
        b0 = (a0 + 1) * x.var() / K**2
        return cls(alpha=2.0, m0=x.mean(), tau=3 * x.std(), a0=a0, b0=b0)

    def logpdf(self, w, mu, sd) -> float:
        var = sd**2
        lp = (self.alpha - 1) * np.log(w).sum()
        lp += norm.logpdf(mu, self.m0, self.tau).sum()
        lp += (-(self.a0 + 1) * np.log(var) - self.b0 / var).sum()
        return float(lp)


# ---------------------------------------------------------------------------
# EM / MAP-EM
# ---------------------------------------------------------------------------
def loglik(x, w, mu, sd) -> float:
    comp = np.log(w) + norm.logpdf(x[:, None], mu, sd)
    return float(logsumexp(comp, axis=1).sum())


def em(x, w, mu, sd, prior: Prior | None = None, iters: int = 500, tol: float = 1e-9):
    """EM for MLE (prior=None) or MAP (prior given). Returns (w, mu, sd, objective)."""
    n, K = len(x), len(w)
    obj_old = -np.inf
    for _ in range(iters):
        # E-step
        comp = np.log(w) + norm.logpdf(x[:, None], mu, sd)
        lse = logsumexp(comp, axis=1, keepdims=True)
        r = np.exp(comp - lse)
        Nk = r.sum(0) + 1e-12

        # M-step
        if prior is None:
            w = Nk / n
            mu = (r * x[:, None]).sum(0) / Nk
            S = (r * (x[:, None] - mu) ** 2).sum(0)
            var = S / Nk
        else:
            w = (Nk + prior.alpha - 1) / (n + K * (prior.alpha - 1))
            var_old = sd**2
            mu = ((r * x[:, None]).sum(0) / var_old + prior.m0 / prior.tau**2) / (
                Nk / var_old + 1 / prior.tau**2
            )
            S = (r * (x[:, None] - mu) ** 2).sum(0)
            # mode of Inverse-Gamma(a0 + Nk/2, b0 + S/2)
            var = (2 * prior.b0 + S) / (2 * prior.a0 + Nk + 2)
        sd = np.maximum(np.sqrt(var), SD_FLOOR)

        obj = loglik(x, w, mu, sd) + (prior.logpdf(w, mu, sd) if prior else 0.0)
        if abs(obj - obj_old) < tol:
            break
        obj_old = obj
    return w, mu, sd, obj


def random_init(x, rng, K=2):
    mu = rng.choice(x, size=K, replace=False)
    sd = np.full(K, x.std()) * rng.uniform(0.2, 1.0, size=K)
    w = rng.dirichlet(np.ones(K))
    return w, mu, sd


def fit_best(x, rng, prior=None, n_starts=30):
    """Run EM from many random starts; keep the highest objective (that IS the estimator)."""
    best = None
    for _ in range(n_starts):
        res = em(x, *random_init(x, rng), prior=prior)
        if best is None or res[3] > best[3]:
            best = res
    w, mu, sd, obj = best
    order = np.argsort(mu)  # resolve label switching by ordering means
    return w[order], mu[order], sd[order], obj


def mixture_pdf(grid, w, mu, sd):
    return (w * norm.pdf(grid[:, None], mu, sd)).sum(1)


# ---------------------------------------------------------------------------
# Figure 1: the singularity
# ---------------------------------------------------------------------------
def fig_singularity(rng):
    x = sample(40, rng)
    prior = Prior.weak(x)
    # Component 1 is a broad Gaussian covering all the data, so every point keeps a
    # finite likelihood; component 2 sits exactly on one data point. Then the only
    # sigma_2-dependent term that matters as sigma_2 -> 0 is -log(sigma_2) -> +inf.
    anchor = x[np.argmin(np.abs(x - TRUE_MU[1]))]
    mu = np.array([x.mean(), anchor])
    w = np.array([0.5, 0.5])
    sds = np.logspace(-40, np.log10(3), 800)
    ll, lpost = [], []
    for s in sds:
        sd = np.array([x.std(), s])
        l = loglik(x, w, mu, sd)
        ll.append(l)
        lpost.append(l + prior.logpdf(w, mu, sd))
    ll, lpost = np.array(ll), np.array(lpost)
    ref = np.argmin(np.abs(sds - 1.0))  # anchor both curves at sigma_2 = 1

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(sds, ll - ll[ref], color=C_MLE, lw=2.5, label="log-likelihood (MLE objective)")
    ax.plot(sds, lpost - lpost[ref], color=C_MAP, lw=2.5, label="log-posterior (MAP objective)")
    ax.set_xscale("log")
    ax.set_ylim(-45, 75)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\sigma_2$  ($\mu_2$ fixed on a single data point)")
    ax.set_ylabel(r"objective $-$ objective at $\sigma_2=1$")
    ax.set_title("Likelihood grows without bound as $\\sigma_2 \\to 0$; posterior does not")
    ax.annotate(r"$\log L \sim -\log\sigma_2 \to +\infty$", xy=(1e-30, ll[sds.searchsorted(1e-30)] - ll[ref]),
                xytext=(1e-24, 60), arrowprops=dict(arrowstyle="->", color=C_MLE), color=C_MLE)
    ax.annotate(r"prior term $-b_0/\sigma_2^2 \to -\infty$", xy=(0.3, -40), xytext=(1e-17, -35),
                arrowprops=dict(arrowstyle="->", color=C_MAP), color=C_MAP)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_singularity.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: one small dataset
# ---------------------------------------------------------------------------
def fig_single_dataset(seed=None):
    # Pick a seed where the best-likelihood MLE is degenerate (common at small n).
    for s in range(seed or 0, (seed or 0) + 500):
        rng = np.random.default_rng(s)
        x = sample(25, rng)
        mle = fit_best(x, np.random.default_rng(s + 10_000))
        if mle[2].min() < COLLAPSE_SD:
            break
    prior = Prior.weak(x)
    map_ = fit_best(x, np.random.default_rng(s + 20_000), prior=prior)

    grid = np.linspace(-6, 6, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, fit, c) in zip(axes, [("MLE (best of 30 EM starts)", mle, C_MLE),
                                          ("MAP (best of 30 EM starts)", map_, C_MAP)]):
        ax.hist(x, bins=15, density=True, color="0.85", edgecolor="white")
        ax.plot(grid, mixture_pdf(grid, TRUE_W, TRUE_MU, TRUE_SD), color=C_TRUE, ls="--", lw=1.5, label="truth")
        ax.plot(grid, mixture_pdf(grid, *fit[:3]), color=c, lw=2.5, label="fit")
        ax.plot(x, np.full_like(x, -0.015), "|", color="0.3", ms=10)
        w, mu, sd, _ = fit
        txt = "\n".join(f"$\\pi$={w[k]:.2f}  $\\mu$={mu[k]:+.2f}  $\\sigma$={sd[k]:.3g}" for k in range(2))
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10, family="monospace")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylim(-0.03, 0.6)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("density")
    fig.suptitle(f"n = {len(x)}: the highest-likelihood solution collapses to a near-zero-variance spike", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_single_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return s, mle, map_


# ---------------------------------------------------------------------------
# Figure 3: Monte Carlo over sample size
# ---------------------------------------------------------------------------
def fig_monte_carlo(ns=(15, 25, 50, 100, 250, 1000), reps=200, seed=1):
    rng = np.random.default_rng(seed)
    true = np.concatenate([TRUE_W, TRUE_MU, TRUE_SD])
    res = {m: {"degen": [], "err": []} for m in ("MLE", "MAP")}
    for n in ns:
        d = {"MLE": [], "MAP": []}
        e = {"MLE": [], "MAP": []}
        for _ in range(reps):
            x = sample(n, rng)
            for m, prior in (("MLE", None), ("MAP", Prior.weak(x))):
                w, mu, sd, _ = fit_best(x, rng, prior=prior, n_starts=15)
                d[m].append(sd.min() < COLLAPSE_SD)
                e[m].append(np.sqrt(np.mean((np.concatenate([w, mu, sd]) - true) ** 2)))
        for m in d:
            res[m]["degen"].append(np.mean(d[m]))
            res[m]["err"].append(np.median(e[m]))
        print(f"n={n:5d}  degenerate MLE={res['MLE']['degen'][-1]:.2f} MAP={res['MAP']['degen'][-1]:.2f}  "
              f"median RMSE MLE={res['MLE']['err'][-1]:.3f} MAP={res['MAP']['err'][-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, c in (("MLE", C_MLE), ("MAP", C_MAP)):
        axes[0].plot(ns, res[m]["degen"], "o-", color=c, lw=2.5, label=m)
        axes[1].plot(ns, res[m]["err"], "o-", color=c, lw=2.5, label=m)
    axes[0].set_ylabel("fraction of fits with a collapsed component")
    axes[1].set_ylabel("median parameter RMSE")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("sample size n")
        ax.legend(frameon=False)
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle(f"{reps} simulated datasets per n; prior influence fades as n grows", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_monte_carlo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return res


if __name__ == "__main__":
    fig_singularity(np.random.default_rng(0))
    seed, mle, map_ = fig_single_dataset()
    print(f"fig2 seed={seed}\n  MLE: w={mle[0]}, mu={mle[1]}, sd={mle[2]}\n  MAP: w={map_[0]}, mu={map_[1]}, sd={map_[2]}")
    fig_monte_carlo()