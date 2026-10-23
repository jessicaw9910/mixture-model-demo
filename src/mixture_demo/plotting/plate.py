"""Plate diagram of the Bayesian mixture model that MAP assumes (README §2.1).

    pi             ~ Dirichlet(alpha, ..., alpha)
    mu_k           ~ Normal(m0, tau^2)              k = 1..K
    sigma^2_k      ~ Inv-Gamma(a0, b0)              k = 1..K
    z_i | pi       ~ Categorical(pi)                i = 1..n
    x_i | z_i, ... ~ Normal(mu_{z_i}, sigma^2_{z_i})
"""

from __future__ import annotations

import daft
from matplotlib.figure import Figure

from mixture_demo.plotting.base import FigureRenderer

# (parent, child) pairs: one arrow per conditional dependence in the generative model
EDGES = [
    ("alpha", "pi"), ("pi", "z"), ("z", "x"),
    ("m0", "mu"), ("tau", "mu"), ("a0", "s2"), ("b0", "s2"),
    ("mu", "x"), ("s2", "x"),
]


class PlateDiagramFigure(FigureRenderer[None]):
    """Fig. 0: plate diagram of the generative model.

    Fixed hyperparameters are small dots, the observed ``x_i`` is shaded, and
    plates mark repetition over observations ``i`` and components ``k``. Needs
    no experiment, so :meth:`draw` ignores its argument.
    """

    stem = "fig0_plate_diagram"
    style = {"font.size": 13}

    def draw(self, result: None = None) -> Figure:
        pgm = daft.PGM(shape=None, node_unit=1.0, grid_unit=2.0, line_width=1.2, dpi=self.dpi)

        # Fixed hyperparameters (small filled dots, labels offset)
        fixed = dict(fixed=True, offset=(0, 8))
        pgm.add_node("alpha", r"$\alpha$", 0.0, 2.6, **fixed)
        side = dict(fixed=True, offset=(14, -6))
        pgm.add_node("m0", r"$m_0$", 4.45, 2.6, **side)
        pgm.add_node("tau", r"$\tau$", 4.45, 1.9, **side)
        pgm.add_node("a0", r"$a_0$", 4.45, 1.3, **side)
        pgm.add_node("b0", r"$b_0$", 4.45, 0.6, **side)

        # Random variables
        pgm.add_node("pi", r"$\boldsymbol{\pi}$", 0.0, 1.6)
        pgm.add_node("z", r"$z_i$", 1.0, 1.6)
        pgm.add_node("x", r"$x_i$", 2.0, 1.6, observed=True)
        pgm.add_node("mu", r"$\mu_k$", 3.4, 2.25)
        pgm.add_node("s2", r"$\sigma^2_k$", 3.4, 0.95)

        for parent, child in EDGES:
            pgm.add_edge(parent, child)

        pgm.add_plate([0.5, 0.95, 2.0, 1.3], label=r"$i = 1,\ldots,n$", shift=-0.1, position="bottom left")
        pgm.add_plate([2.65, 0.3, 1.5, 2.5], label=r"$k = 1,\ldots,K$", shift=-0.1, position="bottom right")

        pgm.render()
        return pgm.figure
