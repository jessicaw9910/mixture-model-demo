"""Maximum a posteriori EM."""

from __future__ import annotations

import numpy as np

from mixture_demo.models.base import EMGaussianMixture
from mixture_demo.models.mixins import MultiStartMixin, PriorMixin
from mixture_demo.models.params import MixtureParams


class MAPMixture(PriorMixin, MultiStartMixin, EMGaussianMixture):
    """EM with one extra pseudo-count term per M-step; the objective is bounded.

    Parameters
    ----------
    prior : Prior
        Prior over the mixture parameters, e.g. :meth:`Prior.weak(x) <mixture_demo.models.priors.Prior.weak>`.
    **kwargs
        See :class:`~mixture_demo.models.base.EMGaussianMixture`.
    """

    label = "MAP"

    def m_step(self, x: np.ndarray, resp: np.ndarray, params: MixtureParams):
        p = self.prior
        n, K = len(x), self.n_components
        nk = self.effective_counts(resp)
        weights = (nk + p.alpha - 1) / (n + K * (p.alpha - 1))
        var_old = params.sds**2
        means = ((resp * x[:, None]).sum(0) / var_old + p.m0 / p.tau**2) / (nk / var_old + 1 / p.tau**2)
        scatter = (resp * (x[:, None] - means) ** 2).sum(0)
        # mode of Inverse-Gamma(a0 + Nk/2, b0 + S/2)
        variances = (2 * p.b0 + scatter) / (2 * p.a0 + nk + 2)
        return weights, means, variances
