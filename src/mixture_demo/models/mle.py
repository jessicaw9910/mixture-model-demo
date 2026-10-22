"""Maximum-likelihood EM."""

from __future__ import annotations

import numpy as np

from mixture_demo.models.base import EMGaussianMixture
from mixture_demo.models.mixins import MultiStartMixin
from mixture_demo.models.params import MixtureParams


class MLEMixture(MultiStartMixin, EMGaussianMixture):
    """Plain EM; the objective is the (unbounded) log-likelihood.

    Parameters
    ----------
    **kwargs
        See :class:`~mixture_demo.models.base.EMGaussianMixture`.
    """

    label = "MLE"

    def m_step(self, x: np.ndarray, resp: np.ndarray, params: MixtureParams):
        nk = self.effective_counts(resp)
        weights = nk / len(x)
        means = (resp * x[:, None]).sum(0) / nk
        variances = (resp * (x[:, None] - means) ** 2).sum(0) / nk
        return weights, means, variances
