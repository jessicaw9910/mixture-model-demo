"""
MLE vs MAP for a 1-D Gaussian mixture.

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
"""

from importlib.metadata import PackageNotFoundError, version

from mixture_demo.config import DemoConfig
from mixture_demo.models import FitResult, MAPMixture, MixtureParams, MLEMixture
from mixture_demo.priors import Prior
from mixture_demo.runner import DemoRunner

try:
    __version__ = version("mixture-model-demo")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "DemoConfig",
    "DemoRunner",
    "FitResult",
    "MAPMixture",
    "MLEMixture",
    "MixtureParams",
    "Prior",
    "__version__",
]
