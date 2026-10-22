import numpy as np
import pytest

from mixture_demo import MAPMixture, MixtureParams, MLEMixture, Prior
from mixture_demo.models import EMGaussianMixture

TRUTH = MixtureParams([0.6, 0.4], [-2.0, 2.0], [1.0, 0.7])


@pytest.fixture
def x():
    return TRUTH.sample(40, np.random.default_rng(0))


def _spike(x, sd):
    return MixtureParams([0.5, 0.5], [x.mean(), x[0]], [x.std(), sd])


def test_likelihood_is_unbounded(x):
    lls = [_spike(x, sd).log_likelihood(x) for sd in (1e-2, 1e-10, 1e-30)]
    assert lls[0] < lls[1] < lls[2]


def test_posterior_is_bounded(x):
    prior = Prior.weak(x)
    lp = [(p := _spike(x, sd)).log_likelihood(x) + prior.logpdf(p) for sd in (1.0, 1e-2, 1e-10)]
    assert lp[0] > lp[1] > lp[2]


def test_class_hierarchy():
    assert issubclass(MLEMixture, EMGaussianMixture)
    assert issubclass(MAPMixture, EMGaussianMixture)
    with pytest.raises(TypeError):
        EMGaussianMixture()  # abstract


@pytest.mark.parametrize("make", [lambda x: MLEMixture(), lambda x: MAPMixture(prior=Prior.weak(x))])
def test_large_sample_recovers_truth(make):
    x = TRUTH.sample(2000, np.random.default_rng(1))
    fit = make(x).fit_best(x, np.random.default_rng(2), n_starts=5)
    np.testing.assert_allclose(fit.params.means, TRUTH.means, atol=0.15)
    np.testing.assert_allclose(fit.params.weights, TRUTH.weights, atol=0.05)
    assert not fit.params.is_collapsed(0.05)


def test_map_objective_includes_prior(x):
    prior = Prior.weak(x)
    model = MAPMixture(prior=prior)
    p = model.random_init(x, np.random.default_rng(0))
    assert model.objective(x, p) == pytest.approx(p.log_likelihood(x) + prior.logpdf(p))
