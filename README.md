# MLE vs MAP for Gaussian mixtures: the variance singularity

## 1. The pathology

### 1.1 Model

A $K$-component univariate Gaussian mixture with component-specific (heteroscedastic) variances:

$$
p(x \mid \theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \sigma_k^2),
\qquad \theta = (\pi_{1:K}, \mu_{1:K}, \sigma^2_{1:K}),
\qquad \pi_k \ge 0,\ \textstyle\sum_k \pi_k = 1 .
$$

For i.i.d. data $x_{1:n}$ the log-likelihood is

$$
\ell(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \sigma_k^2).
$$

### 1.2 The likelihood has no upper bound

Fix any data point $x_j$. Set $\mu_2 = x_j$ and leave component 1 at any non-degenerate setting with $\pi_1 > 0$ and $\sigma_1 > 0$. Then:

- **Point $j$:** its contribution is
  $\log\big[\pi_1 \mathcal{N}(x_j \mid \mu_1,\sigma_1^2) + \pi_2 (2\pi\sigma_2^2)^{-1/2}\big]$.
  As $\sigma_2 \to 0$ the second term dominates, so this is $-\log\sigma_2 + O(1) \to +\infty$.
- **Every other point $i \ne j$:** the component-2 density $\mathcal{N}(x_i \mid x_j, \sigma_2^2) \to 0$, but component 1 still gives $x_i$ a finite, non-zero density. Each of these terms therefore converges to the finite value $\log[\pi_1 \mathcal{N}(x_i \mid \mu_1,\sigma_1^2)]$.

Hence

$$
\ell(\theta) = -\log \sigma_2 + C + o(1) \quad\xrightarrow[\sigma_2 \to 0]{}\quad +\infty .
$$

Consequences:

1. **The global MLE does not exist.** $\sup_\theta \ell(\theta) = +\infty$ for *every* dataset with $n \ge 1$ and $K \ge 2$. This holds at every sample size. (Kiefer & Wolfowitz 1956; Day 1969.)
2. **The singularities are numerous.** There is one per (data point, component) pair: $nK$ in total, plus combinations in which several components collapse at once.
3. **The divergence is slow.** It is logarithmic, about $\ln 10 \approx 2.3$ nats per decade of $\sigma_2$. A numerical optimiser therefore sees a gently rising ridge, not an obvious failure. Figure 1 covers 40 decades of $\sigma_2$ so that the trend is visible.
4. **Practical frequentist estimators are local maxima.** Theory guarantees a consistent *local* maximiser, a root of the likelihood equations (Redner & Walker 1984). In practice that is what EM returns when it starts in a reasonable basin. If you pick the run with the highest likelihood across many restarts (the "correct" MLE procedure), you move *towards* the degenerate solutions.

### 1.3 Why EM walks into it

The MLE M-step for the variance is

$$
\hat\sigma_k^2 = \frac{S_k}{N_k}, \qquad
N_k = \sum_i r_{ik}, \qquad
S_k = \sum_i r_{ik} (x_i - \hat\mu_k)^2 ,
$$

where $r_{ik}$ is the responsibility of component $k$ for point $i$. Suppose a component's responsibilities concentrate on a single point $x_j$. Then $S_k$ shrinks faster than $N_k$, so $\sigma_k^2$ shrinks. The component becomes sharper, its responsibility for $x_j$ grows, and its responsibility for every other point drops. This positive feedback ends at the singularity. Nothing in the update pushes back.

### 1.4 What the singularity requires

- **Component-specific variances.** If all components share one variance $\sigma^2$, then letting $\sigma \to 0$ sends the density of every point not sitting on a mean to 0. The log-likelihood goes to $-\infty$, and it is bounded above for $n > K$. Hathaway (1985) builds on this: he bounds the variance *ratios* $\min_{k,l} \sigma_k/\sigma_l \ge c > 0$, which restores a well-defined constrained MLE.
- **Continuous data with no measurement error.** If observations are interval-censored or rounded, the likelihood of a point is a probability rather than a density, which is bounded by 1.

**Bimodality alone is not the problem.** A well-separated bimodal mixture is the *easy* case for locating the good local maximum. The pathology comes from the mixture likelihood combined with free variances.

---

## 2. How MAP removes it

### 2.1 Prior (as implemented in `mixture_demo.priors.Prior`)

The priors are independent across parameter blocks and across components. They are semi-conjugate, not the fully conjugate Normal–Inverse-Gamma:

| Parameter | Prior | Default in `Prior.weak(x)` | Config key |
|---|---|---|---|
| $\pi$ | $\text{Dirichlet}(\alpha, \dots, \alpha)$ | $\alpha = 2$ | `prior.alpha` |
| $\mu_k$ | $\mathcal{N}(m_0, \tau^2)$ | $m_0 = \bar{x}$, $\tau = 3 \cdot \text{sd}(x)$ | `prior.tau_scale` (the 3) |
| $\sigma_k^2$ | $\text{Inv-Gamma}(a_0, b_0)$ | $a_0 = 2$, $b_0 = \dfrac{(a_0+1) \cdot \text{var}(x)}{K^2}$ $\left(= \tfrac{3}{4} \text{var}(x) \text{ for } K = 2\right)$ | `prior.a0` |

$b_0$ is chosen so that the **prior mode** of each variance, $b_0/(a_0+1)$, equals $\text{var}(x)/K^2$. That is a rough guess at a single component's variance if the $K$ components split the data's spread evenly. This follows the data-scaled defaults in Fraley & Raftery (2007).

Log prior, up to a constant (`Prior.logpdf`):

$$
\log p(\theta) = (\alpha - 1)\sum_k \log \pi_k
-\sum_k \frac{(\mu_k - m_0)^2}{2\tau^2}
-\sum_k \Big[(a_0+1)\log\sigma_k^2 + \frac{b_0}{\sigma_k^2}\Big].
$$

### 2.2 The posterior is bounded

The MAP objective is $\ell(\theta) + \log p(\theta)$. Along the same singular path ($\mu_2 = x_j$, $\sigma_2 \to 0$):

$$
\underbrace{-\log\sigma_2}_{\text{likelihood}}
\underbrace{-2(a_0+1)\log\sigma_2}_{\text{IG power term}}
\underbrace{-\frac{b_0}{\sigma_2^2}}_{\text{IG exponential term}}
\quad\xrightarrow[\sigma_2\to 0]{}\quad -\infty .
$$

The **IG power term makes the divergence worse**: it also goes to $+\infty$. What rescues the objective is the exponential term $-b_0/\sigma^2$, which goes to $-\infty$ faster than any power of $\log\sigma$ goes to $+\infty$. So any prior on the variance whose density vanishes faster than polynomially at 0 removes the singularity. A prior that is flat or polynomial near zero does not. For example, the Jeffreys-like $p(\sigma^2) \propto 1/\sigma^2$ makes things worse.

### 2.3 MAP-EM updates (in `MAPMixture.m_step`)

The E-step is the same as for MLE (it is shared in `EMGaussianMixture.responsibilities`). `MLEMixture` and `MAPMixture` differ only in their `m_step`, and in the objective: `MAPMixture` adds `Prior.logpdf` through `PriorMixin.log_prior`. The M-step maximises the expected complete-data log posterior:

$$
Q(\theta) = \sum_k \Big[ N_k \log \pi_k - \tfrac{N_k}{2}\log\sigma_k^2 - \tfrac{1}{2\sigma_k^2}\textstyle\sum_i r_{ik}(x_i-\mu_k)^2 \Big] + \log p(\theta).
$$

**Weights.** Maximise $\sum_k (N_k + \alpha - 1)\log\pi_k$ subject to $\sum_k\pi_k = 1$ (Lagrange multiplier):

$$
\hat\pi_k = \frac{N_k + \alpha - 1}{n + K(\alpha - 1)} .
$$

With $\alpha = 2$, this adds one pseudo-count to each component, so no weight can reach 0 or 1. Setting $\alpha = 1$ recovers the MLE update $N_k / n$.

**Means.** This is a precision-weighted average of the data and the prior mean. It uses the variance from the previous iteration:

$$
\hat\mu_k = \frac{\sum_i r_{ik} x_i / \sigma_k^{2(\text{old})} + m_0/\tau^2}{N_k/\sigma_k^{2(\text{old})} + 1/\tau^2} .
$$

**Variances.** Set $\partial Q/\partial\sigma_k^2 = 0$, with $S_k$ computed at the *new* $\hat\mu_k$:

$$
\hat\sigma_k^2 = \frac{2b_0 + S_k}{2a_0 + N_k + 2} .
$$

This is the mode of $\text{Inv-Gamma}(a_0 + N_k/2, b_0 + S_k/2)$. The key property is that it gives a **data-adaptive variance floor**:

$$
\hat\sigma_k^2 \ge \frac{2b_0}{2a_0 + n + 2} > 0,
$$

because $S_k \ge 0$ and $N_k \le n$. The floor shrinks as $O(1/n)$, so the prior's influence fades with more data.

Updating $\mu$ and then $\sigma^2$ in turn, conditional on each other, is an ECM step (Meng & Rubin 1993). It still increases the log posterior monotonically, but a single M-step is not a joint maximisation.

---

## 3. Assumptions and design choices

### 3.1 Data-generating process

- **Truth** (`truth.weights`, `truth.means`, `truth.sds`): $K = 2$; $\pi = (0.6, 0.4)$; $\mu = (-2, +2)$; $\sigma = (1.0, 0.7)$. The two modes are well separated (about 4 pooled SDs apart), and the weights and variances are deliberately unequal.
- **Model correctly specified.** The fitted model has the same family and the same $K$ as the truth. Misspecification, and overfitting $K$, are not explored. Fitting $K = 3$ to these data would make collapse much more frequent, because a surplus component has nothing better to do than collapse.

### 3.2 Prior

- **Empirical Bayes.** $m_0$, $\tau$ and $b_0$ come from $\bar x$ and $\text{var}(x)$, so the data are used twice. This is standard for weak regularising priors (Fraley & Raftery 2007), but it is not a fully Bayesian prior.
- **The prior is not neutral on $\sigma$.** It shrinks each $\sigma_k^2$ towards $\text{var}(x)/4$. For these data $\text{var}(x) \approx 0.6(1) + 0.4(0.49) + 0.24\cdot 4^2 \approx 4.6$, so the prior mode is $\sigma \approx 1.08$. That pulls the narrow component ($\sigma_2 = 0.7$) upwards. This explains why figure 2's MAP fit reports $\hat\sigma_2 = 0.97$. **MAP trades a little bias for removing the singularity.**
- **Near-flat prior on the means.** $\tau = 3 \cdot \text{sd}(x) \approx 6.4$, so the mean prior has essentially no effect. All of the regularisation comes from the weight and variance priors.
- **MAP depends on parameterisation.** The prior is placed on $\sigma^2$ and the mode is taken in $\sigma^2$. Taking the mode in $\sigma$ or $\log\sigma$ under the transformed prior would give a different estimate. MAP is a point summary, not the posterior.

### 3.3 Optimisation

- **Initialisation (`MultiStartMixin.random_init`):** the means are $K$ distinct data points drawn at random; $\sigma_k = \text{sd}(x)\cdot U(0.2, 1)$; $\pi \sim \text{Dirichlet}(1,\dots,1)$.
- **Multi-start (`MultiStartMixin.fit_best`):** EM runs from $N$ random starts (30 in figure 2, `single_dataset.n_starts`; 15 in figure 3, `monte_carlo.n_starts`), and the run with the highest objective is kept. For MLE this is the honest "maximise the likelihood" procedure, and it is exactly what favours degenerate solutions.
- **Convergence (`EMGaussianMixture.fit`):** stop when |Δobjective| < $10^{-9}$ (`model.tol`), or after at most 500 iterations (`model.max_iter`).
- **Numerical floor:** `model.sd_floor = 1e-3` is applied in both methods. It only prevents NaN/inf. In MLE it truncates the singularity, so a fit sitting at the floor would have kept collapsing. It never binds for MAP, because the MAP floor is orders of magnitude larger.
- **Label switching** is resolved after fitting by sorting components by $\hat\mu$.

### 3.4 Evaluation

- **"Collapsed" component:** $\min_k \hat\sigma_k < 0.05$ (`model.collapse_sd`; checked by `MixtureParams.is_collapsed`), which is 14× below the smallest true SD.
- **Error metric:** the RMSE over all six parameters $(\pi_1,\pi_2,\mu_1,\mu_2,\sigma_1,\sigma_2)$ after label alignment, with the median taken across replicates. The metric pools parameters on different scales. The median also hides the degenerate tail: mean or upper-quantile RMSE would show a larger MLE penalty at small $n$.
- **The collapse rate is about EM, not the estimator.** The supremum of the likelihood is infinite for *every* dataset. Figure 3's collapse rate (about 10% at $n = 15$, 0% from $n = 50$) measures how often *15 EM restarts from this initialisation* reach a degenerate basin and score highest. It is a lower bound on the pathology. More restarts, or inits that place a mean on a data point with a small $\sigma$, would raise it.
- **Figure 2 is a chosen example.** `SingleDatasetExperiment` loops over seeds, starting at `single_dataset.seed` and trying up to `single_dataset.max_search` of them, until the best-of-30 MLE is degenerate. With the defaults, seed 8 is the first. If no seed in the range gives a collapse, the last seed is plotted and the figure title says so. It illustrates the failure mode and does not show how often it occurs; figure 3 is the frequency estimate.

---

## 4. What each figure computes

Each figure is computed by an experiment class in `mixture_demo.experiments` and drawn by a renderer class in `mixture_demo.plotting`. The values below are the defaults; the config key for each is shown in brackets.

### Figure 1: `SingularityExperiment` → `fig1_singularity.png`
- $n = 40$ draws (`singularity.n`), seed 0 (`singularity.seed`).
- Parameters fixed at $\pi = (0.5, 0.5)$, $\mu_1 = \bar x$, $\sigma_1 = \text{sd}(x)$ (a broad "background" component), and $\mu_2 = $ the data point nearest the largest true mean ($+2$).
- $\sigma_2$ is swept log-uniformly over $[10^{-40}, 3]$ (`singularity.log10_sd_min`, 800 points: `singularity.n_grid`), and $\ell$ and $\ell + \log p$ are evaluated along the sweep.
- Both curves are shifted to 0 at $\sigma_2 = 1$. The likelihood rises linearly in $-\log_{10}\sigma_2$ at about 2.3 nats per decade. The posterior falls off a cliff below $\sigma_2 \approx 0.3$.
- This is a 1-D slice through $\theta$. It shows that the singularity exists, not where EM goes.

### Figure 2: `SingleDatasetExperiment` → `fig2_single_dataset.png`
- $n = 25$ (`single_dataset.n`), using the first seed whose best-of-30 MLE fit is collapsed. The MLE and MAP fits use separate random-start streams (seed + 10 000 and seed + 20 000), which are independent of the data seed.
- Shows a histogram, the true density, and each fitted density, annotated with $(\hat\pi, \hat\mu, \hat\sigma)$.
- Result (seed 8):

  | | $\pi_1, \pi_2$ | $\mu_1, \mu_2$ | $\sigma_1, \sigma_2$ |
  |---|---|---|---|
  | Truth | 0.60, 0.40 | −2.00, +2.00 | 1.00, 0.70 |
  | MLE | 0.89, 0.11 | −1.46, +1.96 | 1.85, **0.047** |
  | MAP | 0.69, 0.31 | −2.16, +1.55 | 1.19, 0.97 |

  The collapse doesn't just break the spiked component. The broad component then has to explain all the remaining data, so $\hat\sigma_1$ and $\hat\pi_1$ are badly inflated too.

### Figure 3: `MonteCarloExperiment` → `fig3_monte_carlo.png`
- $n \in \lbrace 15, 25, 50, 100, 250, 1000\rbrace$ (`monte_carlo.sample_sizes`), 200 replicate datasets for each $n$ (`monte_carlo.reps`), best of 15 EM starts per fit (`monte_carlo.n_starts`), seed 1 (`monte_carlo.seed`). The numbers below are also written to each run's `summary.json`.

  | $n$ | Collapsed: MLE | Collapsed: MAP | Median RMSE: MLE | Median RMSE: MAP |
  |---:|---:|---:|---:|---:|
  | 15 | 0.10 | 0.00 | 0.283 | 0.203 |
  | 25 | 0.01 | 0.00 | 0.198 | 0.174 |
  | 50 | 0.00 | 0.00 | 0.127 | 0.120 |
  | 100 | 0.00 | 0.00 | 0.092 | 0.090 |
  | 250 | 0.00 | 0.00 | 0.059 | 0.059 |
  | 1000 | 0.00 | 0.00 | 0.027 | 0.027 |

- The two estimators converge as $n$ grows. This is consistent with the MAP variance floor shrinking as $O(1/n)$ and the data overwhelming the prior.

---

## 5. Takeaways for the talk

1. The mixture likelihood is unbounded once variances are free, so "the MLE" of a heteroscedastic Gaussian mixture is not a well-defined object. What practitioners use is a well-behaved local maximum.
2. A weak prior on the variances that vanishes faster than any polynomial at zero (Inverse-Gamma) makes the objective bounded. In EM it shows up as pseudo-counts, which act as an adaptive variance floor.
3. The benefit is concentrated at small $n$. At large $n$ the two estimators coincide.
4. Frequentist fixes exist and are close relatives: variance-ratio constraints (Hathaway 1985) and penalised likelihood (Ciuperca, Ridolfi & Idier 2003). MAP can be read as a penalised MLE whose penalty is $-\log p(\theta)$.
5. MAP is still a single point. It is not invariant to reparameterisation and gives no uncertainty. The natural next step is the full posterior (e.g., MCMC), which also exposes label-switching multimodality.

---

## References

- Ciuperca, G., Ridolfi, A. & Idier, J. (2003). Penalized maximum likelihood estimator for normal mixtures. *Scandinavian Journal of Statistics* 30(1):45–59.
- Day, N. E. (1969). Estimating the components of a mixture of normal distributions. *Biometrika* 56(3):463–474.
- Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B* 39(1):1–38.
- Fraley, C. & Raftery, A. E. (2007). Bayesian regularization for normal mixture estimation and model-based clustering. *Journal of Classification* 24:155–181.
- Hathaway, R. J. (1985). A constrained formulation of maximum-likelihood estimation for normal mixture distributions. *Annals of Statistics* 13(2):795–800.
- Kiefer, J. & Wolfowitz, J. (1956). Consistency of the maximum likelihood estimator in the presence of infinitely many incidental parameters. *Annals of Mathematical Statistics* 27(4):887–906.
- Meng, X.-L. & Rubin, D. B. (1993). Maximum likelihood estimation via the ECM algorithm: a general framework. *Biometrika* 80(2):267–278.
- Redner, R. A. & Walker, H. F. (1984). Mixture densities, maximum likelihood and the EM algorithm. *SIAM Review* 26(2):195–239.

---

## Setup

Requires Python ≥ 3.10.

```bash
git clone https://github.com/jessicaw9910/mixture-model-demo.git mixture_model_demo
cd mixture_model_demo
```

**With uv (recommended)**

```bash
uv sync                    # creates .venv, installs the package + pytest
uv run mixture-demo --help
```

Either prefix commands with `uv run`, or activate the environment with `source .venv/bin/activate`.

**With pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"    # drop [dev] if you don't need pytest
mixture-demo --help
```

To check the install, run `pytest` (or `uv run pytest`).

## Running the demo

```bash
mixture-demo run                      # all three figures, default parameters (~7 min, mostly fig. 3)
mixture-demo run monte_carlo.reps=20  # quicker: fewer Monte Carlo replicates (under a minute)
```

`python -m mixture_demo run` works too.

Each run writes to a new timestamped folder:

```
images/<YYYY-MM-DD_HH-MM-SS>/
├── fig1_singularity.png
├── fig2_single_dataset.png
├── fig3_monte_carlo.png
├── config.yaml     # every parameter used; reusable with --config
└── summary.json    # fitted parameters, collapse rates, timings
```

Run folders are git-ignored. To commit one, use `git add -f images/<run>`.

### Changing parameters

Every parameter can be overridden with a `key=value` argument. To print all keys and their current values, run `mixture-demo show-config`.

```bash
# only figures 1 and 2, with a larger dataset for figure 2
mixture-demo run -f singularity -f single-dataset single_dataset.n=40

# a different ground truth and a stronger variance prior
mixture-demo run 'truth.means=[-1,1]' 'truth.sds=[1,1]' prior.a0=5

# a shorter Monte Carlo sweep
mixture-demo run 'monte_carlo.sample_sizes=[20,50,200]' monte_carlo.reps=50
```

> In zsh, put quotes around any value written in `[...]`. Otherwise the shell reads the brackets as a filename pattern and the command fails.

| Section | Keys |
|---|---|
| `truth` | `weights`, `means`, `sds` |
| `model` | `max_iter`, `tol`, `sd_floor`, `collapse_sd` |
| `prior` | `alpha`, `a0`, `tau_scale` |
| `singularity` (fig. 1) | `n`, `seed`, `log10_sd_min`, `n_grid` |
| `single_dataset` (fig. 2) | `n`, `seed`, `max_search`, `n_starts`, `bins` |
| `monte_carlo` (fig. 3) | `sample_sizes`, `reps`, `seed`, `n_starts` |
| `output` | `images_dir`, `dpi` |

### Reproducing a run

Pass a previous run's `config.yaml` to rerun it with the same parameters. Any `key=value` overrides you add on top take precedence over the file:

```bash
mixture-demo run --config images/<run>/config.yaml
mixture-demo run --config images/<run>/config.yaml monte_carlo.reps=500
```

To write your own config, start from the defaults:

```bash
mixture-demo show-config > my_config.yaml
mixture-demo run --config my_config.yaml
```

Other options: `-o DIR` writes into a specific folder instead of a timestamped one, and `-q` hides progress logging. For the full list, run `mixture-demo run --help`.

### From Python

```python
from mixture_demo import DemoConfig, DemoRunner

cfg = DemoConfig().with_overrides(["monte_carlo.reps=20", "figures=[singularity,monte-carlo]"])
run_dir = DemoRunner(cfg).run()
```

The estimators can also be used on their own:

```python
import numpy as np
from mixture_demo import MAPMixture, MixtureParams, MLEMixture, Prior

truth = MixtureParams(weights=[0.6, 0.4], means=[-2, 2], sds=[1.0, 0.7])
x = truth.sample(25, np.random.default_rng(8))

mle = MLEMixture().fit_best(x, np.random.default_rng(0), n_starts=30)
map_ = MAPMixture(prior=Prior.weak(x)).fit_best(x, np.random.default_rng(0), n_starts=30)
print(mle.params.sds, map_.params.sds)
```

## Code layout

```
src/mixture_demo/
├── models/
│   ├── params.py       # MixtureParams: weights/means/sds, log-likelihood, pdf, sampling
│   ├── base.py         # EMGaussianMixture (abstract): E-step, EM loop, sd floor; FitResult
│   ├── mixins.py       # MultiStartMixin (random_init, fit_best); PriorMixin (adds log prior)
│   ├── mle.py          # MLEMixture(MultiStartMixin, EMGaussianMixture): MLE M-step
│   └── map.py          # MAPMixture(PriorMixin, MultiStartMixin, EMGaussianMixture): MAP M-step
├── priors.py           # Prior, Prior.weak(x)
├── experiments/        # computation only: Singularity-, SingleDataset-, MonteCarloExperiment
├── plotting/           # drawing only: one renderer per figure, shared style
├── config.py           # DemoConfig (OmegaConf structured config): every tunable parameter
├── runner.py           # DemoRunner: runs experiments, writes images/<timestamp>/
└── cli/main.py         # `mixture-demo` (typer)
```
