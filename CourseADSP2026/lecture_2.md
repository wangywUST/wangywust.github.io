# Modern Digital Signal Processing
## Chapter 2: Analysis of Discrete Random Signals

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005
> Chapters covered: Ch. 3 (Random Variables, Vectors, and Sequences) · §3.5 (Whitening and Innovations) · §3.6 (Principles of Estimation Theory) · §4.1–§4.3 (Linear Signal Models) · §2.4 (Spectral Factorization)

---

## Table of Contents

1. [§1 Random Variables](#1-random-variables)
2. [§2 Stochastic Processes and Their Statistical Description](#2-stochastic-processes-and-their-statistical-description)
3. [§3 Random Processes through Filters](#3-random-processes-through-filters)
4. [§4 Spectral Factorization](#4-spectral-factorization)
5. [§5 Special Types of Random Processes](#5-special-types-of-random-processes)
6. [§6 Basic Orthogonal Transforms](#6-basic-orthogonal-transforms)
7. [§7 Basic Parametric Estimation Methods](#7-basic-parametric-estimation-methods)

---

## Notation and Variable Definitions

All symbols used in this chapter are listed below. Where a symbol carries different meanings in different sections, the context is explicitly noted.

### Probability and Statistics

| Symbol | Definition |
|--------|-----------|
| $S = \lbrace \zeta_1, \zeta_2, \ldots\rbrace$ | Sample space (universal set of all outcomes) |
| $\zeta$ | A random outcome (element of the abstract probability space) |
| $x(\zeta)$ | Random variable: a mapping from outcome $\zeta$ to a real number $x$ |
| $F_x(x) \triangleq \Pr\lbrace x(\zeta) \le x\rbrace$ | Cumulative distribution function (CDF) |
| $f_x(x) = dF_x(x)/dx$ | Probability density function (PDF) |
| $p_k = \Pr\lbrace x(\zeta) = x_k\rbrace$ | Probability mass function (PMF), for discrete-valued RVs |
| $\mu_x = E\lbrace x(\zeta)\rbrace$ | Mean (first moment) of the random variable $x(\zeta)$ |
| $\sigma_x^2 = E\lbrace \lvert x - \mu_x\rvert^2\rbrace$ | Variance (second central moment) |
| $\sigma_x$ | Standard deviation |
| $r_x^{(m)} = E\lbrace x^m(\zeta)\rbrace$ | $m$-th order moment |
| $\gamma_x^{(m)} = E\lbrace (x-\mu_x)^m\rbrace$ | $m$-th order central moment |
| $\tilde{\kappa}_x^{(3)}$ | Skewness (normalized 3rd central moment) |
| $\tilde{\kappa}_x^{(4)}$ | Kurtosis (normalized 4th central moment $-3$) |
| $\Phi_x(\xi) = E\lbrace e^{j\xi x(\zeta)}\rbrace$ | Characteristic function |

### Random Vectors

| Symbol | Definition |
|--------|-----------|
| $\mathbf{x}(\zeta) = [x_1(\zeta),\ldots,x_M(\zeta)]^T$ | Random vector of dimension $M$ |
| $\boldsymbol{\mu}_x = E\lbrace \mathbf{x}\rbrace$ | Mean vector |
| $\mathbf{R}_x = E\lbrace \mathbf{x}\mathbf{x}^H\rbrace$ | Correlation matrix ($M\times M$, Hermitian non-negative definite) |
| $\boldsymbol{\Sigma}_x = E\lbrace (\mathbf{x}-\boldsymbol{\mu}_x)(\mathbf{x}-\boldsymbol{\mu}_x)^H\rbrace$ | Covariance matrix |
| $r_{x_i x_j} = E\lbrace x_i x_j^*\rbrace$ | Cross-correlation between $x_i$ and $x_j$ |
| $\gamma_{x_i x_j}$ | Cross-covariance between $x_i$ and $x_j$ |
| $\rho_{x_i x_j} = \gamma_{x_i x_j}/(\sigma_{x_i}\sigma_{x_j})$ | Correlation coefficient, $\lvert\rho\rvert\le 1$ |

### Random Processes (Sequences)

| Symbol | Definition |
|--------|-----------|
| $x(n) \equiv x(n,\zeta)$ | Discrete-time stochastic process (random sequence) |
| $\mu_x(n) = E\lbrace x(n)\rbrace$ | Mean function (may be time-varying) |
| $\sigma_x^2(n) = E\lbrace \lvert x(n)-\mu_x(n)\rvert^2\rbrace$ | Variance function |
| $r_{xx}(n_1, n_2) = E\lbrace x(n_1)x^*(n_2)\rbrace$ | Autocorrelation function |
| $\gamma_{xx}(n_1, n_2)$ | Autocovariance function |
| $r_{xy}(n_1, n_2) = E\lbrace x(n_1)y^*(n_2)\rbrace$ | Cross-correlation function |
| $\gamma_{xy}(n_1, n_2)$ | Cross-covariance function |
| $\rho_{xy}(n_1,n_2)$ | Normalized cross-correlation coefficient |
| $r_x(l) = E\lbrace x(n+l)x^*(n)\rbrace$ | WSS autocorrelation (function of lag $l$ only) |
| $\gamma_x(l) = r_x(l) - \lvert\mu_x\rvert^2$ | WSS autocovariance |
| $r_{xy}(l) = E\lbrace x(n)y^*(n-l)\rbrace$ | WSS cross-correlation |

### Power Spectral Density and z-Domain

| Symbol | Definition |
|--------|-----------|
| $R_x(e^{j\omega}) = \sum_{l=-\infty}^{\infty} r_x(l)e^{-j\omega l}$ | Power spectral density (PSD, or autoPSD) |
| $R_x(z) = \sum_{l=-\infty}^{\infty} r_x(l) z^{-l}$ | Complex spectral density (z-transform of $r_x(l)$) |
| $R_{xy}(e^{j\omega})$ | Cross-power spectral density |
| $G_{xy}(e^{j\omega})$ | Coherence function |
| $\lvert G_{xy}(e^{j\omega})\rvert^2$ | Magnitude-squared coherence (MSC) |
| $\sigma_w^2$ | Variance of white noise process |
| $w(n) \sim \mathrm{WN}(\mu_w, \sigma_w^2)$ | White noise with mean $\mu_w$, variance $\sigma_w^2$ |
| $w(n) \sim \mathrm{WGN}(\mu_w, \sigma_w^2)$ | White Gaussian noise |
| $w(n) \sim \mathrm{IID}(\mu_w, \sigma_w^2)$ | Independently and identically distributed (strict white noise) |

### Correlation Matrices

| Symbol | Definition |
|--------|-----------|
| $\mathbf{R}_x$ | Autocorrelation matrix of stationary $x(n)$: Hermitian, Toeplitz, non-negative definite |
| $\mathbf{Q}_x = [\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_M]$ | Eigenmatrix (unitary, columns are eigenvectors of $\mathbf{R}_x$) |
| $\boldsymbol{\Lambda}_x = \mathrm{diag}\lbrace \lambda_1, \ldots, \lambda_M\rbrace$ | Diagonal matrix of eigenvalues, $\lambda_i \ge 0$ |
| $\mathbf{R}_x = \mathbf{Q}_x \boldsymbol{\Lambda}_x \mathbf{Q}_x^H$ | Eigendecomposition of the correlation matrix |
| $\mathbf{L}$ | Unit lower triangular matrix in LDL$^H$ decomposition |
| $\mathbf{D}_L = \mathrm{diag}\lbrace \xi_1,\ldots,\xi_M\rbrace$ | Diagonal matrix with positive elements (LDL$^H$ decomposition) |

### Linear Signal Models

| Symbol | Definition |
|--------|-----------|
| $a_k$, $k=1,\ldots,p$ | AR (autoregressive) coefficients |
| $b_k$, $k=0,\ldots,q$ | MA (moving-average) coefficients |
| $H(z)$ | Transfer function of the signal model filter |
| $\sigma_w^2$ | Driving white-noise variance (model excitation power) |
| $P_x(z) = \sigma_w^2 H(z)H^{\ast}(1/z^{\ast})$ | Complex power spectral density (rational spectral density) |
| $H_+(z)$ | Minimum-phase spectral factor of $P_x(z)$ |

### Estimation Theory

| Symbol | Definition |
|--------|-----------|
| $\hat{\theta}$ | Estimate of the deterministic parameter $\theta$ |
| $b(\hat{\theta}) = E\lbrace \hat{\theta}\rbrace - \theta$ | Bias of the estimator |
| $\mathrm{MSE}(\hat{\theta}) = E\lbrace \lvert\hat{\theta}-\theta\rvert^2\rbrace$ | Mean-squared error |
| $\mathrm{CRLB}$ | Cramér–Rao lower bound on the estimator variance |
| $\mathbf{J}(\boldsymbol{\theta})$ | Fisher information matrix |
| $\mathbf{A}^T, \mathbf{A}^H$ | Transpose and Hermitian (conjugate) transpose |
| $\hat{\boldsymbol{\theta}}_{\mathrm{LS}}$ | Least-squares estimate |
| $\hat{\boldsymbol{\theta}}_{\mathrm{ML}}$ | Maximum-likelihood estimate |
| $\hat{\boldsymbol{\theta}}_{\mathrm{MAP}}$ | Maximum a posteriori estimate |
| $\hat{\boldsymbol{\theta}}_{\mathrm{MMSE}}$ | Minimum mean-square-error estimate |

---

# §1 Random Variables

> 📖 Textbook §3.1 (Random Variables)

## 1.1 Probability Distribution and Density Functions

### Motivation: From Abstract Events to Numbers

In engineering, we must handle random phenomena — the amplitude of acoustic noise, the path delay of a radar return, the channel coefficient in a wireless link — in a quantitative way. The theory of probability begins with an abstract probability space $(S, \mathcal{F}, \Pr)$, where $S$ is the sample space of all possible outcomes $\zeta$. This abstraction is difficult to manipulate with calculus. A **random variable** solves this problem by mapping each abstract outcome to a number on the real line.

> **Definition 2.1 (Random Variable).** A *random variable* $x(\zeta)$ is a function that assigns a real number $x$ to every outcome $\zeta \in S$, satisfying: (1) the set $\lbrace x(\zeta) \le x\rbrace$ is an event in $S$ for every $x$; (2) $\Pr\lbrace x(\zeta)=\pm\infty\rbrace=0$.

> ![Figure 1.1](./CourseADSP2026/Fig/Chapter_2/fig_1_1.png)
>
> *Figure 1.1 (Textbook Fig. 3.1, p. 76): Graphical illustration of the random variable mapping — the abstract sample space $S$ on the left is mapped to the real line $\mathbb{R}$ on the right. Each outcome $\zeta_k$ (an abstract event) gets assigned a numerical value $x(\zeta_k)$.*

**Key insight:** once outcomes are mapped to numbers, we can apply integration and differentiation — the standard tools of analysis — to compute probabilities and expectations.

### Cumulative Distribution Function (CDF)

$$\boxed{F_x(x) \triangleq \Pr\lbrace x(\zeta) \le x\rbrace}$$

Properties of any valid CDF:
- $0 \le F_x(x) \le 1$
- $F_x(-\infty) = 0$, $F_x(+\infty) = 1$
- Monotonically non-decreasing and right-continuous

### Probability Density Function (PDF)

For continuous-valued random variables, the **PDF** is the formal derivative of the CDF:

$$\boxed{f_x(x) \triangleq \frac{dF_x(x)}{dx}}$$

The PDF is **not** a probability by itself; to get a probability, you must integrate it over an interval:

$$\Pr\lbrace x_1 < x(\zeta) \le x_2\rbrace = F_x(x_2) - F_x(x_1) = \int_{x_1}^{x_2} f_x(x)\, dx$$

Any valid PDF satisfies:
$$f_x(x) \ge 0 \quad \text{and} \quad \int_{-\infty}^{+\infty} f_x(x)\, dx = 1$$

For **discrete-valued** random variables, the **probability mass function (PMF)** $p_k = \Pr\lbrace x(\zeta) = x_k\rbrace$ replaces the PDF. We can unify both cases by allowing impulses in the PDF: $f_x(x) = \sum_k p_k \delta(x - x_k)$.

---

## 1.2 Statistical Averages (Moments)

Rather than specifying the entire PDF, it is often sufficient to summarize a random variable by a few numerical descriptors — its **moments**. These are computed via the **expectation operator** $E\lbrace \cdot\rbrace$.

### Mean (First Moment)

$$\boxed{\mu_x = E\lbrace x(\zeta)\rbrace = \int_{-\infty}^{+\infty} x\, f_x(x)\, dx}$$

The mean $\mu_x$ is the "center of gravity" of the density function $f_x(x)$ — the location around which the random variable tends to cluster.

**Key property (linearity):** $E\lbrace \alpha x(\zeta) + \beta\rbrace = \alpha \mu_x + \beta$.

### Moments and Central Moments

The **$m$-th moment** is $r_x^{(m)} = E\lbrace x^m(\zeta)\rbrace$. In particular:
- $r_x^{(0)} = 1$ (normalization)
- $r_x^{(1)} = \mu_x$ (mean)
- $r_x^{(2)} = E\lbrace x^2\rbrace$ (**mean-square value** — total average power if $x$ is a signal)

The **$m$-th central moment** measures spread around the mean: $\gamma_x^{(m)} = E\lbrace (x - \mu_x)^m\rbrace$. The relationship between moments and central moments is:

$$\gamma_x^{(m)} = \sum_{k=0}^{m} \binom{m}{k} (-1)^k \mu_x^k\, r_x^{(m-k)}$$

### Variance and Standard Deviation

$$\boxed{\sigma_x^2 \triangleq \gamma_x^{(2)} = E\lbrace \lvert x(\zeta) - \mu_x\rvert^2\rbrace = E\lbrace x^2\rbrace - \mu_x^2}$$

The variance $\sigma_x^2$ measures the **spread** of the distribution around the mean. The **standard deviation** $\sigma_x = \sqrt{\sigma_x^2}$ has the same physical units as $x(\zeta)$.

> **Physical meaning:** $\sigma_x^2$ is the AC power of the random variable (the DC component having been removed), while $r_x^{(2)} = E\lbrace x^2\rbrace$ is the total average power:
>
> $$\underbrace{E\lbrace x^2\rbrace}_{\text{total power}} = \underbrace{\mu_x^2}_{\text{DC power}} + \underbrace{\sigma_x^2}_{\text{AC power}}$$

### Higher-Order Descriptors

> ![Figure 1.2](./CourseADSP2026/Fig/Chapter_2/fig_1_2.png)
>
> *Figure 1.2 (Textbook Fig. 3.2, p. 78): Illustration of mean, variance, skewness, and kurtosis for two different distributions. (a) Mean: the "balance point" of the density. (b) Variance: the spread. (c) Skewness: the asymmetry around the mean. (d) Kurtosis: the relative peakedness compared to a Gaussian.*

**Skewness** characterizes the asymmetry of the distribution:

$$\tilde{\kappa}_x^{(3)} \triangleq \frac{\gamma_x^{(3)}}{\sigma_x^3} = E\!\left\lbrace \left(\frac{x - \mu_x}{\sigma_x}\right)^3\right\rbrace$$

- $\tilde{\kappa}_x^{(3)} = 0$: symmetric distribution
- $\tilde{\kappa}_x^{(3)} > 0$: right-tailed (leans to the right)
- $\tilde{\kappa}_x^{(3)} < 0$: left-tailed

**Kurtosis** characterizes the peakedness relative to a Gaussian distribution:

$$\tilde{\kappa}_x^{(4)} \triangleq \frac{\gamma_x^{(4)}}{\sigma_x^4} - 3$$

The $-3$ normalization makes the kurtosis zero for a Gaussian. Distributions with $\tilde{\kappa}_x^{(4)} > 0$ (leptokurtic) have heavier tails than Gaussian — they produce more "outliers." This matters greatly for radar, sonar, and communication channel models.

**Chebyshev's Inequality:** A universal (distribution-free) bound on how far a random variable can stray from its mean:

$$\Pr\lbrace \lvert x(\zeta) - \mu_x\rvert \ge k\sigma_x\rbrace \le \frac{1}{k^2}, \qquad k > 0$$

---

## 1.3 Joint Statistical Description of Two Random Variables

When two random variables $x_1(\zeta)$ and $x_2(\zeta)$ are defined on the same probability space, they are jointly described by the **joint PDF** $f_{x_1 x_2}(x_1, x_2)$. The individual (marginal) PDFs are recovered by integration:

$$f_{x_1}(x_1) = \int_{-\infty}^{+\infty} f_{x_1 x_2}(x_1, x_2)\, dx_2$$

### Cross-Correlation and Covariance

The **cross-correlation** between $x_1(\zeta)$ and $x_2(\zeta)$:

$$r_{x_1 x_2} = E\lbrace x_1(\zeta)\, x_2^*(\zeta)\rbrace = \int\!\!\int x_1 x_2^*\, f_{x_1 x_2}(x_1, x_2)\, dx_1\, dx_2$$

The **cross-covariance** (centered version):

$$\gamma_{x_1 x_2} = E\lbrace (x_1 - \mu_{x_1})(x_2 - \mu_{x_2})^*\rbrace = r_{x_1 x_2} - \mu_{x_1}\mu_{x_2}^*$$

The **correlation coefficient** (normalized to $[-1, 1]$):

$$\rho_{x_1 x_2} = \frac{\gamma_{x_1 x_2}}{\sigma_{x_1} \sigma_{x_2}}, \qquad \lvert\rho_{x_1 x_2}\rvert \le 1$$

$\lvert\rho\rvert = 1$ indicates a perfect linear relationship; $\lvert\rho\rvert = 0$ indicates **uncorrelated** variables.

### Independence vs. Uncorrelated

Two random variables are:
- **Statistically independent** if $f_{x_1 x_2}(x_1, x_2) = f_{x_1}(x_1) f_{x_2}(x_2)$ — the full joint density factors.
- **Uncorrelated** if $\gamma_{x_1 x_2} = 0$, equivalently $r_{x_1 x_2} = \mu_{x_1}\mu_{x_2}^*$ — only the second-order moment factors.

> **Important distinction:** Independence implies uncorrelated. The converse is **not** true in general — uncorrelated random variables can still be statistically dependent through higher-order moments. The **only exception** is the Gaussian distribution: for jointly Gaussian random variables, uncorrelated implies independence.

---

## 1.4 Common Random Variable Distributions

> ![Figure 1.3](./CourseADSP2026/Fig/Chapter_2/fig_1_3.png)
>
> *Figure 1.3 (Textbook Fig. 3.3, p. 81): PDF plots of three common distributions — Uniform (flat top, bounded support), Gaussian (bell-shaped, infinite support), and Bernoulli/Exponential. The Gaussian plays a central role because of the Central Limit Theorem.*

### Gaussian (Normal) Distribution

$$\boxed{f_x(x) = \frac{1}{\sqrt{2\pi}\,\sigma_x}\exp\!\left(-\frac{(x-\mu_x)^2}{2\sigma_x^2}\right), \qquad -\infty < x < +\infty}$$

Notation: $x(\zeta) \sim \mathcal{N}(\mu_x, \sigma_x^2)$.

The Gaussian distribution is uniquely characterized by just two parameters $(\mu_x, \sigma_x^2)$ — all higher-order cumulants are zero. This makes Gaussian processes analytically tractable and is why they appear so frequently in statistical signal processing.

**Why Gaussian is ubiquitous — the Central Limit Theorem (CLT):** If $x_1, x_2, \ldots, x_N$ are IID random variables with mean $\mu$ and finite variance $\sigma^2$, then the normalized sum

$$\frac{1}{\sqrt{N}}\sum_{k=1}^{N}(x_k - \mu) \xrightarrow{\mathcal{D}} \mathcal{N}(0, \sigma^2) \quad \text{as } N \to \infty$$

converges in distribution to a Gaussian. In practice, thermal noise is the superposition of countless independent electron-hole pair events — by the CLT, the aggregate is Gaussian to an excellent approximation.

### Uniform Distribution

$$f_x(x) = \frac{1}{b-a}, \quad a \le x \le b; \qquad f_x(x) = 0 \text{ otherwise}$$

Mean $\mu_x = (a+b)/2$; Variance $\sigma_x^2 = (b-a)^2/12$.

Used to model phase angles (uniformly on $[0, 2\pi]$) and to generate other distributions via the **inverse-transform method**: if $U \sim \text{Uniform}[0,1]$, then $x = F_x^{-1}(U)$ has CDF $F_x$.

### Bernoulli Distribution

A discrete-valued distribution with $p_1 = \Pr\lbrace x = +1\rbrace = 1/2$, $p_2 = \Pr\lbrace x = -1\rbrace = 1/2$. Models binary random variables such as coin-flip outcomes. Used as the excitation for Bernoulli white noise.

---

# §2 Stochastic Processes and Their Statistical Description

> 📖 Textbook §3.3 (Discrete-Time Stochastic Processes, §3.3.1–§3.3.6)

## 2.1 Definition of a Stochastic Process

A **discrete-time stochastic process** (or random sequence) is a family of random variables indexed by the discrete time $n$:

> **Definition 2.2 (Random Sequence).** A function $x(n, \zeta)$, $n \in \mathbb{Z}$, is a random sequence if, for any fixed time $n_0$, the quantity $x(n_0, \zeta)$ is a random variable.

The complete collection of all possible sequences $\lbrace x(n, \zeta)\rbrace$ for all $\zeta \in S$ is called the **ensemble**. Each individual sequence $x(n, \zeta_k)$ for a specific $\zeta_k$ is one **realization** (or sample function) of the process.

> ![Figure 2.1](./CourseADSP2026/Fig/Chapter_2/fig_2_1.png)
>
> *Figure 2.1 (Textbook Fig. 3.7, p. 98): Graphical description of a random sequence. The abstract sample space $S$ (left) maps to an ensemble of deterministic sequences (right). Each row is one realization $x(n, \zeta_k)$ that could be observed in practice. At a fixed time $n_0$ (vertical slice), we have a random variable $x(n_0, \zeta)$.*

There are four possible interpretations of $x(n, \zeta)$, depending on whether $n$ and $\zeta$ are fixed or variable:

| $n$ | $\zeta$ | Interpretation |
|-----|---------|---------------|
| Fixed $n_0$ | Variable | **Random variable** $x(n_0, \zeta)$ at a snapshot in time |
| Variable | Fixed $\zeta_k$ | **Sample sequence** (single deterministic realization) |
| Both fixed | — | A single **number** $x(n_0, \zeta_k)$ |
| Both variable | — | The **stochastic process** itself |

**Compact notation:** We drop $\zeta$ and simply write $x(n)$ to denote either the process or a single realization, with meaning clear from context.

---

## 2.2 Ensemble Averages (First- and Second-Order Statistics)

**Ensemble averages** are defined by freezing time and averaging over all realizations:

### Mean and Variance

$$\mu_x(n) = E\lbrace x(n)\rbrace = \int_{-\infty}^{+\infty} x\, f_x(x; n)\, dx$$

$$\sigma_x^2(n) = E\lbrace \lvert x(n) - \mu_x(n)\rvert^2\rbrace = E\lbrace \lvert x(n)\rvert^2\rbrace - \lvert\mu_x(n)\rvert^2$$

Both $\mu_x(n)$ and $\sigma_x^2(n)$ are, in general, functions of time $n$.

### Autocorrelation and Autocovariance

$$\boxed{r_{xx}(n_1, n_2) = E\lbrace x(n_1)\, x^*(n_2)\rbrace}$$

$$\gamma_{xx}(n_1, n_2) = E\lbrace [x(n_1)-\mu_x(n_1)][x(n_2)-\mu_x(n_2)]^*\rbrace = r_{xx}(n_1, n_2) - \mu_x(n_1)\mu_x^*(n_2)$$

The autocorrelation $r_{xx}(n_1, n_2)$ is a function on a **two-dimensional** grid of time indices.

### Cross-Correlation and Cross-Covariance

For two processes $x(n)$ and $y(n)$ defined on the same probability space:

$$r_{xy}(n_1, n_2) = E\lbrace x(n_1)\, y^*(n_2)\rbrace$$

$$\gamma_{xy}(n_1, n_2) = E\lbrace [x(n_1) - \mu_x(n_1)][y(n_2) - \mu_y(n_2)]^*\rbrace$$

$$\rho_{xy}(n_1, n_2) = \frac{\gamma_{xy}(n_1, n_2)}{\sigma_x(n_1)\sigma_y(n_2)} \quad \text{(normalized, } \lvert\rho\rvert\le 1\text{)}$$

### Special Process Types Defined by Their Statistics

| Type | Definition | Key Equation |
|------|-----------|-------------|
| **Independent** | All sample values are mutually independent | $f_x(x_1,\ldots,x_k; n_1,\ldots,n_k) = \prod_i f_i(x_i; n_i)$ |
| **Uncorrelated** | Zero cross-covariance for $n_1 \ne n_2$ | $\gamma_x(n_1, n_2) = \sigma_x^2(n_1)\delta(n_1-n_2)$ |
| **Orthogonal** | Zero cross-correlation for $n_1 \ne n_2$ | $r_x(n_1, n_2) = E\lbrace \lvert x(n_1)\rvert^2\rbrace\delta(n_1-n_2)$ |
| **Gaussian** | All finite-order joint distributions are Gaussian | Fully characterized by mean and correlation |

> **Relationship between the types:** Independent $\Rightarrow$ uncorrelated $\Rightarrow$ orthogonal (for zero-mean processes). The converses do not hold in general, but for **Gaussian** processes, uncorrelated $\Leftrightarrow$ independent.

---

## 2.3 Gaussian Random Sequences

If all kth-order joint distributions of a process $x(n)$ are jointly Gaussian for every $k$ and every choice of time indices $(n_1, \ldots, n_k)$, then $x(n)$ is called a **Gaussian random sequence**.

Its joint density for any $k$-sample snapshot $\mathbf{x} = [x(n_1), \ldots, x(n_k)]^T$ is:

$$f_x(\mathbf{x}; n_1, \ldots, n_k) = \frac{1}{(2\pi)^{k/2}\lvert\mathbf{R}\rvert^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^H \mathbf{R}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu}$ is the mean vector and $\mathbf{R}$ is the covariance matrix at those time indices.

**Critical property:** A Gaussian process is **completely characterized by its first- and second-order statistics** — the mean $\mu_x(n)$ and the correlation $r_{xx}(n_1, n_2)$. No higher-order statistics are needed. This is why Gaussian process models are so widely used: they are tractable while still capturing the key dependence structure.

---

## 2.4 Stationarity

### Strict-Sense Stationarity (SSS)

> **Definition 2.3 (Stationary of Order $N$).** A process $x(n)$ is stationary of order $N$ if its $N$-th order distribution function is shift-invariant:
>
> $$f_x(x_1, \ldots, x_N;\, n_1, \ldots, n_N) = f_x(x_1, \ldots, x_N;\, n_1+k, \ldots, n_N+k) \quad \forall k$$
>
> If this holds for all $N = 1, 2, \ldots$, the process is **strict-sense stationary (SSS)**.

SSS is a very strong condition — it requires all statistics at all orders to be time-invariant. An IID sequence is SSS.

### Wide-Sense Stationarity (WSS)

For most engineering applications, a weaker form of stationarity is both sufficient and practically verifiable:

> **Definition 2.4 (Wide-Sense Stationarity).** A random signal $x(n)$ is **wide-sense stationary (WSS)** if:
> 1. Its mean is constant: $E\lbrace x(n)\rbrace = \mu_x$ for all $n$
> 2. Its variance is constant: $\text{var}[x(n)] = \sigma_x^2$ for all $n$
> 3. Its autocorrelation depends only on the lag $l = n_1 - n_2$:
>
> $$\boxed{r_x(n_1, n_2) = r_x(n_1 - n_2) = r_x(l) = E\lbrace x(n+l)\, x^*(n)\rbrace}$$

**Consequence:** For a WSS process, the two-dimensional autocorrelation collapses to a one-dimensional function of lag $l$ alone. The autocovariance is then:

$$\gamma_x(l) = r_x(l) - \lvert\mu_x\rvert^2$$

The **two jointly WSS processes** $x(n)$ and $y(n)$ have a cross-correlation that depends only on lag:

$$r_{xy}(l) = E\lbrace x(n)\, y^*(n-l)\rbrace$$

> **SSS vs. WSS:** SSS implies WSS, but WSS does not generally imply SSS. The exception is Gaussian processes: for them, WSS $\Leftrightarrow$ SSS, because a Gaussian distribution is completely determined by its first- and second-order moments.

### Properties of the WSS Autocorrelation Sequence

For a WSS process $x(n)$, the autocorrelation sequence $r_x(l)$ satisfies:

**Property 1 (Maximum at zero lag):**
$$\boxed{r_x(0) = \sigma_x^2 + \lvert\mu_x\rvert^2 \ge 0 \quad \text{and} \quad r_x(0) \ge \lvert r_x(l)\rvert \quad \text{for all } l}$$

$r_x(0)$ is the **total average power** — the DC component $\lvert\mu_x\rvert^2$ plus the AC power $\sigma_x^2$. The autocorrelation is always maximum at zero lag.

**Property 2 (Conjugate symmetry):**
$$r_x^*(-l) = r_x(l)$$

For real-valued $x(n)$, this simplifies to $r_x(-l) = r_x(l)$ — the autocorrelation is an **even function**.

**Property 3 (Nonnegative definiteness):** For any $M > 0$ and any vector $\boldsymbol{\alpha} \in \mathbb{R}^M$:

$$\sum_{k=1}^{M}\sum_{l=1}^{M} \alpha_k^* \alpha_l\, r_x(k-l) \ge 0$$

This ensures that the **power spectral density is nonnegative** at all frequencies (proved below in §2.2.6).

---

## 2.5 Correlation Matrices of Stationary Processes

For a WSS process, we frequently work with the **autocorrelation matrix** formed from an $M$-sample snapshot vector:

$$\mathbf{x}(n) = [x(n),\ x(n-1),\ \ldots,\ x(n-M+1)]^T$$

The $M\times M$ correlation matrix $\mathbf{R}_x = E\lbrace \mathbf{x}(n)\mathbf{x}^H(n)\rbrace$ is:

$$\mathbf{R}_x = \begin{bmatrix} r_x(0) & r_x(1) & r_x(2) & \cdots & r_x(M-1) \\ r_x^*(1) & r_x(0) & r_x(1) & \cdots & r_x(M-2) \\ r_x^*(2) & r_x^*(1) & r_x(0) & \cdots & r_x(M-3) \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ r_x^*(M-1) & r_x^*(M-2) & r_x^*(M-3) & \cdots & r_x(0) \end{bmatrix}$$

This matrix is:
- **Hermitian**: $\mathbf{R}_x = \mathbf{R}_x^H$ (because $r_x(l) = r_x^*(-l)$)
- **Toeplitz**: each diagonal (parallel to the main diagonal) has identical entries (because $r_x$ depends only on the lag)
- **Non-negative definite**: $\mathbf{v}^H \mathbf{R}_x \mathbf{v} \ge 0$ for any vector $\mathbf{v}$ (because of Property 3)

> **Why does the Hermitian Toeplitz structure matter?** The Toeplitz structure means that $\mathbf{R}_x$ is fully specified by its first row $[r_x(0), r_x(1), \ldots, r_x(M-1)]$ — only $M$ numbers instead of $M^2$. This enables efficient algorithms like Levinson-Durbin (Chapter 3) to solve $\mathbf{R}_x \mathbf{a} = \mathbf{b}$ in $O(M^2)$ operations instead of the $O(M^3)$ needed for a general matrix.

**Eigenvalue spread and spectral dynamic range (Theorem 3.5):** The eigenvalues $\lambda_i$ of $\mathbf{R}_x$ are bounded by the minimum and maximum of the PSD:

$$\min_\omega R_x(e^{j\omega}) \le \lambda_i \le \max_\omega R_x(e^{j\omega}), \quad i = 1, \ldots, M$$

A "flat" power spectrum (white noise) gives all equal eigenvalues; a narrowband process gives widely spread eigenvalues. The **condition number** $\lambda_\text{max}/\lambda_\text{min}$ (eigenvalue spread) directly affects the convergence rate of adaptive algorithms (Chapter 7) — a large spread causes slow convergence with LMS.

---

## 2.6 Ergodicity

A fundamental practical problem: in real applications, we never observe the entire ensemble — we have only **one realization** of the process. Can we still estimate ensemble averages from a single finite-length record?

### Ensemble Averages vs. Time Averages

**Ensemble average** (theoretical): freeze time $n_0$, average over all $\zeta$:
$$\mu_x(n_0) = E\lbrace x(n_0)\rbrace \quad \text{(requires the ensemble)}$$

**Time average** (practical): fix the realization $\zeta_k$, average over time:
$$\langle x(n)\rangle = \lim_{N\to\infty} \frac{1}{2N+1} \sum_{n=-N}^{N} x(n, \zeta_k) \quad \text{(only one realization needed)}$$

Any time average is itself a random variable (it depends on which realization $\zeta_k$ was observed).

### Ergodic Processes

> **Definition 2.5 (Ergodic in the Mean).** $x(n)$ is ergodic in the mean if $\langle x(n)\rangle = E\lbrace x(n)\rbrace$, i.e., time averaging equals ensemble averaging.

> **Definition 2.6 (Ergodic in Correlation).** $x(n)$ is ergodic in correlation if
>
> $$\langle x(n)x^{\ast}(n-l)\rangle = E\lbrace x(n)x^{\ast}(n-l)\rbrace = r_x(l).$$

**Physical interpretation:** Ergodicity says that a single typical realization, observed long enough, visits all "states" of the process with the same frequency as the full ensemble would at any fixed time. A random sequence is ergodic in the mean if:

$$\lim_{N\to\infty} E\!\left\lbrace \left\lvert\frac{1}{2N+1}\sum_{n=-N}^{N} x(n) - \mu_x\right\rvert^2\right\rbrace = 0 \qquad \text{(mean-square convergence)}$$

This requires that the **autocovariance sequence decays** sufficiently fast: $\sum_{l=-\infty}^{\infty} \lvert\gamma_x(l)\rvert < \infty$.

> **Stationarity vs. Ergodicity:**
> - **Stationarity** ensures that the statistics of $x(n)$ do not change over time.
> - **Ergodicity** additionally ensures that any single realization, observed long enough, reflects those statistics.
>
> WSS does **not** imply ergodicity. Example: $x(n) = c$, where $c$ is a random constant drawn once and then held fixed forever. $x(n)$ is WSS (mean $E\lbrace c\rbrace$ is constant, autocorrelation $r_x(l) = E\lbrace c^2\rbrace$ doesn't depend on $l$), but it is **not** ergodic in the mean — the time average of any single realization is just $c$ itself, not $E\lbrace c\rbrace$.

In practice, however, almost all stationary processes encountered in engineering applications are also ergodic. So from now on we will use ensemble averages and time averages interchangeably when the process is WSS.

---

## 2.7 White Noise

The simplest possible random sequence is one with no temporal structure:

> **Definition 2.7 (White Noise).** A process $w(n)$ is a (second-order) white noise with mean $\mu_w$ and variance $\sigma_w^2$, written $w(n) \sim \mathrm{WN}(\mu_w, \sigma_w^2)$, if and only if:
>
> $$\boxed{r_w(l) = E\lbrace w(n)w^*(n-l)\rbrace = \sigma_w^2\, \delta(l)}$$

The autocorrelation is nonzero **only at zero lag** — any two different samples are uncorrelated. Equivalently, the **power spectral density is flat**:

$$\boxed{R_w(e^{j\omega}) = \sigma_w^2, \qquad -\pi \le \omega \le \pi}$$

The name "white" is analogous to white light: all frequencies contribute equally.

**Important variants:**
- **White Gaussian Noise (WGN)**: $w(n) \sim \mathrm{WGN}(\mu_w, \sigma_w^2)$ — additionally, $w(n)$ is Gaussian at each $n$.
- **IID noise** (strict white noise): $w(n) \sim \mathrm{IID}(\mu_w, \sigma_w^2)$ — all samples are statistically independent and identically distributed. IID $\Rightarrow$ WN (because independence implies uncorrelated). WGN is IID.

White noise is the fundamental **building block** for constructing more complex processes. By driving a linear filter with white noise, we can generate a process with any desired autocorrelation structure (§2.3) and spectral shape (§2.4).

---

## 2.8 Power Spectral Density (Wiener–Khinchin Theorem)

For a zero-mean WSS process with absolutely summable autocorrelation $\sum_l \lvert r_x(l)\rvert < \infty$, the **power spectral density (PSD)** is defined as the DTFT of the autocorrelation sequence:

$$\boxed{R_x(e^{j\omega}) = \sum_{l=-\infty}^{\infty} r_x(l)\, e^{-j\omega l}} \qquad \text{(Wiener–Khinchin theorem)}$$

and the inverse relation:

$$r_x(l) = \frac{1}{2\pi}\int_{-\pi}^{\pi} R_x(e^{j\omega})\, e^{j\omega l}\, d\omega$$

The total average power equals the area under the PSD:

$$E\lbrace \lvert x(n)\rvert^2\rbrace = r_x(0) = \frac{1}{2\pi}\int_{-\pi}^{\pi} R_x(e^{j\omega})\, d\omega$$

### Properties of the PSD

**Property 1 (Real-valued):** $R_x(e^{j\omega}) \in \mathbb{R}$ for all $\omega$ (because $r_x(l) = r_x^*(-l)$ implies its DTFT is real). For real-valued $x(n)$, additionally $R_x(e^{j\omega}) = R_x(e^{-j\omega})$ — the PSD is an even function.

**Property 2 (Non-negative):**
$$R_x(e^{j\omega}) \ge 0 \quad \forall\omega$$

This follows from the nonnegative definiteness of $r_x(l)$ and has a physical meaning: power cannot be negative at any frequency. A useful derivation: apply a narrow bandpass filter $H(e^{j\omega}) = 1$ near $\omega_c$ and $0$ elsewhere; the output power is $\approx R_x(e^{j\omega_c})\cdot\Delta\omega/\pi \ge 0$, so $R_x(e^{j\omega_c}) \ge 0$.

**Property 3 (Power interpretation):** $R_x(e^{j\omega})\, d\omega/(2\pi)$ is the fraction of total average power in the frequency band $[\omega, \omega+d\omega]$.

### Complex Spectral Density (z-Domain)

The z-transform of the autocorrelation sequence:

$$R_x(z) = \sum_{l=-\infty}^{\infty} r_x(l)\, z^{-l}$$

On the unit circle: $R_x(e^{j\omega}) = R_x(z)\big\vert_{z=e^{j\omega}}$.

From the conjugate symmetry $r_x(l) = r_x^*(-l)$, for real $r_x(l)$:

$$R_x(z) = R_x\!\left(\frac{1}{z}\right)$$

This **palindrome property** will be central to spectral factorization (§2.4): the poles and zeros of $R_x(z)$ appear in **reciprocal pairs** $\lbrace z_0, 1/z_0\rbrace$.

**Example.** Consider $r_x(l) = a^{\lvert l\rvert}$, $\lvert a\rvert < 1$. Then:

$$R_x(e^{j\omega}) = \frac{1-a^2}{1 + a^2 - 2a\cos\omega}$$

This is a **real, nonnegative, even** function of $\omega$, as required. At $\omega = 0$: $R_x(1) = \frac{1-a^2}{(1-a)^2} = \frac{1+a}{1-a}$, which is large when $a \to 1$ (low-frequency dominated, "colored" spectrum). At $\omega = \pi$: $R_x(-1) = \frac{1-a^2}{(1+a)^2} = \frac{1-a}{1+a}$, which is small — the process has little high-frequency content.

### Cross-Power Spectral Density

$$R_{xy}(e^{j\omega}) = \sum_{l=-\infty}^{\infty} r_{xy}(l)\, e^{-j\omega l}$$

Note: $R_{xy}(e^{j\omega})$ is **complex-valued** in general. The **coherence function** and **magnitude-squared coherence (MSC)**:

$$G_{xy}(e^{j\omega}) = \frac{R_{xy}(e^{j\omega})}{\sqrt{R_x(e^{j\omega}) R_y(e^{j\omega})}}, \qquad 0 \le \lvert G_{xy}(e^{j\omega})\rvert^2 \le 1$$

MSC $= 1$ means perfect linear correlation at that frequency; MSC $= 0$ means no correlation. MSC generalizes the concept of a correlation coefficient to the frequency domain.

### Harmonic Processes

A **harmonic process** consists of a sum of sinusoids with random phases:

$$x(n) = \sum_{k=1}^{M} A_k \cos(\omega_k n + \phi_k)$$

where $\phi_k \sim \text{Uniform}[0, 2\pi]$ are mutually independent. The mean is zero, and the autocorrelation is:

$$r_x(l) = \frac{1}{2}\sum_{k=1}^{M} A_k^2 \cos(\omega_k l)$$

The PSD consists of **impulses** (line spectrum) at frequencies $\pm\omega_k$:

$$R_x(e^{j\omega}) = \pi \sum_{k=-M}^{M} A_k^2\, \delta(\omega - \omega_k) \quad \text{(with sign convention)}$$

> ![Figure 2.2](./CourseADSP2026/Fig/Chapter_2/fig_2_2.png)
>
> *Figure 2.2 (Textbook Fig. 3.9, p. 112): Time and frequency-domain description of the harmonic process in Example 3.3.5 — $x(n) = \cos(0.1\pi n + \phi_1) + 2\sin(1.5n + \phi_2)$. (a) A sample realization; (b) the line spectrum showing impulse amplitudes at discrete frequencies $\pm 0.1\pi$ and $\pm 1.5$; (c) the corresponding continuous power spectrum.*

---

# §3 Random Processes through Filters

> 📖 Textbook §3.4 (Linear Systems with Stationary Random Inputs, §3.4.1–§3.4.2)

## 3.1 Time-Domain Analysis

### Setup

When a random process $x(n)$ is applied as the input to a BIBO-stable LTI system with impulse response $h(n)$, each realization $x(n, \zeta_k)$ produces a deterministic output $y(n, \zeta_k) = h(n) * x(n, \zeta_k)$. The collection of all output realizations forms the **output stochastic process** $y(n)$.

The fundamental theorem guarantees well-posedness: if $x(n)$ is stationary with $E\lbrace \lvert x(n)\rvert\rbrace < \infty$ and the system is BIBO-stable ($\sum_n \lvert h(n)\rvert < \infty$), then the output $y(n)$ converges absolutely with probability 1 and is also stationary.

### Output Mean

$$\boxed{\mu_y = E\lbrace y(n)\rbrace = \sum_{k=-\infty}^{\infty} h(k)\, E\lbrace x(n-k)\rbrace = \mu_x \sum_{k=-\infty}^{\infty} h(k) = \mu_x\, H(e^{j0})}$$

The DC component of the output is the input DC scaled by the filter's DC gain $H(e^{j0})$.

### Input-Output Cross-Correlation

Postmultiply both sides of $y(n) = \sum_k h(k) x(n-k)$ by $x^{\ast}(n-l)$ and take expectation:

$$r_{yx}(l) = E\lbrace y(n)\, x^{\ast}(n-l)\rbrace = \sum_{k} h(k)\, E\lbrace x(n-k)\, x^{\ast}(n-l)\rbrace = \sum_k h(k)\, r_x(l-k)$$

$$\boxed{r_{yx}(l) = h(l) \ast r_x(l)}$$

Similarly, $r_{xy}(l) = h^{\ast}(-l) \ast r_x(l)$.

### Output Autocorrelation

Continuing the derivation by multiplying $y^{\ast}(n-l)$ from the right:

$$r_y(l) = h(l) \ast r_{xy}(l) = h(l) \ast h^{\ast}(-l) \ast r_x(l) = r_h(l) \ast r_x(l)$$

where

$$r_h(l) = h(l) \ast h^{\ast}(-l) = \sum_n h(n)\, h^{\ast}(n-l)$$

is the **system correlation sequence** (autocorrelation of the impulse response).

$$\boxed{r_y(l) = r_h(l) \ast r_x(l) = \sum_{k} r_h(k)\, r_x(l-k)}$$

> ![Figure 3.1](./CourseADSP2026/Fig/Chapter_2/fig_3_1.png)
>
> *Figure 3.1 (Textbook Fig. 3.10, p. 117): Equivalent cascade interpretation of autocorrelation filtering. The autocorrelation $r_x(l)$ of the input is "filtered" by a two-stage system $h(l)$ and $h^{\ast}(-l)$ (whose combined impulse response is $r_h(l) = h(l)\ast h^{\ast}(-l)$), yielding the output autocorrelation $r_y(l)$.*

> **Interpretation of Figure 3.1:** The output autocorrelation is *not* simply $r_y(l) = h(l) \ast r_x(l)$. Intuitively: filtering $x(n)$ affects correlations at *both* endpoints of the lag. Formally, the autocorrelation is shaped by the **matched pair** of filters $h(l)$ and $h^{\ast}(-l)$ — one forward in time, one backward — whose combined effect is the autocorrelation of the impulse response $r_h(l)$.

### Output Power (Mean-Square Value)

Setting $l = 0$ in the output autocorrelation formula:

$$\boxed{P_y = E\lbrace \lvert y(n)\rvert^2\rbrace = r_y(0) = \sum_{k=-\infty}^{\infty} r_h(k)\, r_x(-k) = \mathbf{h}^H \mathbf{R}_x \mathbf{h}}$$

where the last expression is the **quadratic form** for FIR filters with coefficient vector $\mathbf{h} = [h(0), \ldots, h(M-1)]^T$.

---

## 3.2 Frequency-Domain Analysis

Taking the DTFT/z-transform of the time-domain relationships gives the elegant frequency-domain formulas. All results are collected in Table 2.1.

**Key result for the output PSD:**

$$\boxed{R_y(e^{j\omega}) = \lvert H(e^{j\omega})\rvert^2\, R_x(e^{j\omega})}$$

**In the z-domain:**

$$R_y(z) = H(z)\, H^{\ast}\!\left(\frac{1}{z^{\ast}}\right)\, R_x(z)$$

**Cross-PSDs:**

$$R_{yx}(e^{j\omega}) = H(e^{j\omega})\, R_x(e^{j\omega}), \qquad R_{xy}(e^{j\omega}) = H^{\ast}(e^{j\omega})\, R_x(e^{j\omega})$$

**Table 2.1: Second-order statistics of stationary random sequences through LTI systems**

| Quantity | Time Domain | Frequency Domain | z-Domain |
|----------|------------|-----------------|---------|
| Output | $y(n) = h(n) * x(n)$ | — | — |
| Cross (yx) | $r_{yx}(l) = h(l) \ast r_x(l)$ | $R_{yx} = H R_x$ | $R_{yx}(z) = H(z) R_x(z)$ |
| Cross (xy) | $r_{xy}(l) = h^{\ast}(-l) \ast r_x(l)$ | $R_{xy} = H^{\ast} R_x$ | $R_{xy}(z) = H^{\ast}(1/z^{\ast})R_x(z)$ |
| Auto | $r_y(l) = h(l)\ast h^{\ast}(-l)\ast r_x(l)$ | $R_y = \lvert H\rvert^2 R_x$ | $R_y(z) = H(z)H^{\ast}(1/z^{\ast})R_x(z)$ |
| Power | $P_y = \mathbf{h}^H\mathbf{R}_x\mathbf{h}$ | $P_y = \frac{1}{2\pi}\int\lvert H\rvert^2 R_x d\omega$ | — |

> **Key insight from $R_y(e^{j\omega}) = \lvert H(e^{j\omega})\rvert^2 R_x(e^{j\omega})$:**
>
> The output PSD is the input PSD **weighted** by the squared magnitude response of the filter. The filter **shapes** the spectrum. This is the fundamental reason for "colored noise" — white noise ($R_x = \sigma_w^2$, flat) passed through a filter $H(z)$ produces "colored" noise $R_y = \sigma_w^2 \lvert H(e^{j\omega})\rvert^2$ with a spectral shape matching $\lvert H\rvert^2$.
>
> Conversely: if we know $R_x$ and $R_y$, we can identify $\lvert H(e^{j\omega})\rvert^2 = R_y/R_x$ — this is the foundation of **blind deconvolution** and system identification. However, we can only recover the **magnitude** of $H$, not its phase, from the autocorrelations alone. To recover the phase, we need the **cross-spectrum** $R_{xy}$.

### Output Variance Calculation Example

**Example:** White noise $w(n) \sim \mathrm{WN}(0, \sigma_w^2)$ is filtered by $H(z) = 1/(1 - a z^{-1})$, $\lvert a\rvert < 1$ (a first-order IIR).

Output PSD: $R_y(e^{j\omega}) = \sigma_w^2 \lvert H(e^{j\omega})\rvert^2 = \frac{\sigma_w^2}{\lvert 1 - ae^{-j\omega}\rvert^2} = \frac{\sigma_w^2}{1 + a^2 - 2a\cos\omega}$

Output autocorrelation: $r_y(l) = \frac{\sigma_w^2}{1-a^2}\, a^{\lvert l\rvert}$

Output power: $r_y(0) = \frac{\sigma_w^2}{1 - a^2}$

This confirms that the variance of a first-order AR process driven by white noise of variance $\sigma_w^2$ is $\sigma_w^2/(1-a^2)$, which grows as $a \to 1$ (pole approaching the unit circle → near-instability → large variance).

---

# §4 Spectral Factorization

> 📖 Textbook §2.4 (Minimum-Phase and System Invertibility) · §3.5.2 (Whitening and Innovations)

## 4.1 The Central Question

The output PSD formula $R_y(z) = H(z) H^{\ast}(1/z^{\ast}) R_x(z)$ invites a reverse question:

> Given a non-negative PSD $R_x(z)$ (or equivalently, a positive semi-definite autocorrelation sequence $r_x(l)$), can we write it as $R_x(z) = H_+(z) H_+^{\ast}(1/z^{\ast})$ for some **minimum-phase** filter $H_+(z)$?

The answer is **yes** for any rational PSD satisfying the Paley–Wiener condition, and the factorization is **unique** if we require $H_+(z)$ to be minimum-phase (all zeros inside the unit circle). This is the **spectral factorization theorem**.

## 4.2 Pole-Zero Structure of the PSD

From Chapter 1, we know that for real autocorrelation sequences, $R_x(z) = R_x(1/z)$ — poles and zeros appear in **reciprocal pairs** $\lbrace z_0, 1/z_0\rbrace$. Combined with Hermitian symmetry $r_x(l) = r_x(-l)$ (real sequences), poles and zeros also appear in **conjugate pairs** $\lbrace z_0, z_0^{\ast}\rbrace$. Therefore, poles and zeros of $R_x(z)$ come in **quadruples**: $\lbrace z_0,\ z_0^{\ast},\ 1/z_0,\ 1/z_0^{\ast}\rbrace$.

On the unit circle ($\lvert z_0\rvert = 1$), conjugate reciprocals coincide ($z_0^{\ast} = 1/z_0$ for $\lvert z_0\rvert=1$), so zeros on the unit circle appear in conjugate pairs with **even multiplicity** (to maintain $R_x(e^{j\omega}) \ge 0$).

## 4.3 Minimum-Phase Spectral Factor

> **Theorem 2.1 (Spectral Factorization).** Any real-valued PSD $R_x(z)$ of a rational form can be uniquely factored as:
>
> $$\boxed{R_x(z) = \sigma_0^2\, H_+(z)\, H_+^{\ast}\!\left(\frac{1}{z^{\ast}}\right)}$$
>
> where $H_+(z)$ is **minimum-phase** (all zeros and poles strictly inside the unit circle), and $\sigma_0^2 > 0$ is a real positive scalar.

On the unit circle: $R_x(e^{j\omega}) = \sigma_0^2 \lvert H_+(e^{j\omega})\rvert^2$.

**Construction:** From each quadruple $\lbrace z_0, z_0^{\ast}, 1/z_0, 1/z_0^{\ast}\rbrace$:
- Assign the **inner pair** $\lbrace 1/z_0^{\ast}, 1/z_0\rbrace$ to $H_+(z)$ (inside the unit circle)
- The **outer pair** $\lbrace z_0, z_0^{\ast}\rbrace$ goes to $H_+^{\ast}(1/z^{\ast})$ automatically

**Physical meaning of $H_+(z)$:** The minimum-phase spectral factor $H_+(z)$ is the **causal shaping filter** that generates the process $x(n)$ from white noise:

$$x(n) = H_+(z)\, w(n), \quad w(n) \sim \mathrm{WN}(0, \sigma_0^2)$$

Its inverse $1/H_+(z)$ is the **whitening filter** — it converts $x(n)$ back to white noise, because

$$\frac{1}{H_+(z)}\, R_x(z)\, \frac{1}{H_+^{\ast}(1/z^{\ast})} = \frac{1}{H_+(z)}\, \sigma_0^2 H_+(z) H_+^{\ast}(1/z^{\ast})\, \frac{1}{H_+^{\ast}(1/z^{\ast})} = \sigma_0^2$$

## 4.4 Innovations Representation

A **regular** (non-predictable) stationary process $x(n)$ admits an **innovations representation**:

$$\boxed{x(n) = H_+(z)\, e(n)}$$

where $e(n)$ is the **innovations process** (white noise with variance $\sigma_e^2 = \sigma_0^2$) and $H_+(z)$ is the causal minimum-phase spectral factor. The innovations process $e(n)$ represents the truly new, unpredictable information added to the process at each time step.

**White noise processes** are their own innovations — they are completely non-predictable.

**Example of a non-factorizable process:** Any process containing a deterministic sinusoidal component $A_0\cos(\omega_0 n + \phi_0)$ has a PSD with an **impulse** at $\omega_0$. The spectral density has zeros on the unit circle at $z = e^{\pm j\omega_0}$ — but these zeros must have even multiplicity (since $R_x(e^{j\omega}) \ge 0$), meaning the factorization exists formally but $H_+(z)$ has zeros on the unit circle. Such a process is **predictable** (not regular), and the innovations variance $\sigma_e^2 = 0$ — a purely predictable process has zero entropy.

---

# §5 Special Types of Random Processes

> 📖 Textbook §4.1–§4.3 (Linear Signal Models: All-Pole, All-Zero, Pole-Zero); §3.3.6 (Harmonic Processes)

The spectral factorization framework tells us that any regular WSS process can be represented as white noise filtered through a causal minimum-phase filter. Three important special cases correspond to specific filter structures:

## 5.1 ARMA (Autoregressive Moving-Average) Processes

### Definition

An **ARMA($p$, $q$) process** is generated by driving a **pole-zero filter** with white noise:

$$\boxed{x(n) + \sum_{k=1}^{p} a_k x(n-k) = w(n) + \sum_{k=1}^{q} b_k w(n-k)}$$

where $w(n) \sim \mathrm{WN}(0, \sigma_w^2)$. The transfer function is:

$$H(z) = \frac{B(z)}{A(z)} = \frac{1 + \sum_{k=1}^q b_k z^{-k}}{1 + \sum_{k=1}^p a_k z^{-k}} = \frac{B(z)}{A(z)}$$

For stability, all poles of $H(z)$ (roots of $A(z)$) must lie strictly inside the unit circle.

### Power Spectral Density

$$R_x(e^{j\omega}) = \sigma_w^2 \left\lvert\frac{B(e^{j\omega})}{A(e^{j\omega})}\right\rvert^2 = \sigma_w^2 \frac{\lvert B(e^{j\omega})\rvert^2}{\lvert A(e^{j\omega})\rvert^2}$$

### Yule-Walker Equations for ARMA

Multiplying the ARMA difference equation by $x^*(n-l)$ and taking expectations, one can show that for lags $l > q$, the moving-average terms vanish and only the AR coefficients appear:

$$r_x(l) + \sum_{k=1}^{p} a_k r_x(l-k) = 0, \qquad l > q$$

This is a system of linear equations in the unknown AR coefficients $\lbrace a_k\rbrace$ that becomes exploitable for ARMA estimation (Chapter 4).

---

## 5.2 AR (Autoregressive) Processes

### Definition

An **AR($p$) process** is the special case $q = 0$ — an **all-pole model** driven by white noise:

$$\boxed{x(n) = -\sum_{k=1}^{p} a_k\, x(n-k) + w(n), \qquad w(n) \sim \mathrm{WN}(0, \sigma_w^2)}$$

The filter is $H(z) = 1/A(z)$ where $A(z) = 1 + \sum_{k=1}^p a_k z^{-k}$.

### Yule-Walker Equations for AR

Multiplying the AR equation by $x^*(n-l)$ and taking expectations:

$$r_x(l) + \sum_{k=1}^{p} a_k\, r_x(l-k) = \sigma_w^2\, \delta(l), \qquad l = 0, 1, \ldots, p$$

For $l > 0$, $\delta(l) = 0$, giving the **homogeneous Yule-Walker equations**:

$$r_x(l) = -\sum_{k=1}^{p} a_k\, r_x(l-k), \qquad l = 1, 2, \ldots, p$$

In matrix form (setting $l = 1, 2, \ldots, p$):

$$\begin{bmatrix} r_x(0) & r_x(1) & \cdots & r_x(p-1) \\ r_x(1) & r_x(0) & \cdots & r_x(p-2) \\ \vdots & \vdots & \ddots & \vdots \\ r_x(p-1) & r_x(p-2) & \cdots & r_x(0) \end{bmatrix} \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_p \end{bmatrix} = -\begin{bmatrix} r_x(1) \\ r_x(2) \\ \vdots \\ r_x(p) \end{bmatrix}$$

$$\boxed{\mathbf{R}_x \mathbf{a} = -\mathbf{r}_x}$$

This is the **Toeplitz linear system** that Levinson-Durbin solves efficiently in $O(p^2)$ (Chapter 3).

The driving noise variance is recovered from $l=0$:

$$\sigma_w^2 = r_x(0) + \sum_{k=1}^{p} a_k\, r_x(k) = r_x(0) + \mathbf{a}^T\mathbf{r}_x$$

### Power Spectral Density of AR($p$)

$$\boxed{R_x(e^{j\omega}) = \frac{\sigma_w^2}{\lvert A(e^{j\omega})\rvert^2} = \frac{\sigma_w^2}{\left\lvert1 + \sum_{k=1}^p a_k e^{-j\omega k}\right\rvert^2}}$$

The AR spectrum has **peaks at frequencies near the poles** of $H(z)$. This makes AR models excellent for processes with **sharp spectral peaks** (narrowband resonances), such as voiced speech, sonar echoes at discrete angles, or vibration resonance modes.

> **Key use case:** AR processes are the workhorse of **linear prediction** (Chapter 3). Because the Yule-Walker equations are Toeplitz, the AR coefficients can be estimated efficiently from the autocorrelation using the Levinson-Durbin algorithm. This will be the bridge between Chapter 2 (analysis) and Chapter 3 (prediction and estimation).

---

## 5.3 MA (Moving-Average) Processes

### Definition

An **MA($q$) process** is the special case $p = 0$ — an **all-zero (FIR filter) model** driven by white noise:

$$\boxed{x(n) = w(n) + \sum_{k=1}^{q} b_k\, w(n-k) = B(z)\, w(n), \qquad w(n) \sim \mathrm{WN}(0, \sigma_w^2)}$$

### Power Spectral Density of MA($q$)

$$R_x(e^{j\omega}) = \sigma_w^2\, \lvert B(e^{j\omega})\rvert^2 = \sigma_w^2 \left\lvert1 + \sum_{k=1}^{q} b_k e^{-j\omega k}\right\rvert^2$$

The MA spectrum has **notches at frequencies near the zeros** of $B(z)$.

### Autocorrelation of MA($q$)

A crucial property: the autocorrelation of an MA($q$) process is **exactly zero for lags $\lvert l\rvert > q$**:

$$r_x(l) = \begin{cases} \sigma_w^2 \sum_{k=0}^{q-\lvert l\rvert} b_k b_{k+\lvert l\rvert}^* & \lvert l\rvert \le q \\ 0 & \lvert l\rvert > q \end{cases}$$

(where $b_0 = 1$). This **finite memory** property makes MA processes well-suited for modeling processes where sample-to-sample dependence dies out quickly.

**Non-uniqueness:** Unlike AR models, MA models are not unique — the same PSD $\lvert B(e^{j\omega})\rvert^2$ can be generated by the minimum-phase factorization $H_+(z)$ or any all-pass-modified version $H_+(z) \cdot H_{ap}(z)$. The **minimum-phase MA** model is unique, but there are $2^q$ possible MA models (each zero of $B(z)$ can be inside or outside the unit circle). This non-uniqueness makes MA estimation harder than AR estimation.

---

## 5.4 ARMA, AR, MA: Summary and Comparison

| Property | AR($p$) | MA($q$) | ARMA($p$,$q$) |
|----------|---------|---------|--------------|
| Transfer function $H(z)$ | $1/A(z)$ (all-pole) | $B(z)$ (all-zero) | $B(z)/A(z)$ |
| Poles | $p$ (inside unit circle) | None (0) | $p$ |
| Zeros | None (0) | $q$ | $q$ |
| Autocorrelation $r_x(l)$ | Infinite, exponentially decaying | **Exactly zero for $\lvert l\rvert>q$** | Infinite, decays to $0$ |
| PSD shape | Peaks (narrowband good) | Notches (wideband good) | Peaks and notches |
| Estimation via | Yule-Walker / Levinson (efficient) | Spectral factorization / Durbin | Iterative / MYWE |
| Universality | Can approximate any PSD with enough poles | Can approximate any PSD with enough zeros | Most general, fewest parameters needed |

---

# §6 Basic Orthogonal Transforms

> 📖 Textbook §3.5 (Whitening and Innovations; KL Transform: §3.5.1, §3.5.3)

## 6.1 Hilbert Space and Orthogonal Transforms

### Hierarchy of Vector Spaces

Signal processing transforms live in a hierarchy of mathematical spaces:

$$\text{Linear vector space} \subset \text{Normed linear space} \subset \text{Inner product space} \subset \text{Hilbert space}$$

- **Linear vector space:** Closed under addition and scalar multiplication.
- **Normed space:** Equipped with a length (norm) $\|\mathbf{x}\|$.
- **Inner product space:** Equipped with an inner product $\langle \mathbf{x}, \mathbf{y}\rangle$ that induces a norm and notion of angle/orthogonality.
- **Hilbert space:** A complete inner product space — every Cauchy sequence converges within the space. The space $\ell^2$ of square-summable sequences and $L^2[-\pi, \pi]$ are Hilbert spaces.

The **Euclidean space** $\mathbb{R}^M$ (or $\mathbb{C}^M$) is the finite-dimensional special case. Every finite-dimensional inner product space is a Hilbert space.

### Orthogonal Transforms in Finite Dimensions

An **orthogonal (unitary) transform** $\mathbf{y} = \mathbf{A}\mathbf{x}$ satisfies $\mathbf{A}^H \mathbf{A} = \mathbf{I}$, which implies:
- **Parseval's theorem:** $\|\mathbf{y}\|^2 = \|\mathbf{x}\|^2$ — the $L^2$-norm (total energy) is preserved.
- **Unique invertibility:** $\mathbf{x} = \mathbf{A}^H \mathbf{y}$ — the inverse transform is $\mathbf{A}^H$.
- **Decorrelation potential:** Can diagonalize the covariance matrix if $\mathbf{A}$ is chosen as the eigenmatrix of $\mathbf{R}_x$.

### Advantages of Orthogonal Transforms

1. **Energy preservation:** The total signal energy is the same in the original and transformed domain — no energy is created or destroyed by the transformation.
2. **Decorrelation:** The transformed components can be made uncorrelated (independent for Gaussian processes), simplifying analysis and processing.
3. **Optimal truncation:** Representing the signal using only the most energetically significant transformed components minimizes the mean-squared reconstruction error (the KL transform achieves this optimally).
4. **Fast algorithms:** Some special orthogonal transforms (DFT, DCT) have $O(N\log N)$ fast algorithms.

---

## 6.2 KL Transform (Karhunen-Loève Transform)

### Motivation

Given a random vector $\mathbf{x}$ with covariance matrix $\boldsymbol{\Sigma}_x$, we seek a unitary transform $\mathbf{w} = \mathbf{A}^H \mathbf{x}$ such that the components of $\mathbf{w}$ are **mutually uncorrelated**. Uncorrelated components are much easier to process independently.

### Derivation: Diagonalizing the Covariance Matrix via Eigendecomposition

The covariance matrix $\boldsymbol{\Sigma}_x$ is Hermitian and non-negative definite, so it admits an eigendecomposition:

$$\boldsymbol{\Sigma}_x = \mathbf{Q}_x \boldsymbol{\Lambda}_x \mathbf{Q}_x^H$$

where $\mathbf{Q}_x = [\mathbf{q}_1, \ldots, \mathbf{q}_M]$ is the unitary eigenmatrix (columns are orthonormal eigenvectors) and $\boldsymbol{\Lambda}_x = \mathrm{diag}\lbrace \lambda_1, \ldots, \lambda_M\rbrace$ has real non-negative eigenvalues.

Choose the **KL transform matrix** $\mathbf{A} = \mathbf{Q}_x$, i.e., $\mathbf{w} = \mathbf{Q}_x^H (\mathbf{x} - \boldsymbol{\mu}_x)$. Then:

$$\boldsymbol{\Sigma}_w = E\lbrace \mathbf{w}\mathbf{w}^H\rbrace = \mathbf{Q}_x^H \boldsymbol{\Sigma}_x \mathbf{Q}_x = \boldsymbol{\Lambda}_x = \mathrm{diag}\lbrace \lambda_1, \ldots, \lambda_M\rbrace$$

The transformed vector $\mathbf{w}$ has:
- **Zero mean:** $E\lbrace \mathbf{w}\rbrace = \mathbf{0}$
- **Uncorrelated components:** $E\lbrace w_i w_j^*\rbrace = 0$ for $i \ne j$
- **Component variances equal to eigenvalues:** $E\lbrace \lvert w_i\rvert^2\rbrace = \lambda_i$

> ![Figure 6.1](./CourseADSP2026/Fig/Chapter_2/fig_6_1.png)
>
> *Figure 6.1 (Textbook Fig. 3.11, p. 126): Geometric interpretation of the KL (orthonormal) transformation in two dimensions. The original coordinate axes ($x_1$, $x_2$) are rotated to align with the eigenvectors $\mathbf{q}_1$, $\mathbf{q}_2$ of $\boldsymbol{\Sigma}_x$ (principal axes of the distribution). The elongated ellipse in the original frame becomes an axis-aligned ellipse in the $\mathbf{w}$-frame; the eigenvalues $\lambda_1 \ge \lambda_2$ give the variance along each principal axis.*

### Optimal Reduced-Basis Representation (PCA)

Suppose we approximate $\mathbf{x}$ using only the first $K < M$ basis vectors:

$$\hat{\mathbf{x}} = \boldsymbol{\mu}_x + \sum_{i=1}^{K} w_i \mathbf{q}_i, \qquad w_i = \mathbf{q}_i^H(\mathbf{x} - \boldsymbol{\mu}_x)$$

The resulting MSE is:

$$E_K = E\!\left\lbrace \|\mathbf{x} - \hat{\mathbf{x}}\|^2\right\rbrace = \sum_{i=K+1}^{M} \lambda_i$$

To minimize $E_K$, retain the $K$ eigenvectors with the **largest** eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_M$. This is **Principal Component Analysis (PCA)** — the KL transform is the optimal linear dimensionality reduction in the MSE sense.

> ![Figure 6.2](./CourseADSP2026/Fig/Chapter_2/fig_6_2.png)
>
> *Figure 6.2 (Textbook Fig. 3.13, p. 132): Signal compression using the DKLT. The transmitter applies the KL transform to $x(n)$, retains only the $K$ largest-energy components via a reduced-basis selection scheme, and transmits the coded signal $\hat{w}(n)$. The receiver reconstructs $\hat{x}(n)$ via the inverse DKLT. If $K \ll M$, significant compression is achieved with bounded MSE $= \sum_{i=K+1}^{M}\lambda_i$.*

### Properties of the KL Transform (Advantages and Disadvantages)

| **Advantages** | **Disadvantages** |
|---------------|-----------------|
| Provides optimal decorrelation of the random vector | No fixed basis — depends on the data statistics |
| Minimizes truncation MSE among all orthogonal transforms | Requires estimating the covariance matrix $\boldsymbol{\Sigma}_x$ from data |
| Concentrates energy into the fewest components (energy compaction) | No fast algorithm (requires $O(M^2)$ to $O(M^3)$ computation) |
| For Gaussian processes, also provides independence (not just uncorrelation) | Computationally expensive for large $M$ |

### Isotropic (Whitening) Transformation

An extension of the KL transform that normalizes all component variances to 1:

$$\mathbf{y} = \boldsymbol{\Lambda}_x^{-1/2} \mathbf{Q}_x^H (\mathbf{x} - \boldsymbol{\mu}_x)$$

The result: $\boldsymbol{\Sigma}_y = \mathbf{I}$ — all components have unit variance, are uncorrelated, and are isotropically distributed (hence the name). This transformation is also called **whitening**, because it transforms a colored random vector into an uncorrelated, unit-variance vector (analogous to white noise).

> ![Figure 6.3](./CourseADSP2026/Fig/Chapter_2/fig_6_3.png)
>
> *Figure 6.3 (Textbook Fig. 3.12, p. 127): Isotropic (whitening) transformation in two dimensions. After the KL rotation (Figure 6.1), an additional axis-scaling step $\boldsymbol{\Lambda}_x^{-1/2}$ normalizes the variance in each direction to 1. The resulting distribution becomes a circle — isotropic, invariant to any further rotation.*

---

## 6.3 Discrete Cosine Transform (DCT)

### Motivation

The KL transform is optimal but data-dependent. An open question is: does any **fixed** (data-independent) orthogonal transform come close to the KL transform in terms of decorrelation performance?

The answer is yes — under a common signal model:

> **Theorem:** For a first-order AR process $x(n) = a\, x(n-1) + w(n)$ (Markov-1 process) with correlation coefficient $a \to 1$, the KL transform converges to the **DCT-II**.

Since natural signals (speech, images) are well-modeled as Markov-1 with $a$ close to 1, the DCT provides near-optimal decorrelation for these signals — and it has a fast $O(N\log N)$ algorithm.

### DCT-II Definition

The most widely used variant is **DCT-II** (used in JPEG, HEVC, MP3):

$$W(k) = \sqrt{\frac{2}{N}}\, c(k) \sum_{n=0}^{N-1} x(n)\cos\!\left(\frac{(2n+1)k\pi}{2N}\right), \quad k = 0, 1, \ldots, N-1$$

where $c(0) = 1/\sqrt{2}$, $c(k) = 1$ for $k \ge 1$.

**Matrix form:** $\mathbf{W} = \mathbf{C}_N \mathbf{x}$, where $\mathbf{C}_N$ is the $N\times N$ DCT matrix. The inverse DCT is $\mathbf{x} = \mathbf{C}_N^T \mathbf{W}$ (since $\mathbf{C}_N^T \mathbf{C}_N = \mathbf{I}$ — the DCT is an **orthogonal** transform over the reals).

### Four Variants of DCT

| Type | Defining property | Period 2N extension type |
|------|------------------|-----------------------|
| DCT-I | Even symmetry, includes both endpoints | Symmetric extension around both endpoints |
| DCT-II (most common) | Even symmetry, midpoint at half-sample | Symmetric extension, half-sample offset |
| DCT-III | Inverse of DCT-II | Mirror at half-sample (right side only) |
| DCT-IV | Even symmetry, half-sample offset on both ends | Symmetric, both half-sample offsets |

### Advantages over KL Transform

- **Fixed basis:** The cosine functions $\lbrace \cos((2n+1)k\pi/(2N))\rbrace$ do not depend on the signal statistics.
- **Real-valued:** Unlike the DFT (which uses complex exponentials), the DCT operates entirely in the real domain — more efficient for real signals.
- **Fast algorithm:** Can be computed as an $N$-point or $2N$-point FFT in $O(N\log N)$ time.
- **Near-optimal for correlated signals:** Within 1–2 dB of the optimal KL transform for typical image/audio signals.

**Relation to DFT:** DCT-II of $x(n)$ equals the real part of the DFT of a symmetrically extended version of $x(n)$. This connection enables fast computation via the FFT.

---

# §7 Basic Parametric Estimation Methods

> 📖 Textbook §3.6 (Principles of Estimation Theory, §3.6.1–§3.6.3); §8.1–§8.2 (Least-Squares)

## 7.1 Performance of Estimators

### The Estimation Problem

Let $\boldsymbol{\theta} = [\theta_1, \ldots, \theta_P]^T$ be a vector of **unknown deterministic parameters**. We observe a vector $\mathbf{x} = [x(0), x(1), \ldots, x(N-1)]^T$ that depends statistically on $\boldsymbol{\theta}$. Our goal is to find a function $\hat{\boldsymbol{\theta}} = g(\mathbf{x})$ that estimates $\boldsymbol{\theta}$ as accurately as possible.

### Bias

$$b(\hat{\theta}_i) = E\lbrace \hat{\theta}_i\rbrace - \theta_i$$

An estimator is **unbiased** if $b(\hat{\theta}_i) = 0$ — on average, it hits the correct value. Otherwise it is **biased**. An estimator is **asymptotically unbiased** if $b(\hat{\theta}_i) \to 0$ as $N \to \infty$.

### Variance and MSE

$$\text{var}(\hat{\theta}_i) = E\lbrace \lvert\hat{\theta}_i - E\lbrace \hat{\theta}_i\rbrace\rvert^2\rbrace, \qquad \text{MSE}(\hat{\theta}_i) = E\lbrace \lvert\hat{\theta}_i - \theta_i\rvert^2\rbrace$$

$$\text{MSE} = \text{var}(\hat{\theta}_i) + b^2(\hat{\theta}_i)$$

A good estimator has small MSE — which requires both small bias and small variance. There is typically a **bias-variance tradeoff**: reducing one can increase the other.

### Cramér-Rao Lower Bound (CRLB)

A fundamental limit on how accurately any **unbiased** estimator can perform:

> **Theorem 2.2 (Cramér-Rao Inequality).** The variance of any unbiased estimator $\hat{\theta}_i$ is bounded below:
>
> $$\text{var}(\hat{\theta}_i) \ge [\mathbf{J}^{-1}(\boldsymbol{\theta})]_{ii}$$
>
> where $\mathbf{J}(\boldsymbol{\theta})$ is the **Fisher information matrix** with elements:
>
> $$J_{ij}(\boldsymbol{\theta}) = E\!\left\lbrace \frac{\partial \ln f_x(\mathbf{x};\boldsymbol{\theta})}{\partial \theta_i}\cdot\frac{\partial \ln f_x(\mathbf{x};\boldsymbol{\theta})}{\partial \theta_j}\right\rbrace = -E\!\left\lbrace \frac{\partial^2 \ln f_x(\mathbf{x};\boldsymbol{\theta})}{\partial \theta_i \partial \theta_j}\right\rbrace$$

The CRLB gives a performance benchmark: if we can achieve $\text{var}(\hat{\theta}) = [\mathbf{J}^{-1}]_{ii}$, the estimator is **efficient** (achieves the MVUE — minimum variance unbiased estimator). In general, achieving the CRLB requires the score function to have a specific form related to the estimator.

### Consistency

An estimator $\hat{\theta}(x_1, \ldots, x_N)$ is **mean-square consistent** if:

$$\lim_{N\to\infty} E\lbrace \lvert\hat{\theta}_N - \theta\rvert^2\rbrace = 0$$

This requires both bias $\to 0$ and variance $\to 0$ as $N \to \infty$. A consistent estimator becomes arbitrarily accurate with enough data.

---

## 7.2 Sample Estimates of Random Signal Statistics (Method of Moments)

### Sample Mean

Given $N$ samples $\lbrace x(0), x(1), \ldots, x(N-1)\rbrace$ of an ergodic WSS process:

$$\hat{\mu}_x = \frac{1}{N}\sum_{n=0}^{N-1} x(n)$$

This is unbiased ($E\lbrace \hat{\mu}_x\rbrace = \mu_x$) and consistent (variance $\to 0$ as $N\to\infty$, provided $\sum_{l=-\infty}^{\infty}\lvert\gamma_x(l)\rvert < \infty$).

### Sample Autocorrelation — Biased Estimate

$$\hat{r}_x(l) = \frac{1}{N}\sum_{n=0}^{N-1-\lvert l\rvert} x(n+\lvert l\rvert)\, x^*(n), \qquad \lvert l\rvert \le N-1$$

The division by $N$ (not $N-\lvert l\rvert$) makes this a **biased** estimate:

$$E\lbrace \hat{r}_x(l)\rbrace = \frac{N - \lvert l\rvert}{N}\, r_x(l) \ne r_x(l)$$

But it is **asymptotically unbiased** as $N \to \infty$, and — more importantly — the resulting $N \times N$ estimated autocorrelation matrix $\hat{\mathbf{R}}_x$ is guaranteed to be **positive semi-definite** (nonnegative eigenvalues). This is crucial for stability in filter design.

### Sample Autocorrelation — Unbiased Estimate

$$\hat{r}_x^{(\text{unb})}(l) = \frac{1}{N-\lvert l\rvert}\sum_{n=0}^{N-1-\lvert l\rvert} x(n+\lvert l\rvert)\, x^*(n)$$

This is unbiased but may yield a non-positive-definite matrix — leading to unstable models. It also has higher variance at large lags (few samples available).

**In practice:** The biased estimate is preferred in most applications (linear prediction, AR modeling) because of its guaranteed positive definiteness.

---

## 7.3 Least-Squares (LS) Estimation

### Problem Setup

Let the observations follow the linear model:

$$\mathbf{x} = \mathbf{H}\boldsymbol{\theta} + \mathbf{e}$$

where $\mathbf{x} \in \mathbb{R}^N$ is observed, $\mathbf{H} \in \mathbb{R}^{N\times P}$ is the known **observation matrix** (or regressor matrix), $\boldsymbol{\theta} \in \mathbb{R}^P$ is the unknown parameter vector, and $\mathbf{e}$ is the error vector.

The **least-squares estimate** minimizes the sum of squared residuals:

$$J(\boldsymbol{\theta}) = \|\mathbf{x} - \mathbf{H}\boldsymbol{\theta}\|^2 = (\mathbf{x} - \mathbf{H}\boldsymbol{\theta})^T(\mathbf{x} - \mathbf{H}\boldsymbol{\theta})$$

### Normal Equations and LS Solution

Setting $\partial J/\partial \boldsymbol{\theta} = 0$:

$$\mathbf{H}^T\mathbf{H}\,\hat{\boldsymbol{\theta}}_{\mathrm{LS}} = \mathbf{H}^T \mathbf{x}$$

This is the **normal equation**. If $\mathbf{H}^T\mathbf{H}$ is invertible (i.e., $\mathbf{H}$ has full column rank: $\mathrm{rank}(\mathbf{H}) = P$), the unique LS solution is:

$$\boxed{\hat{\boldsymbol{\theta}}_{\mathrm{LS}} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{x} = \mathbf{H}^\dagger \mathbf{x}}$$

where $\mathbf{H}^\dagger = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T$ is the **Moore-Penrose pseudo-inverse** of $\mathbf{H}$.

### Properties of the LS Estimate

1. **Unbiased:** $E\lbrace \hat{\boldsymbol{\theta}}_{\mathrm{LS}}\rbrace = \boldsymbol{\theta}$ (if errors have zero mean).
2. **Error covariance:** $\boldsymbol{\Sigma}_{\hat{\theta}} = E\lbrace (\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})^T\rbrace = \sigma_e^2 (\mathbf{H}^T\mathbf{H})^{-1}$ (if errors are IID with variance $\sigma_e^2$).
3. **Gauss-Markov theorem:** Among all **linear unbiased** estimators, the LS estimate is the Best Linear Unbiased Estimator (BLUE) when errors are IID — it has the smallest variance for each parameter.
4. **Minimum achieved cost:** $J(\hat{\boldsymbol{\theta}}_{\mathrm{LS}}) = \|\mathbf{x}\|^2 - \mathbf{x}^T\mathbf{H}(\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{x}$.

### Weighted Least Squares (WLS)

If errors are not identically distributed (heteroscedastic), we can weight the residuals by the inverse of their variances. Let $\mathbf{W}$ be a positive definite **weighting matrix** (e.g., $\mathbf{W} = \sigma_e^{-2}\mathbf{I}$ for IID errors). The WLS estimate:

$$\hat{\boldsymbol{\theta}}_{\mathrm{WLS}} = (\mathbf{H}^T\mathbf{W}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{W}\mathbf{x}$$

When $\mathbf{W} = \boldsymbol{\Sigma}_e^{-1}$ (inverse of the error covariance), the WLS estimate is the BLUE even for correlated, non-identically distributed errors.

---

## 7.4 Linear Minimum Mean-Square-Error (LMMSE) Estimation

### Orthogonality Principle

We seek the linear estimator $\hat{\boldsymbol{\theta}} = \mathbf{W}\mathbf{x} + \mathbf{b}$ that minimizes the mean-squared error $E\lbrace \lvert\hat{\theta}_i - \theta_i\rvert^2\rbrace$. The optimal estimator satisfies the **orthogonality principle**:

$$E\lbrace (\hat{\theta}_i - \theta_i)\, x(n)^*\rbrace = 0 \quad \text{for all } n$$

**Physical meaning:** The estimation error must be **uncorrelated with every observation** $x(n)$. If any observation were correlated with the error, we could improve the estimate by using that observation more aggressively. At the optimum, we have already extracted all the information that the observations carry.

This principle leads to the **Wiener-Hopf equations** (to be derived in detail in Chapter 6):

$$\mathbf{R}_x \mathbf{w}_{\mathrm{opt}} = \mathbf{r}_{\theta x}$$

where $\mathbf{R}_x = E\lbrace \mathbf{x}\mathbf{x}^H\rbrace$ is the observation autocorrelation matrix and $\mathbf{r}_{\theta x} = E\lbrace \theta\, \mathbf{x}^H\rbrace$ is the cross-correlation between the desired signal and the observations.

The LMMSE framework is the **bridge to Wiener filtering** (Chapter 6): when the parameter $\boldsymbol{\theta}$ is replaced by a random desired signal $d(n)$ and the observations are $\mathbf{x} = [x(n), x(n-1), \ldots]^T$, the LMMSE estimator becomes the FIR Wiener filter.

---

## 7.5 Maximum Likelihood (ML) Estimation

### Definition

The **likelihood function** $L(\boldsymbol{\theta}) = f_x(\mathbf{x}; \boldsymbol{\theta})$ is the PDF of the observations, viewed as a function of the unknown parameters $\boldsymbol{\theta}$ with $\mathbf{x}$ fixed at the observed values.

The **ML estimate** maximizes the likelihood over $\boldsymbol{\theta}$:

$$\boxed{\hat{\boldsymbol{\theta}}_{\mathrm{ML}} = \arg\max_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) = \arg\max_{\boldsymbol{\theta}} \ln L(\boldsymbol{\theta})}$$

(The log-likelihood $\ln L$ is equivalent and usually more convenient.)

### Properties of ML Estimates

Under regularity conditions:
- **Consistency:** $\hat{\boldsymbol{\theta}}_{\mathrm{ML}} \to \boldsymbol{\theta}$ in probability as $N \to \infty$.
- **Asymptotic unbiasedness:** $E\lbrace \hat{\boldsymbol{\theta}}_{\mathrm{ML}}\rbrace \to \boldsymbol{\theta}$ as $N \to \infty$.
- **Asymptotic efficiency:** The ML estimate asymptotically achieves the CRLB — its covariance matrix approaches $\mathbf{J}^{-1}(\boldsymbol{\theta})$.
- **Asymptotic normality:** $\sqrt{N}(\hat{\boldsymbol{\theta}}_{\mathrm{ML}} - \boldsymbol{\theta}) \xrightarrow{D} \mathcal{N}(\mathbf{0}, \mathbf{J}^{-1}(\boldsymbol{\theta}))$.

### Example: ML Estimation of the Mean of a Gaussian Process

Let $\mathbf{x} = [x(0), \ldots, x(N-1)]^T$ be IID Gaussian with unknown mean $\mu$ and known variance $\sigma^2$. The log-likelihood is:

$$\ln L(\mu) = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=0}^{N-1}(x(n)-\mu)^2$$

Differentiating and setting to zero:

$$\frac{d\ln L}{d\mu} = \frac{1}{\sigma^2}\sum_{n=0}^{N-1}(x(n)-\mu) = 0 \implies \hat{\mu}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=0}^{N-1} x(n)$$

The ML estimate of the mean is the **sample mean** — and one can verify that it achieves the CRLB $\sigma^2/N$.

### Application: Time-Delay Estimation via Generalized Cross-Correlation (GCC)

A classic application of ML estimation to signal processing: estimating the **time delay** $\Delta$ between two microphone signals $x_1(n) = s(n) + v_1(n)$ and $x_2(n) = s(n - \Delta) + v_2(n)$, where $s(n)$ is the source and $v_i(n)$ is noise. The ML-GCC estimator finds $\hat{\Delta}$ by locating the peak of the cross-correlation:

$$\hat{\Delta}_{\mathrm{GCC}} = \arg\max_\Delta\, \sum_l \hat{r}_{x_1 x_2}(l)\, \text{window}(l - \Delta)$$

The choice of window (PHAT, SCOT, Roth processor) prewhitens the cross-spectrum to improve peak sharpness in noisy environments.

---

## 7.6 Bayesian Estimation (Introduction)

In the **Bayesian framework**, the parameter $\boldsymbol{\theta}$ is treated as a **random variable** with a known **prior distribution** $f_\theta(\boldsymbol{\theta})$ that encodes any prior knowledge before observing $\mathbf{x}$. After observing $\mathbf{x}$, the **posterior distribution** is:

$$f_{\theta\mid x}(\boldsymbol{\theta}\mid\mathbf{x}) = \frac{f_x(\mathbf{x}\mid\boldsymbol{\theta})\, f_\theta(\boldsymbol{\theta})}{f_x(\mathbf{x})}$$

(Bayes' theorem). The form of the optimal Bayesian estimate depends on the **cost function** chosen:

| Cost Function | Optimal Estimate |
|-------------|-----------------|
| **Quadratic:** $C(\hat{\theta}, \theta) = \lvert\hat{\theta} - \theta\rvert^2$ | **MMSE:** $\hat{\boldsymbol{\theta}}_{\mathrm{MMSE}} = E\lbrace \boldsymbol{\theta}\mid\mathbf{x}\rbrace$ (posterior mean) |
| **Uniform (0-1 loss):** $C = 0$ if $\lvert\hat{\theta}-\theta\rvert < \epsilon$, else $1$ | **MAP:** $\hat{\boldsymbol{\theta}}_{\mathrm{MAP}} = \arg\max_{\boldsymbol{\theta}} f_{\theta\mid x}(\boldsymbol{\theta}\mid\mathbf{x})$ (posterior mode) |

**Relationship to ML:** MAP reduces to ML when the prior $f_\theta(\boldsymbol{\theta})$ is **uniform** (no prior preference for any value of $\boldsymbol{\theta}$) — ML is a special case of MAP with an uninformative (flat) prior.

**LMMSE as linear Bayesian estimation:** When the prior is Gaussian and the likelihood is Gaussian, the MMSE estimate is linear in $\mathbf{x}$ and coincides with the LMMSE estimate. This provides the statistical justification for the Wiener filter (Chapter 6).

---

## Chapter 2 Summary

| Concept | Key Result | Application in Later Chapters |
|---------|-----------|------------------------------|
| Random variable | Characterized by CDF/PDF; mean $\mu_x$, variance $\sigma_x^2$ | Foundation for all stochastic signal processing |
| WSS process | $r_x(n_1,n_2) = r_x(l)$ (lag only); Hermitian Toeplitz $\mathbf{R}_x$ | Wiener filter (Ch. 6), Kalman filter (Ch. 6) |
| Wiener-Khinchin | $R_x(e^{j\omega}) = \mathcal{F}\lbrace r_x(l)\rbrace$, $R_x(e^{j\omega}) \ge 0$ | Spectral estimation (Ch. 5) |
| White noise | $r_w(l) = \sigma_w^2\delta(l)$; flat PSD $R_w = \sigma_w^2$ | Driving noise for all linear models |
| Random through filter | $R_y(e^{j\omega}) = \lvert H(e^{j\omega})\rvert^2 R_x(e^{j\omega})$ | Spectrum shaping, channel modeling |
| Spectral factorization | $R_x(z) = \sigma_0^2 H_+(z) H_+^{\ast}(1/z^{\ast})$ | Innovations, Wiener filter design (Ch. 6) |
| AR($p$) process | $x(n) = -\sum a_k x(n-k) + w(n)$; Yule-Walker $\mathbf{R}_x\mathbf{a} = -\mathbf{r}_x$ | AR spectrum estimation (Ch. 4–5), linear prediction (Ch. 3) |
| MA($q$) process | $x(n) = B(z)w(n)$; $r_x(l) = 0$ for $\lvert l\rvert>q$ | Noise modeling, MA spectrum (Ch. 4) |
| ARMA($p$,$q$) process | $B(z)/A(z)$ model; modified Yule-Walker | General spectral modeling (Ch. 4) |
| Harmonic process | Line spectrum; $r_x(l) = \sum A_k^2\cos\omega_k l$ | Frequency estimation (Ch. 5) |
| KL transform (DKLT) | $\mathbf{w} = \mathbf{Q}_x^H\mathbf{x}$; decorrelates; optimal MSE truncation | Dimensionality reduction; DCT approximation |
| DCT-II | Fixed cosine basis; near-optimal for Markov-1; fast FFT-based algorithm | Image/speech coding; transform-domain adaptive filtering (Ch. 7) |
| Ergodicity | Time averages = ensemble averages | Justifies using sample statistics in practice |
| CRLB | $\text{var}(\hat{\theta}) \ge [J^{-1}(\theta)]_{ii}$ | Performance benchmark for any estimator |
| LS estimation | $\hat{\boldsymbol{\theta}} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{x}$; BLUE when errors IID | Linear prediction equations (Ch. 3), adaptive filters (Ch. 7) |
| ML estimation | Asymptotically efficient, achieves CRLB | Optimal spectral estimation algorithms (Ch. 5) |
| LMMSE / Orthogonality principle | Error $\perp$ observations | Wiener filter (Ch. 6); motivates adaptive filter cost functions (Ch. 7) |

### Looking Ahead

This chapter established the **stochastic signal analysis foundation** for the rest of the course. The key tools we built are:

1. **The Hermitian Toeplitz autocorrelation matrix** $\mathbf{R}_x$ — its structure enables the efficient Levinson-Durbin algorithm (Chapter 3).
2. **The Yule-Walker equations for AR processes** — the bridge between autocorrelation and linear prediction (Chapter 3).
3. **The Wiener-Khinchin theorem** — the foundation for spectral estimation (Chapter 5).
4. **The LMMSE / orthogonality principle** — the foundation for Wiener filtering (Chapter 6) and adaptive filtering (Chapter 7).
5. **Spectral factorization** — essential for Wiener filter design (Chapter 6) and understanding the innovations process.

---

*End of Chapter 2*
