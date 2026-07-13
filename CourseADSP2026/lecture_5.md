# Modern Digital Signal Processing
## Chapter 5: Power Spectrum Estimation

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005  
> Chapters covered: Ch. 5 (Nonparametric Power Spectrum Estimation) · §9.2 (AR Spectral Estimation) · §9.3 (Pole-Zero Models) · §9.5 (Minimum-Variance Spectrum Estimation) · §9.6 (Harmonic Models and Frequency Estimations)

---

> **Relationship to Chapter 4.** Chapter 4 asked a **modeling question**:
> *What low-order stochastic model could have generated the signal, and how can
> its parameters be identified?* It developed AR, MA, and ARMA models through
> coloring/whitening filters, prediction, autocorrelation signatures, and model
> diagnostics. Chapter 5 asks an **estimation question**: *Given only a finite
> data record, how should its spectrum be estimated, and how should competing
> estimators be judged?* The emphasis therefore shifts to bias, variance,
> resolution, leakage, consistency, order sensitivity, and method selection.
> Parametric AR/MA/ARMA spectra reappear here intentionally: they are now
> treated as one family of spectrum estimators and compared with periodogram,
> Bartlett, Welch, Blackman-Tukey, minimum-variance, maximum-entropy, and
> subspace methods.

---

## Table of Contents

1. [§0 Spectrum-Estimation Roadmap](#0-spectrum-estimation-roadmap)
2. [§1 Problem Statement and Performance Criteria](#1-problem-statement-and-performance-criteria)
3. [§2 Classical Nonparametric Spectrum Estimation](#2-classical-nonparametric-spectrum-estimation)
4. [§3 Parametric Spectrum Estimation](#3-parametric-spectrum-estimation)
5. [§4 Minimum-Variance Spectrum Estimation](#4-minimum-variance-spectrum-estimation)
6. [§5 Maximum-Entropy Spectrum Estimation](#5-maximum-entropy-spectrum-estimation)
7. [§6 Frequency Estimation and Subspace Methods](#6-frequency-estimation-and-subspace-methods)
8. [§7 Chapter Summary, Method Selection, and Figure Checklist](#7-chapter-summary-method-selection-and-figure-checklist)

---

## Notation and Variable Definitions

This chapter uses the notation of the previous lectures. The new emphasis is that we no longer assume that the power spectrum is known. Instead, we estimate it from a finite data record.

### Time, Data Length, and Frequency

| Symbol | Definition |
|--------|------------|
| $n$ | Discrete-time sample index |
| $N$ | Total number of observed samples in the available record |
| $L$ | Length of a segment or correlation window |
| $K$ | Number of averaged segments or number of averaged periodograms |
| $D$ | Hop size between adjacent Welch segments |
| $M$ | Time-window vector length in harmonic/subspace methods |
| $p$ | Filter order, AR model order, or number of past lags, depending on context |
| $P$ | Number of complex exponential components, or AR order when emphasized as a model order |
| $Q$ | MA order or number of zeros in a pole-zero model |
| $\omega$ | Discrete-time angular frequency in radians/sample |
| $f=\omega/(2\pi)$ | Normalized frequency in cycles/sample |
| $F$ | Continuous-time frequency in Hz |
| $F_s$ | Sampling frequency in Hz |

### Signals and Random Processes

| Symbol | Definition |
|--------|------------|
| $x(n)$ | Observed discrete-time signal or WSS random process |
| $x_N(n)$ | Finite record of $x(n)$, usually $x(n)$ multiplied by a length-$N$ rectangular window |
| $w(n)$ | White-noise excitation or additive white noise, depending on context |
| $s(n)$ | Deterministic harmonic component in the harmonic model |
| $\nu(n)$ | Additive observation noise |
| $r_x(l)$ | True autocorrelation sequence of a WSS process |
| $\hat r_x(l)$ | Estimated autocorrelation sequence from finite data |
| $R_x(e^{j\omega})$ | True power spectral density (PSD) |
| $\hat R_x(e^{j\omega})$ | Estimated PSD |
| $X_N(e^{j\omega})$ | DTFT of the finite data record |
| $X_N(k)$ | DFT sample of the finite data record at frequency $\omega_k=2\pi k/N$ |

### Windows and Nonparametric Estimators

| Symbol | Definition |
|--------|------------|
| $w_N(n)$ | Data window applied directly to samples before the DFT |
| $v(l)$ | Correlation window or lag window applied to an autocorrelation estimate |
| $U=\sum_{n=0}^{L-1} \vert w(n)\vert^2$ | Data-window energy normalization |
| $\hat R_x^{(P)}(e^{j\omega})$ | Periodogram estimate |
| $\hat R_x^{(MP)}(e^{j\omega})$ | Modified periodogram estimate |
| $\hat R_x^{(BT)}(e^{j\omega})$ | Blackman-Tukey estimate |
| $\hat R_x^{(B)}(e^{j\omega})$ | Bartlett estimate |
| $\hat R_x^{(W)}(e^{j\omega})$ | Welch estimate |

### Parametric and Subspace Quantities

| Symbol | Definition |
|--------|------------|
| $A(z)=1+\sum_{k=1}^{P}a_kz^{-k}$ | AR / all-pole polynomial |
| $D(z)=d_0+\sum_{k=1}^{Q}d_kz^{-k}$ | MA / all-zero polynomial |
| $H(z)=D(z)/A(z)$ | Pole-zero synthesis model |
| $\sigma_w^2$ | Variance of the white-noise excitation |
| $\mathbf{R}_x$ | Autocorrelation matrix of a data vector |
| $\mathbf{a}(\omega)\ \text{or}\ \mathbf{v}(f)$ | Complex exponential steering vector |
| $\mathbf{Q}_s$ | Signal-subspace eigenvector matrix |
| $\mathbf{Q}_n$ | Noise-subspace eigenvector matrix |
| $\lambda_i$ | Eigenvalue of a correlation matrix |
| $\Psi$ | ESPRIT rotational operator |

---

# §0 Spectrum-Estimation Roadmap

> 📖 Textbook Ch. 5 introduction; Fig. 5.1; §9.5; §9.6

## 0.1 Why Spectrum Estimation Is a Separate Topic

In earlier chapters, the power spectrum was usually introduced as a theoretical object. For a WSS process,

$$R_x(e^{j\omega})=\sum_{l=-\infty}^{\infty} r_x(l)e^{-j\omega l}.$$

This definition is exact, but it assumes that the true autocorrelation sequence $r_x(l)$ is known for all lags. In a real experiment, we usually have only one finite record:

$$x(0),x(1),\ldots,x(N-1).$$

The central question of this chapter is therefore:

> **How can we estimate the power spectrum of a stationary random signal from a finite data record?**

The answer is not unique. Different estimators make different compromises among leakage, resolution, variance, model assumptions, and computational cost.

> ![Figure 0.1](./CourseADSP2026/Fig/Chapter_5/fig_0_1_textbook_fig_5_1_p196.png)
>
> *Figure 0.1 (Textbook Fig. 5.1, p. 196): Classification of spectrum-estimation methods. This chapter begins with Fourier/nonparametric methods, then connects them to parametric, minimum-variance, maximum-entropy, and subspace frequency-estimation methods.*

## 0.2 Two Big Families: Nonparametric and Parametric

The textbook classification is useful because it separates methods by what they assume about the signal.

| Family | Main Assumption | Typical Methods | Main Strength | Main Weakness |
|--------|-----------------|-----------------|---------------|---------------|
| Nonparametric | No low-order model is assumed | Periodogram, modified periodogram, Bartlett, Welch, Blackman-Tukey, multitaper | Robust and easy to use | Limited resolution and high variance unless averaged or smoothed |
| Parametric | Signal follows a low-order model | AR, MA, ARMA, maximum entropy | High resolution with short data if model is correct | Can be misleading if model order or model type is wrong |
| Minimum-variance | Use adaptive narrowband filters | Capon / MV spectral estimator | Data-adaptive leakage control | Requires matrix inversion and order selection |
| Subspace frequency estimation | Signal is a sum of a small number of sinusoids plus noise | Pisarenko, MUSIC, EV, minimum-norm, ESPRIT | Very high frequency resolution | Needs correct model order and sufficiently high SNR |

The main teaching point is that **spectrum estimation is not simply “take an FFT.”** The FFT is a computational tool. A spectrum estimator is a statistical procedure that turns finite noisy data into an estimate of a second-order quantity.

## 0.3 What This Chapter Adds Beyond Chapter 4

Chapter 4 and Chapter 5 use some of the same formulas, but they use them to
answer different questions.

| Chapter 4: Signal Modeling | Chapter 5: Spectrum Estimation |
|----------------------------|--------------------------------|
| Starts from a candidate stochastic model | Starts from a finite observed record |
| Asks how the signal may have been generated | Asks how its spectral content should be inferred |
| Emphasizes AR, MA, and ARMA structure | Compares nonparametric, parametric, adaptive, entropy, and subspace methods |
| Studies poles, zeros, ACS/PACS signatures, prediction, and residual whiteness | Studies bias, variance, resolution, leakage, consistency, and computational tradeoffs |
| Uses PSD as a consequence of the fitted model | Uses estimation performance to decide whether a spectral estimate is trustworthy |

A **model** says how the signal could have been generated. A **spectrum
estimator** is a statistical procedure for inferring spectral information from
finite data. These ideas are related, but they are not interchangeable.

For example, an AR model gives a spectrum of the form

$$R_x(e^{j\omega})=\frac{\sigma_w^2}{\vert A(e^{j\omega})\vert^2}.$$

In Chapter 4, this formula describes the spectrum implied by an all-pole signal
model and connects pole locations to spectral peaks. In Chapter 5, the same
formula becomes an estimator after $A(z)$ and $\sigma_w^2$ have been inferred
from a short, noisy record. We must then ask additional questions:

- How biased and variable is the resulting spectral estimate?
- Does its apparent resolution reflect real information or model-induced peaks?
- How sensitive is it to record length, estimation method, and model order?
- When is it preferable to a periodogram, Welch estimate, MV estimate, or
  subspace method?

Thus, the parametric material in Section 3 is not meant to replace or compress
Chapter 4. It revisits AR, MA, and ARMA models from a new viewpoint: their use,
limitations, and comparison as finite-data spectrum estimators.

---

# §1 Problem Statement and Performance Criteria

> 📖 Textbook §5.1-§5.3

## 1.1 The Basic Experimental Situation

The spectrum-estimation problem usually starts from a finite data record:

$$x(0),x(1),\ldots,x(N-1).$$

If $x(n)$ is a deterministic signal, the goal may be to estimate the energy spectrum or identify sinusoidal components. If $x(n)$ is a WSS random process, the goal is to estimate the PSD $R_x(e^{j\omega})$.

The standard DFT-based processing chain is shown below.

> ![Figure 1.1](./CourseADSP2026/Fig/Chapter_5/fig_1_1_textbook_fig_5_2_p197.png)
>
> *Figure 1.1 (Textbook Fig. 5.2, p. 197): DFT-based Fourier analysis system for continuous-time signals. The observed data are produced by sampling, frame blocking, windowing, and DFT computation.*

This figure should be read from left to right:

1. A continuous-time signal is filtered by an anti-aliasing low-pass filter.
2. It is sampled by an A/D converter.
3. A finite frame is selected.
4. A data window is applied.
5. A DFT is computed.

Each step changes what we can observe. In particular, frame blocking and windowing are unavoidable in finite-data analysis. They are the source of many spectral-estimation effects.

## 1.2 Why Finite Data Causes Spectral Distortion

Suppose the infinite signal is $x(n)$, but we observe only $N$ samples. This is equivalent to multiplying the infinite signal by a rectangular window:

$$x_N(n)=x(n)w_R(n),$$

where

$$w_R(n)=\begin{cases}
1, & 0\le n\le N-1,\\
0, & \text{otherwise}.
\end{cases}$$

Multiplication in time corresponds to convolution in frequency. Thus the observed spectrum is not the true spectrum itself; it is the true spectrum smeared by the Fourier transform of the window.

This has two immediate consequences:

| Effect | Meaning |
|--------|---------|
| Mainlobe spreading | A narrow spectral component becomes a finite-width lobe |
| Sidelobe leakage | Energy from a strong component leaks into other frequencies |

Therefore, even before randomness is considered, the finite observation window has already introduced distortion.

## 1.3 Four Performance Criteria

For a random process, $\hat R_x(e^{j\omega})$ is itself random. We evaluate a spectrum estimator using several criteria.

### 1. Bias

The bias at frequency $\omega$ is

$$\operatorname{bias}\{\hat R_x(e^{j\omega})\}=E\{\hat R_x(e^{j\omega})\}-R_x(e^{j\omega}).$$

A biased estimator may systematically smooth peaks, lift valleys, or shift energy into nearby frequencies.

### 2. Variance

The variance is

$$\operatorname{var}\{\hat R_x(e^{j\omega})\}=E\left\{\left\vert\hat R_x(e^{j\omega})-E\{\hat R_x(e^{j\omega})\}\right\vert^2\right\}.$$

High variance means the estimate changes strongly from one data record to another.

### 3. Resolution

Resolution describes the ability to separate two nearby spectral components. It is mainly controlled by the effective length of the data or correlation window. Longer windows have narrower mainlobes and therefore better resolution.

### 4. Consistency

An estimator is consistent if, as $N\to\infty$,

$$\hat R_x(e^{j\omega})\to R_x(e^{j\omega})$$

in an appropriate probabilistic sense. For practical purposes, consistency means that both bias and variance should vanish as the data length grows.

The ordinary periodogram has an important defect: its bias can decrease with increasing $N$, but its variance does not vanish. Therefore, the periodogram is not a consistent PSD estimator.

## 1.4 The Bias-Variance-Resolution Triangle

Most spectrum-estimation methods can be understood as a tradeoff among three goals:

| Goal | How to Improve It | Price Paid |
|------|-------------------|------------|
| Reduce leakage | Use a smoother data window with lower sidelobes | Wider mainlobe, worse resolution |
| Reduce variance | Average or smooth periodograms | Worse resolution or less independent information |
| Improve resolution | Use longer effective data length or a model | Higher variance or stronger model assumptions |

A good spectrum estimate is not necessarily the one with the sharpest peaks. It is the one whose assumptions match the signal and whose tradeoff matches the task.

---

# §2 Classical Nonparametric Spectrum Estimation

> 📖 Textbook Ch. 5, especially §5.1-§5.5

Classical nonparametric methods estimate the PSD directly from data without assuming an AR, MA, or ARMA model. The main methods are:

1. the periodogram,
2. the modified periodogram,
3. Bartlett averaging,
4. Welch averaging,
5. Blackman-Tukey autocorrelation windowing.

These methods are called nonparametric because the spectrum is not represented by a small number of model parameters.

---

## 2.1 Periodogram Method

### 2.1.1 Definition

Given $N$ samples $x(0),\ldots,x(N-1)$, define the finite-record DTFT

$$X_N(e^{j\omega})=\sum_{n=0}^{N-1}x(n)e^{-j\omega n}.$$

The periodogram is

$$\boxed{\hat R_x^{(P)}(e^{j\omega})=\frac{1}{N}\vert X_N(e^{j\omega})\vert^2.}$$

Here the superscript $(P)$ is a **method label**: $P$ stands for *periodogram*. It is not an exponent and does not come from an algebraic operation. Thus, $\hat R_x^{(P)}$ means “the estimate of $R_x$ produced by the periodogram method.” The hat denotes an estimate, and $1/N$ normalizes the squared magnitude of the finite-record DTFT by the number of samples. Do not confuse this method label with a symbol $P$ used elsewhere to denote a model order.

At DFT frequencies $\omega_k=2\pi k/N$,

$$\boxed{\hat R_x^{(P)}(e^{j\omega_k})=\frac{1}{N}\vert X_N(k)\vert^2.}$$

This is the most intuitive PSD estimator: compute the DFT, square the magnitude, and normalize by $N$.

#### Why This Definition Is Natural

The definition is not an arbitrary choice. It follows from the fundamental
definition of the PSD as the DTFT of the autocorrelation sequence. For a WSS
process,

$$R_x(e^{j\omega})=\sum_{l=-\infty}^{\infty}r_x(l)e^{-j\omega l},
\qquad
r_x(l)=E\{x(n+l)x^\ast(n)\}.$$

With only $N$ observed samples, replace the unknown autocorrelation by the
finite-record estimate

$$\hat r_x(l)=\frac{1}{N}\sum_{n=0}^{N-1-l}x(n+l)x^\ast(n),
\qquad 0\le l\le N-1,$$

and use conjugate symmetry for negative lags. Taking its DTFT gives

$$
\begin{aligned}
\sum_{l=-(N-1)}^{N-1}\hat r_x(l)e^{-j\omega l}
&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{m=0}^{N-1}
x(n)x^\ast(m)e^{-j\omega(n-m)}\\
&=\frac{1}{N}
\left(\sum_{n=0}^{N-1}x(n)e^{-j\omega n}\right)
\left(\sum_{m=0}^{N-1}x(m)e^{-j\omega m}\right)^\ast\\
&=\frac{1}{N}\left|X_N(e^{j\omega})\right|^2.
\end{aligned}
$$

Thus, the periodogram is exactly the DTFT of this particular finite-sample
autocorrelation estimate.

The factor $1/N$ converts finite-record spectral energy into average power.
Indeed, Parseval's relation gives

$$\frac{1}{2\pi}\int_{-\pi}^{\pi}
\hat R_x^{(P)}(e^{j\omega})\,d\omega
=\frac{1}{N}\sum_{n=0}^{N-1}|x(n)|^2.$$

Hence the area under the periodogram equals the observed average sample power.
Without the factor $1/N$, the integral would represent total record energy
rather than average power.

#### Why It Is Called a Periodogram

Historically, the method was introduced to reveal periodic components in an
observed record. If the data contain a sinusoid with angular frequency
$\omega_0$, the periodogram has a peak near $\omega_0$ (and, for a real
sinusoid, also near $-\omega_0$). The peak therefore identifies the period
$T=2\pi/\omega_0$. The suffix *-gram* means a record or graphical display, so
*periodogram* originally meant a graphical record of the strengths of possible
periodicities. The name does not mean that the plotted curve itself must be a
periodic signal, although a discrete-time spectrum is of course $2\pi$-periodic
in frequency.

### 2.1.2 Periodogram as an Autocorrelation-Based Estimator

The periodogram can also be written as the DTFT of a finite autocorrelation estimate. Define

$$\hat r_x(l)=\frac{1}{N}\sum_{n=0}^{N-1-l}x(n+l)x^\ast(n),\quad 0\le l\le N-1,$$

and extend it by conjugate symmetry for negative lags:

$$\hat r_x(-l)=\hat r_x^\ast(l).$$

Then the periodogram satisfies

$$\hat R_x^{(P)}(e^{j\omega})=\sum_{l=-(N-1)}^{N-1}\hat r_x(l)e^{-j\omega l}.$$

This identity is very important pedagogically:

> The periodogram is not just “FFT magnitude squared.” It is also an autocorrelation estimate transformed to the frequency domain.

This viewpoint prepares us for the Blackman-Tukey method, which explicitly modifies the autocorrelation estimate before transforming it.

### 2.1.3 Periodogram as a Filter-Bank Output

The textbook gives another useful interpretation. Each DFT bin can be viewed as the output of a narrowband filter centered at that frequency. The periodogram estimates the average output power of that filter.

To see why, start with the $k$th DFT coefficient:

$$X[k]=\sum_{n=0}^{N-1}x(n)e^{-j\omega_k n},
\qquad \omega_k=\frac{2\pi k}{N}.$$

The operation “multiply by $e^{-j\omega_k n}$ and add” tests how much of the frequency $\omega_k$ is present in the data. For example, if

$$x(n)=Ae^{j\omega_k n},$$

then every product is the same constant,

$$x(n)e^{-j\omega_k n}=A,$$

so all $N$ terms add in phase and $X[k]=NA$. The response is large. In contrast, for a component at another frequency $\omega_0$,

$$X[k]=A\sum_{n=0}^{N-1}e^{j(\omega_0-\omega_k)n}.$$

These terms rotate in phase and partially cancel. The farther $\omega_0$ is from $\omega_k$, the stronger the cancellation is in general. Therefore DFT bin $k$ passes frequencies near $\omega_k$ and suppresses frequencies farther away: it behaves like a narrowband filter centered at $\omega_k$.

More formally, apart from an irrelevant phase shift, that filter has frequency response

$$H_k(e^{j\omega})=\sum_{n=0}^{N-1}e^{-j(\omega-\omega_k)n}.$$

Its magnitude is a shifted Dirichlet kernel. Its main lobe is centered at $\omega_k$, and its first zeros are at $\omega_k\pm 2\pi/N$. Hence a longer record produces a narrower effective filter and better frequency resolution. The sidelobes also explain spectral leakage: a component outside the main lobe can still affect bin $k$.

The filter has coefficient energy $N$. If it is normalized to unit energy, its squared output magnitude is

$$\left\lvert\frac{X[k]}{\sqrt N}\right\rvert^2
=\frac{1}{N}|X[k]|^2
=\hat R_x^{(P)}(e^{j\omega_k}).$$

Therefore, “the periodogram estimates the filter output power” means the following: **the DFT acts as a bank of narrowband filters, one for each DFT frequency, and the periodogram is the squared magnitude of each energy-normalized filter output.** No physical filter bank has to be built; the DFT computes all these outputs at once.

> ![Figure 2.1](./CourseADSP2026/Fig/Chapter_5/fig_2_4_textbook_fig_5_11_p215.png)
>
> *Figure 2.1 (Textbook Fig. 5.11, p. 215): Filter-bank interpretation of periodogram computation. Each frequency sample is associated with a narrowband filter, and the periodogram estimates power by looking at the output power of that filter.*

This filter-bank interpretation explains both the usefulness and the limitation of the periodogram:

- It is useful because it measures power around each frequency.
- It is limited because the filters are fixed by the window length and are not adapted to the data.

### 2.1.4 Bias of the Periodogram

For a WSS process, the expected periodogram can be derived directly. Begin by
expanding the squared magnitude:

$$
\hat R_x^{(P)}(e^{j\omega})
=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{m=0}^{N-1}
x(n)x^\ast(m)e^{-j\omega(n-m)}.
$$

WSS implies that the second moment depends only on the lag:

$$E\{x(n)x^\ast(m)\}=r_x(n-m).$$

Consequently,

$$
E\{\hat R_x^{(P)}(e^{j\omega})\}
=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{m=0}^{N-1}
r_x(n-m)e^{-j\omega(n-m)}.
$$

Set $l=n-m$. For a fixed lag $l$ with $|l|<N$, there are exactly
$N-|l|$ pairs $(n,m)$ in the observed record satisfying $n-m=l$. Grouping the
double sum by lag therefore gives

$$
E\{\hat R_x^{(P)}(e^{j\omega})\}
=\sum_{l=-(N-1)}^{N-1}
\left(1-\frac{|l|}{N}\right)r_x(l)e^{-j\omega l}.
$$

Define the triangular lag window

$$
v_N(l)=
\begin{cases}
1-|l|/N, & |l|<N,\\
0, & |l|\ge N.
\end{cases}
$$

Because $v_N(l)=0$ outside the displayed finite-lag interval, the preceding
sum may equivalently be extended over all integer lags:

$$
E\{\hat R_x^{(P)}(e^{j\omega})\}
=\sum_{l=-\infty}^{\infty}r_x(l)v_N(l)e^{-j\omega l}
=\mathcal F\{r_x(l)v_N(l)\}.
$$

The last equality is simply the definition of the DTFT: for any sequence
$g(l)$,

$$\mathcal F\{g(l)\}=\sum_{l=-\infty}^{\infty}g(l)e^{-j\omega l},$$

with $g(l)=r_x(l)v_N(l)$ here. Now define

$$R_x(e^{j\omega})=\mathcal F\{r_x(l)\},
\qquad
W_N(e^{j\omega})=\mathcal F\{v_N(l)\}.$$

Multiplication in the lag domain becomes $2\pi$-periodic convolution in the
frequency domain. Hence

$$
\boxed{
E\{\hat R_x^{(P)}(e^{j\omega})\}
=\frac{1}{2\pi}\int_{-\pi}^{\pi}
R_x(e^{j\theta})W_N(e^{j(\omega-\theta)})\,d\theta.}
$$

For the length-$N$ rectangular data window, this spectral window is the Fejér
kernel

$$
W_N(e^{j\omega})
=\frac{1}{N}\left|\sum_{n=0}^{N-1}e^{-j\omega n}\right|^2
=\frac{1}{N}\left[\frac{\sin(N\omega/2)}{\sin(\omega/2)}\right]^2,
$$

with its value at $\omega=0$ understood by continuity. It is normalized so
that

$$\frac{1}{2\pi}\int_{-\pi}^{\pi}W_N(e^{j\omega})\,d\omega=1.$$

Thus, under the stated WSS assumptions and the ordinary rectangular-window
periodogram definition, the convolution formula is an exact finite-$N$
identity. The word *approximately* is often used only for the interpretation
that a sufficiently concentrated $W_N$ acts like a local smoothing kernel, or
when additional asymptotic assumptions and alternative normalizations are being
used. As $N$ increases, the mainlobe narrows and $W_N$ approaches a periodic
impulse in the distributional sense, so the expected periodogram approaches the
true PSD at frequencies where the PSD is sufficiently well behaved.

This means the periodogram is generally biased. The bias is not arbitrary: it is caused by spectral smoothing and leakage.

#### Why the Periodogram Is Unbiased for White Noise

For white noise, the true PSD is constant at every frequency:

$$R_x(e^{j\theta})=\sigma_x^2.$$

Substituting this constant PSD into the convolution formula gives

$$
\begin{aligned}
E\{\hat R_x^{(P)}(e^{j\omega})\}
&=\frac{1}{2\pi}\int_{-\pi}^{\pi}
\sigma_x^2 W_N(e^{j(\omega-\theta)})\,d\theta\\
&=\sigma_x^2
\frac{1}{2\pi}\int_{-\pi}^{\pi}
W_N(e^{j(\omega-\theta)})\,d\theta.
\end{aligned}
$$

Because $W_N$ is $2\pi$-periodic, shifting its argument by $\omega$ does not
change its integral over any interval of length $2\pi$. Its normalization
therefore implies

$$
\frac{1}{2\pi}\int_{-\pi}^{\pi}
W_N(e^{j(\omega-\theta)})\,d\theta
=\frac{1}{2\pi}\int_{-\pi}^{\pi}W_N(e^{j\phi})\,d\phi
=1.
$$

It follows that

$$
\boxed{
E\{\hat R_x^{(P)}(e^{j\omega})\}
=\sigma_x^2
=R_x(e^{j\omega}).}
$$

Thus, the ordinary periodogram is unbiased for white noise at every frequency.
The intuitive reason is that convolution with the normalized spectral window
forms a weighted average of the true PSD around the frequency of interest. If
all values being averaged are the same constant, their weighted average is the
same constant.

For a non-flat PSD, however, the spectral window averages unequal neighboring
values. Near a sharp peak, the peak value is mixed with smaller surrounding
values, so the expected peak is generally lowered and broadened. Frequencies
beside the peak are mixed with its larger value and are consequently raised.
The sidelobes of $W_N$ can also transfer power from more distant frequencies;
this is spectral leakage. Therefore, for a general non-flat spectrum,

$$E\{\hat R_x^{(P)}(e^{j\omega})\}\ne R_x(e^{j\omega}),$$

and the frequency-dependent bias is

$$
\operatorname{Bias}(\omega)
=E\{\hat R_x^{(P)}(e^{j\omega})\}-R_x(e^{j\omega}).
$$

Finally, **unbiased does not mean that one white-noise periodogram is flat**.
It means that the ensemble average of periodograms over many independent
realizations equals the flat true PSD. An individual white-noise periodogram
still fluctuates strongly because its variance is large, as discussed next.

### 2.1.5 Variance of the Periodogram

A central result is that the periodogram has high variance. This can be seen
most clearly by first considering zero-mean circular complex Gaussian white
noise,

$$x(n)\sim\mathcal{CN}(0,\sigma_x^2),$$

with independent samples. At a DFT frequency $\omega_k=2\pi k/N$,

$$X_N(k)=\sum_{n=0}^{N-1}x(n)e^{-j\omega_k n}.$$

A linear combination of independent Gaussian variables is Gaussian. Moreover,

$$
E\{|X_N(k)|^2\}
=\sum_{n=0}^{N-1}\sum_{m=0}^{N-1}
E\{x(n)x^\ast(m)\}e^{-j\omega_k(n-m)}
=N\sigma_x^2,
$$

because $E\{x(n)x^\ast(m)\}=\sigma_x^2\delta(n-m)$. Therefore,

$$X_N(k)\sim\mathcal{CN}(0,N\sigma_x^2),$$

and it may be represented as

$$X_N(k)=\sqrt{N\sigma_x^2}\,Z,
\qquad Z\sim\mathcal{CN}(0,1).$$

Substitution into the periodogram gives

$$
\hat R_x^{(P)}(e^{j\omega_k})
=\frac{1}{N}|X_N(k)|^2
=\sigma_x^2|Z|^2.
$$

The factor $N$ has canceled completely. For a standard circular complex
Gaussian variable, $|Z|^2$ has a unit-mean exponential distribution, so

$$E\{|Z|^2\}=1,
\qquad
\operatorname{var}\{|Z|^2\}=1.$$

It follows exactly that

$$
E\{\hat R_x^{(P)}(e^{j\omega_k})\}=\sigma_x^2,
\qquad
\boxed{
\operatorname{var}\{\hat R_x^{(P)}(e^{j\omega_k})\}
=\sigma_x^4.}
$$

Since the white-noise PSD is $R_x(e^{j\omega})=\sigma_x^2$, this result can
also be written as

$$
\operatorname{var}\{\hat R_x^{(P)}(e^{j\omega_k})\}
=R_x^2(e^{j\omega_k}),
$$

which contains no $N$. Thus, even in this simplest case, increasing the record
length does not make the variance approach zero.

For a more general WSS process satisfying suitable weak-dependence conditions,
the finite-record Fourier coefficient at a fixed frequency is asymptotically
approximately circular complex Gaussian:

$$
X_N(e^{j\omega})
\overset{\text{approximately}}{\sim}
\mathcal{CN}\!\left(0,N R_x(e^{j\omega})\right).
$$

Equivalently,

$$
X_N(e^{j\omega})
\approx\sqrt{N R_x(e^{j\omega})}\,Z,
\qquad Z\sim\mathcal{CN}(0,1).
$$

Consequently,

$$
\hat R_x^{(P)}(e^{j\omega})
\approx R_x(e^{j\omega})|Z|^2,
$$

and hence

$$
E\{\hat R_x^{(P)}(e^{j\omega})\}\approx R_x(e^{j\omega}),
\qquad
\boxed{
\operatorname{var}\{\hat R_x^{(P)}(e^{j\omega})\}
\approx R_x^2(e^{j\omega}).}
$$

The same conclusion follows directly from the fourth moment. Since

$$
\operatorname{var}\{\hat R_x^{(P)}\}
=\frac{1}{N^2}E\{|X_N|^4\}
-\frac{1}{N^2}\left(E\{|X_N|^2\}\right)^2,
$$

and a zero-mean circular complex Gaussian variable satisfies

$$E\{|X_N|^4\}=2\left(E\{|X_N|^2\}\right)^2,$$

using $E\{|X_N|^2\}\approx N R_x(e^{j\omega})$ gives

$$
\begin{aligned}
\operatorname{var}\{\hat R_x^{(P)}(e^{j\omega})\}
&\approx\frac{1}{N^2}
\left[2N^2R_x^2(e^{j\omega})-N^2R_x^2(e^{j\omega})\right]\\
&=R_x^2(e^{j\omega}).
\end{aligned}
$$

The exact numerical constant depends on the signal type and frequency. For
example, for a real Gaussian process the DFT coefficients at DC and, when $N$
is even, at the Nyquist frequency are real rather than circular complex; their
asymptotic variance has a different factor. The important fact is not this
constant but the dependence on $N$: the variance remains of order
$R_x^2(e^{j\omega})$ rather than tending to zero.

Another way to express the problem is through the coefficient of variation:

$$
\frac{
\sqrt{\operatorname{var}\{\hat R_x^{(P)}(e^{j\omega})\}}}
{E\{\hat R_x^{(P)}(e^{j\omega})\}}
\approx 1.
$$

Thus, the standard deviation of a raw periodogram ordinate is approximately as
large as the PSD value being estimated. A longer record narrows the spectral
window and improves frequency resolution, but each ordinate is still formed
from only one random squared Fourier coefficient. No independent power
estimates have been averaged, so the relative random fluctuation is not reduced.
This is why Bartlett and Welch methods reduce variance by averaging multiple
periodograms.

Thus:

$$\boxed{\text{The ordinary periodogram is not a consistent PSD estimator.}}$$

Increasing $N$ makes the frequency grid denser and reduces bias for many spectra, but the estimate remains noisy from record to record.

> ![Figure 2.2](./CourseADSP2026/Fig/Chapter_5/fig_2_5_textbook_fig_5_12_p216.png)
>
> *Figure 2.2 (Textbook Fig. 5.12, p. 216): Periodograms of white Gaussian noise. Even when the true spectrum is flat, individual periodograms fluctuate strongly. Increasing $N$ alone does not eliminate the random variability of the estimate.*

This figure is one of the most important teaching figures in the chapter. Students often expect a longer FFT to automatically produce a “better” spectrum. The figure shows why that is not enough.

---

## 2.2 Window Length, Zero Padding, Leakage, and Resolution

### 2.2.1 DFT Samples Are Samples of a Windowed Spectrum

When we compute an $N$-point DFT, we are not directly observing the infinite-duration spectrum. We are sampling the DTFT of a finite, windowed record.

> ![Figure 2.3](./CourseADSP2026/Fig/Chapter_5/fig_2_1_textbook_fig_5_3_p200.png)
>
> *Figure 2.3 (Textbook Fig. 5.3, p. 200): Effect of window length on the DFT spectrum shape. A finite window changes the apparent spectral shape, and the number of observed periods matters.*

This figure supports a key lesson:

> The DFT grid may look discrete, but the underlying finite-record spectrum is continuous. The DFT only samples it.

If the signal frequency lands exactly on a DFT bin, the DFT may look clean. If it does not, energy spreads across bins. This is spectral leakage.

### 2.2.2 Zero Padding Improves Display, Not True Resolution

Zero padding means appending zeros before computing the DFT. For example, we may have $N$ real data samples but compute an $N_{FFT}$-point FFT with $N_{FFT}>N$.

This gives more frequency samples:

$$\omega_k=\frac{2\pi k}{N_{FFT}},\quad k=0,1,\ldots,N_{FFT}-1.$$

However, zero padding does not increase the amount of information in the data. It interpolates the finite-record spectrum. Therefore:

$$\boxed{\text{Zero padding gives a smoother-looking plot, but it does not improve true frequency resolution.}}$$

True resolution depends on the effective observation length and window mainlobe width, not on the number of FFT display points.

#### Frequency-Grid Spacing Is Not Resolving Power

It is important to distinguish two quantities that are often both informally
called *frequency resolution*:

1. **Frequency-grid spacing** tells us how closely spaced the reported FFT
   samples are.
2. **Resolving power** tells us whether two nearby spectral components produce
   distinguishable peaks.

Suppose that only $N$ nonzero samples have been observed, but zeros are appended
and an $M$-point FFT is computed, where $M>N$. The displayed frequency-grid
spacing is

$$\Delta\omega_{\mathrm{grid}}=\frac{2\pi}{M}.$$

Increasing $M$ genuinely produces more output frequency points and makes this
spacing smaller. However, the zero-padded FFT values are

$$
X_M(k)=\sum_{n=0}^{N-1}x(n)e^{-j(2\pi k/M)n}
=X_N(e^{j\omega})\bigg|_{\omega=2\pi k/M},
$$

where

$$X_N(e^{j\omega})=\sum_{n=0}^{N-1}x(n)e^{-j\omega n}$$

is the same continuous-frequency DTFT of the original finite record. Thus,
changing $M$ does not create a different underlying spectrum. It merely samples
the same DTFT curve at more closely spaced frequencies. In this sense, zero
padding performs exact interpolation of the finite-record spectrum.

By contrast, the width of the spectral feature produced by one sinusoid is
controlled by the data window. For a length-$N$ rectangular window,

$$
W_R(e^{j\omega})
=e^{-j\omega(N-1)/2}
\frac{\sin(N\omega/2)}{\sin(\omega/2)}.
$$

The first zeros around the central peak occur at

$$\omega=\pm\frac{2\pi}{N},$$

so its null-to-null mainlobe width is

$$\Delta\omega_{\mathrm{mainlobe}}=\frac{4\pi}{N}.$$

This width is determined by the number of actually observed samples $N$, not by
the zero-padded FFT length $M$. Other windows have different, usually wider,
mainlobes, but the same principle holds: the effective observation length and
window shape determine resolving power.

For example, suppose $N=64$ samples are available. A 64-point FFT has grid
spacing $2\pi/64$. If the record is zero-padded to $M=1024$, the grid spacing
becomes $2\pi/1024$, but the rectangular-window mainlobe still has its first
zeros at offsets $\pm2\pi/64$. The 1024-point FFT draws that same mainlobe with
many more points; it does not make the mainlobe narrower.

Consequently, if two nearby sinusoids have already merged into one broad
spectral peak because of the finite window, zero padding produces a more
detailed picture of that merged peak rather than separating it into two peaks.
Increasing the actual observation length can narrow the mainlobes and provide
new information with which to separate the sinusoids.

Zero padding is nevertheless useful. The denser grid produces a smoother plot,
reduces the error caused by reading a peak only at coarse FFT-bin locations, and
can improve numerical localization of an isolated peak. These benefits concern
display and interpolation; they do not constitute an increase in the intrinsic
ability to resolve two nearby components.

An analogy is enlarging a blurred digital image. Adding display pixels allows
the blurred shape to be drawn more smoothly, but it does not recover spatial
detail that was never captured. Similarly, increasing $M$ increases the number
of reported frequency samples, whereas increasing the effective observation
length supplies additional data and can improve true spectral resolution.

### 2.2.3 Leakage and the Rectangular Window

A rectangular window has a narrow mainlobe but high sidelobes. This means it can provide good nominal resolution, but strong components leak energy into distant frequencies.

For a length-$N$ rectangular window, the frequency response shape contains the familiar ratio

$$A(\omega)=\frac{\sin(\omega N/2)}{\sin(\omega/2)}.$$

This function has a mainlobe and sidelobes. The mainlobe controls resolution; the sidelobes control leakage.

### 2.2.4 Window Tradeoff: Mainlobe Width versus Sidelobe Level

To reduce leakage, we can use a smoother window, such as Hann, Hamming, Blackman, or Kaiser. These windows reduce sidelobe height but widen the mainlobe.

> ![Figure 2.4](./CourseADSP2026/Fig/Chapter_5/fig_2_3_textbook_fig_5_8_p207.png)
>
> *Figure 2.4 (Textbook Fig. 5.8, p. 207): Time-domain window functions and their frequency-domain characteristics. Smoother windows lower sidelobes but usually broaden the mainlobe.*

The practical rule is:

| Need | Prefer |
|------|--------|
| Separate very close frequencies | Narrow mainlobe, often rectangular or long data window |
| Detect weak signal near strong signal | Low sidelobes, often Hamming, Blackman, Kaiser, or Dolph-Chebyshev |
| General-purpose smooth estimate | Hamming or Hann |
| Tunable sidelobe control | Kaiser |

### 2.2.5 Modified Periodogram

The modified periodogram applies a data window before taking the DFT:

$$X_w(e^{j\omega})=\sum_{n=0}^{L-1}w(n)x(n)e^{-j\omega n}.$$

The normalized modified periodogram is

$$\boxed{\hat R_x^{(MP)}(e^{j\omega})=\frac{1}{U}\vert X_w(e^{j\omega})\vert^2,\quad U=\sum_{n=0}^{L-1}\vert w(n)\vert^2.}$$

Some texts normalize by $LU$ or by the average window power depending on the exact definition of $U$. The essential idea is the same: the normalization compensates for the energy loss introduced by the window.

The modified periodogram reduces leakage but does not solve the variance problem by itself.

> ![Figure 2.5](./CourseADSP2026/Fig/Chapter_5/fig_2_2_textbook_fig_5_7_p205.png)
>
> *Figure 2.5 (Textbook Fig. 5.7, p. 205): Spectrum of three sinusoids using rectangular and Hamming windows. The Hamming window suppresses sidelobes and makes weak components easier to see, but its wider mainlobe reduces resolution.*

### 2.2.6 Frequency Resolution Example

The finite observation length determines whether two nearby spectral peaks can be distinguished.

> ![Figure 2.6](./CourseADSP2026/Fig/Chapter_5/fig_2_6_textbook_fig_5_15_p220.png)
>
> *Figure 2.6 (Textbook Fig. 5.15, p. 220): Frequency resolution property of the periodogram. The longer record has better ability to separate nearby components, but individual periodograms still show random fluctuations.*

The lesson is subtle:

- A longer record improves resolution.
- A single periodogram can still be noisy.
- Averaging reduces variance but usually reduces effective resolution.

---

## 2.3 Bartlett Method: Averaging Nonoverlapping Periodograms

### 2.3.1 Basic Idea

The periodogram has high variance. The simplest way to reduce variance is to average several periodograms.

Suppose the $N$ samples are split into $K$ nonoverlapping segments, each of length $L$:

$$N=KL.$$

For segment $i$, define

$$x_i(n)=x(n+iL),\quad n=0,1,\ldots,L-1.$$

Compute the periodogram of each segment:

$$\hat R_{x,i}^{(P)}(e^{j\omega})=\frac{1}{L}\left\vert\sum_{n=0}^{L-1}x_i(n)e^{-j\omega n}\right\vert^2.$$

Then average:

$$\boxed{\hat R_x^{(B)}(e^{j\omega})=\frac{1}{K}\sum_{i=0}^{K-1}\hat R_{x,i}^{(P)}(e^{j\omega}).}$$

This is the Bartlett method.

### 2.3.2 Variance Reduction

If the segments are approximately independent, averaging $K$ estimates reduces variance by about $K$:

$$\operatorname{var}\{\hat R_x^{(B)}(e^{j\omega})\}\approx \frac{1}{K}\operatorname{var}\{\hat R_x^{(P)}(e^{j\omega})\}.$$

This is the main benefit of Bartlett averaging.

### 2.3.3 Resolution Loss

The price is that each periodogram uses only $L=N/K$ samples. The effective window is shorter, so its mainlobe is wider. Therefore, frequency resolution is reduced.

The tradeoff is:

$$\boxed{K\uparrow \Rightarrow \text{variance decreases, but } L=N/K\downarrow \Rightarrow \text{resolution worsens}.}$$

Bartlett's method is therefore useful when a stable smooth estimate is more important than resolving very close peaks.

---

## 2.4 Welch Method: Windowed and Overlapped Averaging

### 2.4.1 Motivation

Bartlett's method uses nonoverlapping rectangular segments. Welch's method improves it by adding two ideas:

1. apply a data window to each segment,
2. allow adjacent segments to overlap.

This increases the number of averaged segments and reduces leakage.

> ![Figure 2.7](./CourseADSP2026/Fig/Chapter_5/fig_2_8_textbook_fig_5_20_p231.png)
>
> *Figure 2.7 (Textbook Fig. 5.20, p. 231): Pictorial description of the Welch-Bartlett method. Data are divided into overlapping windowed segments, each segment produces a modified periodogram, and the results are averaged.*

### 2.4.2 Formula

Let the segment length be $L$, the hop size be $D$, and the data window be $w(n)$. The $i$th segment is

$$x_i(n)=x(n+iD),\quad n=0,1,\ldots,L-1.$$

The number of segments is approximately

$$K=1+\left\lfloor\frac{N-L}{D}\right\rfloor.$$

The Welch estimate is

$$\boxed{\hat R_x^{(W)}(e^{j\omega})=\frac{1}{K}\sum_{i=0}^{K-1}\frac{1}{U}\left\vert\sum_{n=0}^{L-1}w(n)x_i(n)e^{-j\omega n}\right\vert^2,}$$

where

$$U=\sum_{n=0}^{L-1}\vert w(n)\vert^2.$$

### 2.4.3 Why Overlap Helps

If a tapered window is used, samples near the segment edges receive small weights. Without overlap, many samples are underused. Overlap lets each sample contribute more evenly across windows.

Common choices are:

| Window | Typical Overlap |
|--------|-----------------|
| Hann | 50% |
| Hamming | 50% |
| Blackman | 50% to 75% |
| Kaiser | depends on parameter and application |

Overlap does not create fully independent new data, but it often improves the practical variance of the estimate.

### 2.4.4 Welch versus Bartlett

| Method | Window | Overlap | Variance | Resolution | Leakage |
|--------|--------|---------|----------|------------|---------|
| Bartlett | Usually rectangular | No | Reduced by averaging | Reduced because segment length is shorter | Can be high |
| Welch | Usually tapered | Yes | Usually lower in practice | Controlled by segment length and window | Lower due to windowing |

Welch's method is widely used because it is simple, robust, and available in most signal-processing software.

---

## 2.5 Blackman-Tukey Method: Smoothing through Autocorrelation Windowing

### 2.5.1 Basic Idea

The Blackman-Tukey method starts from the theoretical relationship

$$R_x(e^{j\omega})=\sum_{l=-\infty}^{\infty}r_x(l)e^{-j\omega l}.$$

Since only finite data are available, we estimate $r_x(l)$ for a limited range of lags and then apply a correlation window $v(l)$:

$$\hat r_{x,v}(l)=v(l)\hat r_x(l),\quad \vert l\vert \le L.$$

The PSD estimate is

$$\boxed{\hat R_x^{(BT)}(e^{j\omega})=\sum_{l=-L}^{L}v(l)\hat r_x(l)e^{-j\omega l}.}$$

This is often called autocorrelation windowing or lag-window spectral estimation.

### 2.5.2 Theory and Practical Computation

The textbook contrasts the theoretical smoothing interpretation with the practical computation procedure.

> ![Figure 2.8](./CourseADSP2026/Fig/Chapter_5/fig_2_7_textbook_fig_5_17_p224.png)
>
> *Figure 2.8 (Textbook Fig. 5.17, p. 224): Theory and practice of the Blackman-Tukey method. The method can be interpreted as smoothing a periodogram or as estimating and windowing the autocorrelation before taking a Fourier transform.*

The practical algorithm is:

1. Estimate the autocorrelation sequence from the data.
2. Multiply the autocorrelation estimate by a lag window.
3. Compute the DTFT or FFT of the windowed autocorrelation.

### 2.5.3 Bias-Variance Behavior

The correlation-window length $L$ is the key parameter.

If $L$ is large:

- more correlation lags are used,
- resolution improves,
- variance increases because high-lag autocorrelation estimates are unreliable.

If $L$ is small:

- fewer correlation lags are used,
- variance decreases,
- resolution worsens because the spectrum is more heavily smoothed.

Thus:

$$\boxed{L\uparrow \Rightarrow \text{resolution improves but variance increases}.}$$

This is the same tradeoff as in periodogram averaging, expressed in the autocorrelation domain.

### 2.5.4 Positive Spectrum Issue

A practical caution: because Blackman-Tukey uses a windowed autocorrelation estimate, the resulting spectral estimate is not always guaranteed to be nonnegative unless the lag window and estimation procedure preserve positive semidefiniteness.

The periodogram is always nonnegative because it is a squared magnitude. Blackman-Tukey estimates can sometimes produce small negative values due to estimation and windowing choices.

---

## 2.6 Statistical Comparison of Classical Methods

The major classical estimators can be compared as follows.

| Method | Main Operation | Main Parameter | Bias | Variance | Resolution | Best Use |
|--------|----------------|----------------|------|----------|------------|----------|
| Periodogram | Square DFT magnitude | Record length $N$ | Decreases with $N$ for many spectra | High, does not vanish | Good if $N$ is large | Exploratory display, deterministic spectral lines |
| Modified periodogram | Window then square DFT magnitude | Window type and length | Lower leakage bias | High | Window-dependent | Weak signal near strong signal |
| Bartlett | Average nonoverlapping periodograms | Segment length $L$, number $K$ | More smoothing | Lower by about $K$ | Worse than full-length periodogram | Stable broad PSD estimate |
| Welch | Average overlapping windowed periodograms | $L$, overlap, window | Lower leakage | Lower in practice | Window/segment dependent | General-purpose PSD estimation |
| Blackman-Tukey | Window autocorrelation then transform | Lag window and lag limit $L$ | Controlled by lag window | Controlled by $L$ | Controlled by $L$ | When autocorrelation interpretation is useful |

The following textbook figure compares the effects of autocorrelation windowing and periodogram averaging.

> ![Figure 2.9](./CourseADSP2026/Fig/Chapter_5/fig_2_9_textbook_fig_5_23_p236.png)
>
> *Figure 2.9 (Textbook Fig. 5.23, p. 236): Properties of power spectrum estimators using autocorrelation windowing and periodogram averaging. Both approaches reduce variance by smoothing, but they do so in different domains.*

## 2.7 Practical Example: Ocean Wave Data

The textbook includes real wave data to show how classical methods behave on measured signals.

> ![Figure 2.10](./CourseADSP2026/Fig/Chapter_5/fig_2_10_textbook_fig_5_24_p237.png)
>
> *Figure 2.10 (Textbook Fig. 5.24, p. 237): Ocean wave data. Real-world records are finite, noisy, and not perfectly idealized.*

> ![Figure 2.11](./CourseADSP2026/Fig/Chapter_5/fig_2_11_textbook_fig_5_25_p237.png)
>
> *Figure 2.11 (Textbook Fig. 5.25, p. 237): Spectrum estimation of ocean wave data using Welch and Blackman-Tukey methods.*

The teaching message is that different estimators may agree on broad spectral structure while differing in local detail. In applications, we should interpret fine peaks carefully unless we have enough data or a strong physical model.

---

# §3 Parametric Spectrum Estimation

> 📖 Textbook §9.2-§9.4, connected to Chapter 4

Classical nonparametric methods do not assume a low-order signal model. Parametric methods do assume a model. The basic idea is:

> Fit a model to the data, then compute the spectrum implied by the fitted model.

The advantage is high resolution from short data. The danger is model mismatch.

---

## 3.1 AR Spectrum Estimation

### 3.1.1 AR Model and Spectrum

An AR($P$) process is generated by passing white noise through an all-pole filter:

$$x(n)+\sum_{k=1}^{P}a_kx(n-k)=w(n),$$

or equivalently,

$$A(z)x(n)=w(n),\quad A(z)=1+\sum_{k=1}^{P}a_kz^{-k}.$$

The synthesis filter is

$$H(z)=\frac{1}{A(z)}.$$

Therefore the AR spectrum is

$$\boxed{R_x(e^{j\omega})=\frac{\sigma_w^2}{\vert A(e^{j\omega})\vert^2}.}$$

After estimating $a_1,\ldots,a_P$ and $\sigma_w^2$, the AR spectral estimate is

$$\boxed{\hat R_x^{(AR)}(e^{j\omega})=\frac{\hat\sigma_w^2}{\left\vert 1+\sum_{k=1}^{P}\hat a_ke^{-j\omega k}\right\vert^2}.}$$

This formula comes from the filtering interpretation of an AR process. The
equation $A(z)x(n)=w(n)$ means that the prediction-error filter $A(z)$ whitens
$x(n)$, while the inverse filter $H(z)=1/A(z)$ synthesizes $x(n)$ from white
noise. Since white noise has flat PSD $R_w(e^{j\omega})=\sigma_w^2$, the LTI
filtering rule gives

$$R_x(e^{j\omega})=\lvert H(e^{j\omega})\rvert^2R_w(e^{j\omega})
=\frac{\sigma_w^2}{\vert A(e^{j\omega})\vert^2}.$$

The comparison with nonparametric PSD estimates is a model check. A periodogram,
Welch estimate, or Blackman-Tukey estimate is obtained more directly from the
data and is usually denoted $\hat R_x(e^{j\omega})$. The AR curve
$\hat R_x^{(AR)}(e^{j\omega})$ is different: it is the PSD implied by the fitted
parameters $\hat a_1,\ldots,\hat a_P$ and $\hat\sigma_w^2$. If this model-implied
spectrum matches the true PSD, or a trustworthy data-based PSD estimate, then
the AR model is credible. If it does not match, the AR assumption, the order
$P$, or the estimation method should be questioned.

> **中文理解.** AR 谱不是“天然等于”从数据直接估计出来的 PSD。它是先拟合 AR 模型，再由模型推出的参数化 PSD。把它和 periodogram/Welch 等数据直接估计的 PSD 放在一起比较，是为了判断这个 AR 模型是否真的抓住了数据的频谱结构。

### 3.1.2 Why AR Methods Can Have High Resolution

Poles near the unit circle produce sharp spectral peaks. Therefore, an AR model can represent narrowband resonances with only a few parameters.

This is why AR spectral estimation is often effective for:

- speech formant estimation,
- narrowband radar or sonar components,
- vibration analysis,
- short-record spectral analysis.

But the same property can be dangerous. If the model order is too high or the data are too short, the method may create artificial peaks.

### 3.1.3 Main AR Estimation Methods

| Method | Data Criterion | Stability | Computation | Typical Behavior |
|--------|---------------|-----------|-------------|------------------|
| Autocorrelation / Yule-Walker | Matches autocorrelation lags with a Toeplitz matrix | Stable if estimated autocorrelation matrix is positive definite | Efficient via Levinson-Durbin | Robust, but can smooth or split peaks |
| Covariance | Least-squares prediction error without assuming data outside the window | Not automatically guaranteed | Solves non-Toeplitz equations | High resolution, but can be unstable |
| Modified covariance | Uses both forward and backward prediction errors | Better practical behavior than covariance | More expensive | Good resolution |
| Burg | Minimizes forward/backward errors recursively and enforces stable reflection coefficients | Stable if $\vert k_m\vert < 1$ | Efficient lattice recursion | High resolution for short records, can be sensitive to order |

> ![Figure 3.1](./CourseADSP2026/Fig/Chapter_5/fig_3_1_textbook_fig_9_14_p468.png)
>
> *Figure 3.1 (Textbook Fig. 9.14, p. 468): Monte Carlo comparison of all-pole PSD estimation techniques. Different AR estimation methods can produce noticeably different spectra from short data records.*

### 3.1.4 Model Order Selection

The AR order $P$ controls the number of poles. Too small an order underfits; too large an order overfits.

A common practical approach is to compare information criteria such as AIC or MDL:

$$\operatorname{AIC}(P)=N\log(\hat\sigma_P^2)+2P,$$

$$\operatorname{MDL}(P)=N\log(\hat\sigma_P^2)+P\log N.$$

The exact constants depend on how the parameter count is defined, but the structure is the same:

- the first term rewards good fit,
- the second term penalizes model complexity.

For short records, order selection can still be unreliable. AIC often selects larger models than MDL; MDL is usually more conservative.

---

## 3.2 MA Spectrum Estimation

### 3.2.1 MA Model and Spectrum

An MA($Q$) process is generated by an all-zero filter:

$$x(n)=\sum_{k=0}^{Q}d_kw(n-k).$$

The model polynomial is

$$D(z)=\sum_{k=0}^{Q}d_kz^{-k}.$$

The spectrum is

$$\boxed{R_x(e^{j\omega})=\sigma_w^2\vert D(e^{j\omega})\vert^2.}$$

MA models are good for spectral valleys and finite-duration correlation structure. An MA($Q$) process has autocorrelation equal to zero outside the range $\vert l\vert > Q$.

### 3.2.2 Direct MA Estimation

One direct method estimates the autocorrelation sequence and then solves for MA coefficients that match it. This is nonlinear because the autocorrelation is quadratic in the MA coefficients:

$$r_x(l)=\sigma_w^2\sum_{k=0}^{Q-l}d_{k+l}d_k^\ast,\quad 0\le l\le Q.$$

This makes MA estimation more complicated than AR estimation.

### 3.2.3 Durbin's Indirect Method

A common indirect approach is:

1. fit a high-order AR model to the data,
2. use the AR model to approximate the impulse response or autocorrelation structure,
3. fit a lower-order MA model from that approximation.

The indirect method avoids solving the original nonlinear MA equations directly, but its accuracy depends on the intermediate AR approximation.

---

## 3.3 ARMA Spectrum Estimation

### 3.3.1 ARMA Model and Spectrum

An ARMA($P,Q$) model is

$$A(z)x(n)=D(z)w(n),$$

with

$$A(z)=1+\sum_{k=1}^{P}a_kz^{-k},\quad D(z)=d_0+\sum_{k=1}^{Q}d_kz^{-k}.$$

The spectrum is

$$\boxed{R_x(e^{j\omega})=\sigma_w^2\frac{\vert D(e^{j\omega})\vert^2}{\vert A(e^{j\omega})\vert^2}.}$$

This model can represent both peaks and notches:

- poles create peaks and resonances,
- zeros create spectral valleys and notches.

### 3.3.2 Modified Yule-Walker Idea

The ARMA autocorrelation satisfies equations that become linear in the AR parameters for lags beyond the MA order. This leads to modified Yule-Walker equations.

For an ARMA($P,Q$) process, for sufficiently large $l$, the autocorrelation satisfies

$$r_x(l)+\sum_{k=1}^{P}a_kr_x(l-k)=0,\quad l>Q.$$

This allows estimation of the AR part first. After the AR part is estimated, the MA part can be estimated from the residual or from the remaining autocorrelation structure.

### 3.3.3 Practical Cautions

ARMA estimation can be powerful, but it is more delicate than AR estimation.

| Issue | Why It Matters |
|------|----------------|
| Nonlinear optimization | Joint pole-zero fitting may have local minima |
| Model order selection | Both $P$ and $Q$ must be chosen |
| Stability and invertibility | Poles and zeros must satisfy constraints for meaningful models |
| Short data | Too many parameters can overfit random fluctuations |

In practice, ARMA models are most useful when there is physical reason to expect both resonances and notches.

---

# §4 Minimum-Variance Spectrum Estimation

> 📖 Textbook §9.5 Minimum-Variance Spectrum Estimation

The minimum-variance method, also known as Capon's method in many contexts, is a data-adaptive spectral estimator. It can be understood as an adaptive filter-bank method.

---

## 4.1 From Fixed Filter Bank to Adaptive Filter Bank

The periodogram uses fixed filters. At each frequency, it measures the output power of a narrowband filter centered at that frequency. But those filters are determined only by the window length, not by the data.

Minimum-variance spectrum estimation asks a different question:

> At frequency $\omega$, can we design a filter that passes that frequency with unit gain but minimizes the output power contributed by all other frequencies?

This leads to a constrained optimization problem.

## 4.2 The MV Optimization Problem

Let

$$\mathbf{a}(\omega)=\begin{bmatrix}1 & e^{-j\omega} & e^{-j2\omega} & \cdots & e^{-jp\omega}\end{bmatrix}^T$$

be a length-$(p+1)$ complex exponential vector.

Let $\mathbf{h}(\omega)$ be an FIR filter coefficient vector. We impose the distortionless constraint

$$\mathbf{h}^H(\omega)\mathbf{a}(\omega)=1.$$

This means the filter has unit response at frequency $\omega$.

The output power is

$$E\{\vert y(n)\vert^2\}=\mathbf{h}^H\mathbf{R}_x\mathbf{h}.$$

The MV design problem is

$$\boxed{\min_{\mathbf{h}}\ \mathbf{h}^H\mathbf{R}_x\mathbf{h}\quad \text{subject to}\quad \mathbf{h}^H\mathbf{a}(\omega)=1.}$$

## 4.3 Solution by Lagrange Multipliers

The solution has the form

$$\boxed{\mathbf{h}_{MV}(\omega)=\frac{\mathbf{R}_x^{-1}\mathbf{a}(\omega)}{\mathbf{a}^H(\omega)\mathbf{R}_x^{-1}\mathbf{a}(\omega)}.}$$

Substituting this back into the output power gives

$$\mathbf{h}_{MV}^H\mathbf{R}_x\mathbf{h}_{MV}=\frac{1}{\mathbf{a}^H(\omega)\mathbf{R}_x^{-1}\mathbf{a}(\omega)}.$$

Depending on normalization, the MV spectrum is written as

$$\boxed{\hat R_{MV}(e^{j\omega})=\frac{p+1}{\mathbf{a}^H(\omega)\hat{\mathbf{R}}_x^{-1}\mathbf{a}(\omega)}.}$$

The factor $p+1$ is a normalization convention used in the textbook form. The essential frequency dependence is in the denominator.

## 4.4 Interpretation

The MV method designs a different filter for every frequency. At the frequency being tested, the response is constrained to be one. Everywhere else, the filter is chosen to minimize output power.

This has a useful interpretation:

- If there is strong signal energy at frequency $\omega$, the constrained filter cannot suppress it, so the output power is large.
- If there is no signal energy at frequency $\omega$, the filter can suppress other components, so the output power is small.

Therefore peaks of the MV estimate indicate likely spectral components.

## 4.5 Relation to AR and Fourier Methods

The textbook compares minimum-variance, all-pole, and Fourier-based estimates.

> ![Figure 4.1](./CourseADSP2026/Fig/Chapter_5/fig_4_1_textbook_fig_9_19_p476.png)
>
> *Figure 4.1 (Textbook Fig. 9.19, p. 476): Comparison of minimum-variance, all-pole, and Fourier-based spectrum estimators for different time-window lengths.*

The typical qualitative behavior is:

| Method | Peak Resolution | Variance | Comments |
|--------|-----------------|----------|----------|
| Fourier / periodogram-based | Limited by window | High unless averaged | Simple and robust |
| All-pole / MEM | Can be very sharp | Can be high or sensitive | Excellent if all-pole model is appropriate |
| Minimum variance | Often good but sometimes less sharp than MEM | Often lower than MEM | Adaptive, less model-specific than pure AR |

## 4.6 Choice of MV Order

The MV estimator uses a correlation matrix of size $(p+1)\times(p+1)$. The order $p$ controls the filter length.

If $p$ is too small:

- the filter has too few degrees of freedom,
- resolution may be poor.

If $p$ is too large:

- the estimated correlation matrix may become ill-conditioned,
- variance increases,
- matrix inversion becomes unstable.

Practical remedies include:

1. using a moderate order,
2. averaging correlation estimates,
3. applying diagonal loading,
4. checking whether peaks are stable across parameter choices.

---

# §5 Maximum-Entropy Spectrum Estimation

> 📖 Textbook §9.2.3 Maximum Entropy Method; connection to Burg algorithm

Maximum-entropy spectrum estimation is closely related to AR modeling and Burg's method.

---

## 5.1 Motivation: Autocorrelation Extrapolation

Classical nonparametric methods implicitly choose how to handle unknown autocorrelation lags.

If we estimate only a finite number of autocorrelation values, then what do we assume about the unobserved lags?

A crude answer is: set them to zero. But setting unknown lags to zero is a strong assumption. It can smear or distort narrowband spectral structure.

The maximum-entropy method asks for a more principled extrapolation:

> Among all spectra consistent with the known autocorrelation lags, choose the one with maximum entropy.

For a Gaussian process, maximizing entropy rate corresponds to maximizing

$$\int_{-\pi}^{\pi}\log R_x(e^{j\omega})\,d\omega$$

subject to matching the known autocorrelation constraints.

## 5.2 Maximum-Entropy Solution Is All-Pole

Suppose we know autocorrelation lags

$$r_x(0),r_x(1),\ldots,r_x(P).$$

The maximum-entropy spectrum consistent with these lags has the form

$$\boxed{R_{MEM}(e^{j\omega})=\frac{\sigma_w^2}{\left\vert 1+\sum_{k=1}^{P}a_ke^{-j\omega k}\right\vert^2}.}$$

This is exactly an AR($P$) or all-pole spectrum.

Therefore:

$$\boxed{\text{Maximum-entropy spectrum estimation is equivalent to all-pole spectral estimation.}}$$

This equivalence is one reason AR spectral estimation is so important.

## 5.3 Burg Method and MEM

Burg's method estimates reflection coefficients by minimizing forward and backward prediction errors while preserving stability. It is often associated with maximum-entropy spectral estimation because it produces a stable all-pole model from finite data and avoids explicitly windowing the autocorrelation sequence in the same way as the autocorrelation method.

The Burg recursion works with forward and backward errors:

$$e_m^f(n)=e_{m-1}^f(n)+k_m e_{m-1}^b(n-1),$$

$$e_m^b(n)=e_{m-1}^b(n-1)+k_m^\ast e_{m-1}^f(n).$$

At each order, $k_m$ is chosen to reduce the sum of forward and backward error energies. Stability follows when

$$\vert k_m\vert < 1.$$

## 5.4 MEM versus Classical Estimators

Maximum-entropy spectra can be much sharper than windowed Fourier estimates. This is helpful when:

- the data record is short,
- the signal is narrowband,
- the all-pole model is physically plausible.

But MEM can also create misleading peaks if the model order is too high or if the signal contains true zeros/notches that an all-pole model cannot represent naturally.

| Feature | MEM / AR | Periodogram / Welch |
|---------|----------|---------------------|
| Assumption | All-pole model | No low-order model |
| Resolution | High if model is right | Limited by window/segment length |
| Variance | Can be high for high order | Reduced by averaging |
| Robustness | Sensitive to order and model mismatch | More robust |
| Interpretation | Model-based spectral envelope | Direct finite-data spectral power |

---

# §6 Frequency Estimation and Subspace Methods

> 📖 Textbook §9.6 Harmonic Models and Frequency Estimation

The final part of this chapter studies a more specialized problem: estimating the frequencies of a small number of complex exponentials in noise.

This is not just PSD estimation. It is a parameter-estimation problem.

---

## 6.0 Harmonic Model and Frequency-Estimation Problem

### 6.0.1 Signal Model

The harmonic model is

$$x(n)=\sum_{p=1}^{P}\alpha_p e^{j(\omega_p n+\phi_p)}+w(n),$$

or, using normalized frequency $f_p=\omega_p/(2\pi)$,

$$x(n)=\sum_{p=1}^{P}\alpha_p e^{j2\pi f_p n}+w(n).$$

The unknowns are the frequencies $f_1,\ldots,f_P$ and often the amplitudes $\alpha_p$.

The PSD of such a process consists of spectral lines plus a noise floor.

> ![Figure 6.1](./CourseADSP2026/Fig/Chapter_5/fig_6_1_textbook_fig_9_20_p479.png)
>
> *Figure 6.1 (Textbook Fig. 9.20, p. 479): Spectrum of complex exponentials in noise. Frequency-estimation methods try to locate the line components accurately even when data are finite and noisy.*

### 6.0.2 Vector Form

Define a length-$M$ data vector

$$\mathbf{x}(n)=\begin{bmatrix}x(n)&x(n+1)&\cdots&x(n+M-1)\end{bmatrix}^T.$$

Define the steering vector

$$\mathbf{v}(f)=\begin{bmatrix}1&e^{j2\pi f}&e^{j2\pi 2f}&\cdots&e^{j2\pi(M-1)f}\end{bmatrix}^T.$$

Then the signal component can be written as a linear combination of steering vectors:

$$\mathbf{s}(n)=\sum_{p=1}^{P}\alpha_p e^{j2\pi f_p n}\mathbf{v}(f_p).$$

Let

$$\mathbf{V}=\begin{bmatrix}\mathbf{v}(f_1)&\mathbf{v}(f_2)&\cdots&\mathbf{v}(f_P)\end{bmatrix}.$$

The correlation matrix has the structure

$$\boxed{\mathbf{R}_x=\mathbf{V}\mathbf{A}\mathbf{V}^H+\sigma_w^2\mathbf{I}.}$$

Here $\mathbf{A}$ contains signal powers and cross terms depending on the amplitude assumptions.

---

## 6.1 Eigen-Decomposition and Signal/Noise Subspaces

Suppose $\mathbf{R}_x$ has eigen-decomposition

$$\mathbf{R}_x=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^H.$$

For $P$ complex exponentials in white noise, the eigenvalues separate into:

- $P$ larger eigenvalues associated with the signal subspace,
- $M-P$ smaller eigenvalues equal to or near $\sigma_w^2$ associated with the noise subspace.

Thus we write

$$\mathbf{Q}=\begin{bmatrix}\mathbf{Q}_s&\mathbf{Q}_n\end{bmatrix}.$$

The signal steering vectors lie in the signal subspace:

$$\mathbf{v}(f_p)\in \operatorname{span}(\mathbf{Q}_s).$$

Therefore they are orthogonal to the noise subspace:

$$\boxed{\mathbf{Q}_n^H\mathbf{v}(f_p)=\mathbf{0}.}$$

This orthogonality is the core idea behind Pisarenko, MUSIC, EV, minimum-norm, and related methods.

---

## 6.2 Pisarenko Harmonic Decomposition

### 6.2.1 Basic Idea

Pisarenko harmonic decomposition is the simplest subspace method. In the ideal case with $P=M-1$, the noise subspace has dimension one. Let $\mathbf{q}_{min}$ be the eigenvector associated with the smallest eigenvalue of $\mathbf{R}_x$.

Then for the true frequencies,

$$\mathbf{q}_{min}^H\mathbf{v}(f_p)=0.$$

Therefore, one can form a polynomial from the entries of $\mathbf{q}_{min}$ and find its roots. The phases of roots near the unit circle give frequency estimates.

### 6.2.2 Pisarenko Pseudospectrum

A pseudospectrum can be defined as

$$\boxed{P_{Pis}(f)=\frac{1}{\vert\mathbf{q}_{min}^H\mathbf{v}(f)\vert^2}.}$$

Peaks occur where $\mathbf{v}(f)$ is nearly orthogonal to the noise eigenvector.

> ![Figure 6.2](./CourseADSP2026/Fig/Chapter_5/fig_6_2_textbook_fig_9_21_p483.png)
>
> *Figure 6.2 (Textbook Fig. 9.21, p. 483): Pisarenko pseudospectrum for a sinusoid in noise. Peaks indicate candidate sinusoidal frequencies.*

### 6.2.3 Limitations

Pisarenko's method is historically important but fragile:

1. It uses only one noise eigenvector.
2. It requires accurate knowledge of the number of sinusoids.
3. It is sensitive to noise and finite-sample errors.
4. It works best in idealized settings.

These limitations motivate MUSIC, which uses the entire noise subspace.

---

## 6.3 MUSIC Method

MUSIC stands for **Multiple Signal Classification**. Its central idea is to test every candidate frequency $f$ by measuring how close $\mathbf{v}(f)$ is to the noise subspace.

### 6.3.1 MUSIC Pseudospectrum

Let $\mathbf{Q}_n$ contain the estimated noise eigenvectors. Define

$$\boxed{P_{MUSIC}(f)=\frac{1}{\mathbf{v}^H(f)\mathbf{Q}_n\mathbf{Q}_n^H\mathbf{v}(f)}.}$$

Equivalently,

$$P_{MUSIC}(f)=\frac{1}{\sum_{i=P+1}^{M}\vert\mathbf{q}_i^H\mathbf{v}(f)\vert^2}.$$

At a true frequency, $\mathbf{v}(f_p)$ is orthogonal to the noise subspace, so the denominator becomes small and the pseudospectrum has a peak.

### 6.3.2 MUSIC Algorithm

The practical MUSIC procedure is:

1. Build data vectors of length $M$.
2. Estimate the correlation matrix $\hat{\mathbf{R}}_x$.
3. Compute the eigen-decomposition of $\hat{\mathbf{R}}_x$.
4. Choose the number of sinusoids $P$.
5. Form the noise-subspace matrix $\hat{\mathbf{Q}}_n$.
6. Evaluate $P_{MUSIC}(f)$ over a frequency grid.
7. Estimate frequencies from the largest peaks.

### 6.3.3 MUSIC versus MV

Both MV and MUSIC use inverse or eigenspace information from a correlation matrix, but they answer different questions.

| Method | Quantity Plotted | Meaning of Peak |
|--------|------------------|-----------------|
| MV | Estimated output power of constrained adaptive filter | Frequency with large power under distortionless constraint |
| MUSIC | Reciprocal distance to noise subspace | Frequency whose steering vector lies in signal subspace |

MUSIC is not a true PSD estimate. It is a pseudospectrum for frequency localization.

---

## 6.4 Eigenvector and Minimum-Norm Methods

The textbook also discusses other eigenvector-based methods.

### 6.4.1 Eigenvector Method

The eigenvector method weights the contributions of different noise eigenvectors. One form is

$$P_{EV}(f)=\frac{1}{\sum_{i=P+1}^{M}\frac{1}{\lambda_i}\vert\mathbf{q}_i^H\mathbf{v}(f)\vert^2}.$$

The weighting can improve performance in some finite-sample settings.

### 6.4.2 Minimum-Norm Method

The minimum-norm method chooses a vector in the noise subspace with a special normalization, often designed to sharpen peaks and reduce spurious behavior.

The details are more specialized, but the conceptual connection is simple:

> It is another way to exploit the fact that true frequency steering vectors are orthogonal to the noise subspace.

> ![Figure 6.3](./CourseADSP2026/Fig/Chapter_5/fig_6_3_textbook_fig_9_23_p488.png)
>
> *Figure 6.3 (Textbook Fig. 9.23, p. 488): Comparison of eigendecomposition-based frequency-estimation methods: Pisarenko, MUSIC, eigenvector method, and minimum-norm method.*

This figure is useful for teaching because it shows that different subspace methods can produce different peak sharpness and sidelobe behavior, even though they share the same signal/noise subspace principle.

---

## 6.5 ESPRIT Method

ESPRIT stands for **Estimation of Signal Parameters via Rotational Invariance Techniques**. It is a subspace method that avoids spectral peak search.

### 6.5.1 Rotational Invariance Principle

For a single complex exponential,

$$s(n+1)=s(n)e^{j2\pi f}.$$

A one-sample shift multiplies the signal by a phase factor. For multiple exponentials, the same idea becomes a matrix relation involving a diagonal matrix of phase factors:

$$\boldsymbol{\Phi}=\operatorname{diag}\{e^{j2\pi f_1},e^{j2\pi f_2},\ldots,e^{j2\pi f_P}\}.$$

If we can estimate the shift relation between two overlapping subarrays or subwindows, then the eigenvalues of the shift operator reveal the frequencies.

### 6.5.2 ESPRIT Processing Flow

> ![Figure 6.4](./CourseADSP2026/Fig/Chapter_5/fig_6_4_textbook_fig_9_24_p489.png)
>
> *Figure 6.4 (Textbook Fig. 9.24, p. 489): Flow of the ESPRIT algorithm from the data matrix to frequency estimates.*

A typical LS-ESPRIT algorithm is:

1. Form a data matrix from overlapping time-window vectors.
2. Compute its SVD or estimate the correlation matrix and compute its signal subspace.
3. Partition the signal subspace into two overlapping parts, $\mathbf{U}_1$ and $\mathbf{U}_2$.
4. Solve

   $$\mathbf{U}_2\approx \mathbf{U}_1\boldsymbol{\Psi}$$

   in a least-squares sense.
5. Compute the eigenvalues $\lambda_p$ of $\boldsymbol{\Psi}$.
6. Estimate frequencies by

   $$\boxed{\hat f_p=\frac{1}{2\pi}\arg(\lambda_p).}$$

### 6.5.3 Time-Staggered Windows

> ![Figure 6.5](./CourseADSP2026/Fig/Chapter_5/fig_6_5_textbook_fig_9_25_p490.png)
>
> *Figure 6.5 (Textbook Fig. 9.25, p. 490): Time-staggered overlapping windows used by ESPRIT. The shift invariance between overlapping windows is the structural property exploited by the algorithm.*

The figure shows two length-$(M-1)$ subwindows inside a length-$M$ window. One subwindow starts at $n$, and the other starts at $n+1$. For complex exponentials, this one-sample shift corresponds to multiplication by $e^{j2\pi f_p}$.

### 6.5.4 LS-ESPRIT and TLS-ESPRIT

In LS-ESPRIT, the shift operator is estimated by ordinary least squares:

$$\hat{\boldsymbol{\Psi}}_{LS}=(\mathbf{U}_1^H\mathbf{U}_1)^{-1}\mathbf{U}_1^H\mathbf{U}_2.$$

In TLS-ESPRIT, errors in both $\mathbf{U}_1$ and $\mathbf{U}_2$ are considered. TLS can improve performance when both subspace partitions are noisy.

### 6.5.5 ESPRIT versus MUSIC

| Method | Frequency Search? | Main Requirement | Strength | Weakness |
|--------|-------------------|------------------|----------|----------|
| MUSIC | Yes, peak search over frequency grid | Accurate noise subspace and model order | Very high resolution, intuitive pseudospectrum | Grid search, peak picking, spurious peaks |
| Root-MUSIC | Polynomial root search | Uniform structure | More accurate than grid MUSIC in some cases | More specialized |
| ESPRIT | No spectral grid search | Shift-invariant data structure | Direct frequency estimates, efficient | Requires matched overlapping subarrays/windows |

In array processing, MUSIC and ESPRIT become direction-of-arrival estimators. In time-series analysis, the same mathematics estimates sinusoidal frequencies.

---

# §7 Chapter Summary, Method Selection, and Figure Checklist

## 7.1 One-Sentence Summary of Each Method

| Method | One-Sentence Description |
|--------|--------------------------|
| Periodogram | Estimate PSD by squared magnitude of the finite-record DFT. |
| Modified periodogram | Apply a data window before the periodogram to reduce leakage. |
| Bartlett | Split data into nonoverlapping segments and average periodograms to reduce variance. |
| Welch | Use overlapping windowed segments and average modified periodograms. |
| Blackman-Tukey | Estimate autocorrelation, apply a lag window, then transform to frequency domain. |
| AR spectrum | Fit an all-pole model and compute $\sigma_w^2/\vert A(e^{j\omega})\vert^2$. |
| MA spectrum | Fit an all-zero model and compute $\sigma_w^2\vert D(e^{j\omega})\vert^2$. |
| ARMA spectrum | Fit a pole-zero model and compute $\sigma_w^2\vert D(e^{j\omega})\vert^2/\vert A(e^{j\omega})\vert^2$. |
| MV spectrum | Use a data-adaptive filter with unit gain at the test frequency and minimum output variance. |
| MEM | Choose the all-pole spectrum with maximum entropy subject to autocorrelation constraints. |
| MUSIC | Locate frequencies by finding steering vectors orthogonal to the noise subspace. |
| ESPRIT | Estimate frequencies from the eigenvalues of a shift-invariance operator. |

## 7.2 How to Choose a Spectrum Estimator

### Case 1: General-Purpose PSD Display

Use Welch.

Recommended settings:

- choose a segment length $L$ long enough for desired resolution,
- use a Hann or Hamming window,
- use around 50% overlap,
- use enough FFT points for a smooth display, while remembering that zero padding does not improve true resolution.

### Case 2: Weak Component Near Strong Component

Use a low-sidelobe data window or a method with stronger leakage control.

Options:

- modified periodogram with Hamming, Blackman, Kaiser, or Dolph-Chebyshev window,
- Welch with a low-sidelobe window,
- MV if a data-adaptive approach is appropriate.

### Case 3: Short Data and Narrowband Peaks

Consider AR, MEM/Burg, MV, MUSIC, or ESPRIT.

But check carefully:

- Is the model order reasonable?
- Are peaks stable across order choices?
- Is the SNR high enough?
- Does the physics support sinusoidal or all-pole modeling?

### Case 4: Frequency Estimation of Sinusoids

Use subspace methods if the model is appropriate:

- MUSIC for high-resolution frequency localization with a pseudospectrum,
- ESPRIT for direct estimates without grid search,
- Root-MUSIC if the uniform structure is available and polynomial rooting is preferred.

## 7.3 Common Misunderstandings

| Misunderstanding | Correction |
|------------------|-----------|
| A longer FFT always gives better resolution. | Zero padding gives denser frequency samples but does not add information. |
| The periodogram becomes smooth as $N$ grows. | Its variance does not vanish; averaging or smoothing is needed. |
| A sharper peak always means a better estimate. | Sharp peaks can be artifacts of model mismatch or overfitting. |
| Welch destroys information because segments are shorter. | It trades resolution for variance reduction; this is often desirable. |
| MUSIC is a PSD estimator. | MUSIC produces a pseudospectrum for frequency localization, not a power-calibrated PSD. |
| AR models are always better than nonparametric methods for short data. | They are better only if the all-pole assumption is appropriate and the order is chosen well. |

## 7.4 Teaching Flow for This Chapter

A good lecture sequence is:

1. Start with the finite-data problem and Figure 1.1.
2. Explain that windowing causes leakage and mainlobe broadening.
3. Define the periodogram and show its high variance using Figure 2.2.
4. Introduce modified periodograms as leakage reduction.
5. Introduce Bartlett and Welch as variance reduction by averaging.
6. Introduce Blackman-Tukey as autocorrelation-domain smoothing.
7. Compare classical methods using Figure 2.9.
8. Move to parametric AR/MA/ARMA spectra.
9. Introduce MV as a data-adaptive filter-bank estimator.
10. Explain MEM as maximum-entropy/all-pole extrapolation.
11. Finish with subspace frequency estimation: signal/noise subspaces, MUSIC, and ESPRIT.

## 7.5 Figure Checklist

All figures used in this chapter are screenshots from the textbook and are stored in:

```text
./CourseADSP2026/Fig/Chapter_5/
```

| Lecture Figure | Textbook Figure | File |
|----------------|-----------------|------|
| Figure 0.1 | Fig. 5.1, p. 196 | `fig_0_1_textbook_fig_5_1_p196.png` |
| Figure 1.1 | Fig. 5.2, p. 197 | `fig_1_1_textbook_fig_5_2_p197.png` |
| Figure 2.1 | Fig. 5.11, p. 215 | `fig_2_4_textbook_fig_5_11_p215.png` |
| Figure 2.2 | Fig. 5.12, p. 216 | `fig_2_5_textbook_fig_5_12_p216.png` |
| Figure 2.3 | Fig. 5.3, p. 200 | `fig_2_1_textbook_fig_5_3_p200.png` |
| Figure 2.4 | Fig. 5.8, p. 207 | `fig_2_3_textbook_fig_5_8_p207.png` |
| Figure 2.5 | Fig. 5.7, p. 205 | `fig_2_2_textbook_fig_5_7_p205.png` |
| Figure 2.6 | Fig. 5.15, p. 220 | `fig_2_6_textbook_fig_5_15_p220.png` |
| Figure 2.7 | Fig. 5.20, p. 231 | `fig_2_8_textbook_fig_5_20_p231.png` |
| Figure 2.8 | Fig. 5.17, p. 224 | `fig_2_7_textbook_fig_5_17_p224.png` |
| Figure 2.9 | Fig. 5.23, p. 236 | `fig_2_9_textbook_fig_5_23_p236.png` |
| Figure 2.10 | Fig. 5.24, p. 237 | `fig_2_10_textbook_fig_5_24_p237.png` |
| Figure 2.11 | Fig. 5.25, p. 237 | `fig_2_11_textbook_fig_5_25_p237.png` |
| Figure 3.1 | Fig. 9.14, p. 468 | `fig_3_1_textbook_fig_9_14_p468.png` |
| Figure 4.1 | Fig. 9.19, p. 476 | `fig_4_1_textbook_fig_9_19_p476.png` |
| Figure 6.1 | Fig. 9.20, p. 479 | `fig_6_1_textbook_fig_9_20_p479.png` |
| Figure 6.2 | Fig. 9.21, p. 483 | `fig_6_2_textbook_fig_9_21_p483.png` |
| Figure 6.3 | Fig. 9.23, p. 488 | `fig_6_3_textbook_fig_9_23_p488.png` |
| Figure 6.4 | Fig. 9.24, p. 489 | `fig_6_4_textbook_fig_9_24_p489.png` |
| Figure 6.5 | Fig. 9.25, p. 490 | `fig_6_5_textbook_fig_9_25_p490.png` |

## 7.6 Final Concept Map

The whole chapter can be remembered by the following chain:

$$\text{finite data}\Rightarrow \text{windowing}\Rightarrow \text{leakage and variance}\Rightarrow \text{smoothing/averaging/modeling}.$$

Classical methods handle the problem by smoothing or averaging. Parametric methods handle it by imposing a signal model. Minimum-variance methods handle it by adaptive filtering. Subspace methods handle it by exploiting the geometry of sinusoidal steering vectors.

The most important practical lesson is:

$$\boxed{\text{A spectrum estimate is always a tradeoff, not a direct picture of truth.}}$$
