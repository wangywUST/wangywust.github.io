# Modern Digital Signal Processing
## Chapter 6: Optimum Linear Filtering — Wiener Filters and Kalman Filters

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005  
> Chapters covered: Ch. 6 (Optimum Linear Filters, §6.1–§6.10) · §7.8 (Kalman Filter Algorithm)

---

## Table of Contents

1. [§0 Chapter Roadmap: From Optimum Filtering to Recursive State Estimation](#0-chapter-roadmap-from-optimum-filtering-to-recursive-state-estimation)
2. [§1 Linear MMSE Estimation: The Mathematical Core of Wiener Filtering](#1-linear-mmse-estimation-the-mathematical-core-of-wiener-filtering)
3. [§2 FIR Wiener Filters](#2-fir-wiener-filters)
4. [§3 Important FIR Wiener Filtering Applications](#3-important-fir-wiener-filtering-applications)
5. [§4 Optimum IIR Wiener Filters](#4-optimum-iir-wiener-filters)
6. [§5 Matched Filters and Eigenfilters](#5-matched-filters-and-eigenfilters)
7. [§6 Discrete Kalman Filtering](#6-discrete-kalman-filtering)
8. [§7 Chapter Summary, Method Selection, and Figure Checklist](#7-chapter-summary-method-selection-and-figure-checklist)

---

## Notation and Variable Definitions

This chapter connects three levels of ideas:

1. **Linear MMSE estimation**: estimate one random variable from several related random variables.
2. **Wiener filtering**: estimate a desired signal from an observed signal by an optimum linear filter.
3. **Kalman filtering**: estimate a time-varying state recursively from noisy measurements and a dynamic model.

The notation below follows the previous lectures: bold lower-case letters denote vectors, bold upper-case letters denote matrices, and superscript $H$ denotes Hermitian transpose.

### Time, Order, and Delays

| Symbol | Definition |
|--------|------------|
| $n$ | Discrete-time sample index |
| $M$ | FIR filter order or number of data samples used by a linear estimator |
| $D$ | Decision delay or deconvolution delay |
| $\Delta$ | Prediction distance, usually $\Delta>0$ |
| $k,l$ | Lag or summation indices |
| $N$ | Number of samples in a finite realization or simulation |

### Signals and Observations

| Symbol | Definition |
|--------|------------|
| $x(n)$ | Observed/input signal available to the filter |
| $y(n)$ | Desired response signal to be estimated |
| $\hat y(n)$ | Estimate of $y(n)$ produced by the filter |
| $e(n)=y(n)-\hat y(n)$ | Estimation error |
| $v(n)$ | Additive observation noise or interference |
| $d(n)$ | Desired clean signal in some filtering/noise-canceling examples |
| $g(n)$ | Impulse response of a distorting channel or system |
| $a(n)$ | Transmitted symbol sequence in a communication equalization model |
| $w(n)$ | White innovation or driving noise, depending on context |

### Linear Estimator and FIR Filter Quantities

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}$ | Generic data vector used for estimating a scalar desired response |
| $\mathbf{x}(n)$ | FIR data vector, usually $[x(n),x(n-1),\ldots,x(n-M+1)]^T$ |
| $\mathbf{c}$ | Linear estimator coefficient vector |
| $\mathbf{h}$ | FIR Wiener filter coefficient vector |
| $\mathbf{R}=E\{\mathbf{x}\mathbf{x}^H\}$ | Data correlation matrix |
| $\mathbf{d}=E\{\mathbf{x}y^\ast\}$ | Cross-correlation vector between the data and the desired response |
| $P(c)$ | Mean-square error as a function of estimator coefficients |
| $P_o$ | Minimum mean-square error (MMSE) |
| $P_y=E\{|y|^2\}$ | Desired-response power |

### Correlation and Spectral Quantities

| Symbol | Definition |
|--------|------------|
| $r_x(l)$ | Autocorrelation sequence of $x(n)$ |
| $r_y(l)$ | Autocorrelation sequence of $y(n)$ |
| $r_{yx}(l)=E\{y(n)x^\ast(n-l)\}$ | Cross-correlation from $x$ to $y$ |
| $R_x(e^{j\omega})$ | Power spectral density of $x(n)$ |
| $R_{yx}(e^{j\omega})$ | Cross-power spectral density between $y(n)$ and $x(n)$ |
| $H(e^{j\omega})$ | Frequency response of a Wiener filter |
| $H_x(z)$ | Minimum-phase spectral factor of $R_x(z)$ |
| $[\cdot]_+$ | Causal / one-sided part of a Laurent series |

### Kalman Filtering Quantities

| Symbol | Definition |
|--------|------------|
| $\mathbf{s}(n)$ | State vector at time $n$ |
| $\mathbf{y}(n)$ | Observation vector at time $n$ in the Kalman section |
| $\mathbf{A}(n)$ | State transition matrix |
| $\mathbf{C}(n)$ | Observation matrix |
| $\boldsymbol{\eta}(n)$ | Process driving noise |
| $\mathbf{v}(n)$ | Observation noise |
| $\mathbf{Q}(n)$ | Process-noise covariance |
| $\mathbf{R}_v(n)$ | Observation-noise covariance |
| $\hat{\mathbf{s}}(n|n-1)$ | A priori state estimate before using observation at time $n$ |
| $\hat{\mathbf{s}}(n|n)$ | A posteriori state estimate after using observation at time $n$ |
| $\mathbf{P}(n|n-1)$ | A priori estimation-error covariance |
| $\mathbf{P}(n|n)$ | A posteriori estimation-error covariance |
| $\mathbf{K}(n)$ | Kalman gain |
| $\boldsymbol{\nu}(n)$ | Innovation / measurement residual |

---

# §0 Chapter Roadmap: From Optimum Filtering to Recursive State Estimation

> 📖 Textbook §6.1 (Optimum Signal Estimation); §6.2 (Linear Mean Square Error Estimation); §7.8 (Kalman Filter Algorithm)

## 0.1 What Problem Are We Solving?

In a conventional deterministic DSP course, a filter is often designed by specifying a desired frequency response. For example, we may design a low-pass filter, a high-pass filter, or a band-pass filter. The filter design target is then mostly stated in the frequency domain.

In this chapter, the starting point is different. We are given random signals. One of them is the signal we can observe, and another is the signal we want. The key question is:

> **Among all linear filters of a chosen structure, which filter gives the smallest mean-square estimation error?**

The filter is called **optimum** not because it is universally best, but because it is best under a clearly specified criterion and a clearly specified set of statistical assumptions.

The criterion is the mean-square error:

$$
P=E\{|e(n)|^2\}=E\{|y(n)-\hat y(n)|^2\}.
$$

Here:

- $y(n)$ is the desired response;
- $\hat y(n)$ is the filter output or estimate;
- $e(n)$ is the estimation error;
- the expectation is taken over the ensemble of possible signal realizations.

This is the core idea behind the **Wiener filter**.

## 0.2 Four Main Wiener Filtering Problems

The same mathematical framework can describe several important signal processing tasks.

| Problem | Goal | Typical Form |
|---------|------|--------------|
| Filtering | Estimate the current desired signal from noisy observations | Estimate $y(n)$ from $x(n),x(n-1),\ldots$ |
| Smoothing | Estimate a sample using past and future observations | Estimate $y(n)$ from all available $x(k)$ |
| Prediction | Estimate a future value | Estimate $x(n+\Delta)$ from current and past samples |
| Deconvolution / equalization | Undo a distorting system in noise | Estimate source symbols from channel output |

The distinction between these problems is not the algebraic criterion. They all use MMSE ideas. The distinction is **which desired response is chosen** and **which samples are allowed in the data vector**.

For example:

- In filtering, $y(n)$ is usually a clean version of a noisy observation.
- In prediction, $y(n)$ is replaced by a future sample, such as $x(n+1)$.
- In equalization, $y(n)$ may be a delayed transmitted symbol, such as $a(n-D)$.
- In smoothing, noncausal samples may be allowed, such as $x(n+1),x(n+2),\ldots$.

The textbook begins by emphasizing that the data used by an estimator may come from different sensors or from delayed samples of a single time series.

> ![Figure 0.1](./CourseADSP2026/Fig/Chapter_6/fig_0_1_textbook_fig_6_1_p262.png)
>
> *Figure 0.1 (Textbook Fig. 6.1, p. 262): Data vectors for array processing and for FIR filtering/prediction. In this lecture we mainly use the single-sensor time-series case, where the vector is made from delayed samples of one signal.*

## 0.3 Why Mean-Square Error?

Many error measures are possible. We could penalize $|e|$, $|e|^2$, $|e|^3$, or even a nonconvex application-specific loss. The squared error has two major advantages:

1. It strongly penalizes large errors.
2. It leads to a simple linear algebra solution when the estimator is linear.

This second point is extremely important. The MMSE criterion turns the design of optimum linear filters into the solution of linear equations.

> ![Figure 0.2](./CourseADSP2026/Fig/Chapter_6/fig_0_2_textbook_fig_6_2_p263.png)
>
> *Figure 0.2 (Textbook Fig. 6.2, p. 263): Different error-weighting functions. The squared-error criterion is a compromise between mathematical tractability and strong penalty for large errors.*

A useful way to remember the chapter is:

$$
\boxed{\text{Statistics} + \text{linear structure} + \text{MSE criterion}
\quad\Longrightarrow\quad
\text{normal equations}.}
$$

---

# §1 Linear MMSE Estimation: The Mathematical Core of Wiener Filtering

> 📖 Textbook §6.2 (Linear Mean Square Error Estimation)

## 1.1 General Linear Estimator

Suppose we want to estimate a scalar desired response $y$ from $M$ random variables

$$
x_1,x_2,\ldots,x_M.
$$

Collect the data into a vector

$$
\mathbf{x}=[x_1,x_2,\ldots,x_M]^T.
$$

A linear estimator has the form

$$
\boxed{\hat y=\mathbf{c}^H\mathbf{x}}
$$

where

$$
\mathbf{c}=[c_1,c_2,\ldots,c_M]^T
$$

is the coefficient vector. For complex signals, the Hermitian transpose is needed so that the estimate is a scalar inner product.

The error is

$$
e=y-\hat y=y-\mathbf{c}^H\mathbf{x}.
$$

The goal is to choose $\mathbf{c}$ to minimize

$$
P(\mathbf{c})=E\{|e|^2\}.
$$

The block diagram is a linear combiner: each data component is weighted, the weighted terms are added, and the result is compared with the desired response.

> ![Figure 1.1](./CourseADSP2026/Fig/Chapter_6/fig_1_1_textbook_fig_6_3_p265.png)
>
> *Figure 1.1 (Textbook Fig. 6.3, p. 265): Block diagram of the linear estimator. This is the algebraic template behind FIR Wiener filters, matched filters, and Kalman measurement updates.*

## 1.2 Expanding the MSE

Starting from

$$
e=y-\mathbf{c}^H\mathbf{x},
$$

the MSE is

$$
\begin{aligned}
P(\mathbf{c})
&=E\{(y-\mathbf{c}^H\mathbf{x})(y^\ast-\mathbf{x}^H\mathbf{c})\} \\
&=E\{|y|^2\}-\mathbf{c}^H E\{\mathbf{x}y^\ast\}-E\{y\mathbf{x}^H\}\mathbf{c}+\mathbf{c}^H E\{\mathbf{x}\mathbf{x}^H\}\mathbf{c}.
\end{aligned}
$$

Define

$$
\boxed{P_y=E\{|y|^2\}},
$$

$$
\boxed{\mathbf{d}=E\{\mathbf{x}y^\ast\}},
$$

and

$$
\boxed{\mathbf{R}=E\{\mathbf{x}\mathbf{x}^H\}}.
$$

Then

$$
\boxed{P(\mathbf{c})=P_y-\mathbf{c}^H\mathbf{d}-\mathbf{d}^H\mathbf{c}+\mathbf{c}^H\mathbf{R}\mathbf{c}}.
$$

This equation is the most important algebraic expression in Wiener filtering. It says that, for a linear estimator, the entire performance surface depends only on **second-order moments**:

- $P_y$;
- the cross-correlation vector $\mathbf{d}$;
- the data correlation matrix $\mathbf{R}$.

No higher-order distributions are needed for the linear MMSE solution.

## 1.3 The Error Performance Surface

For a second-order estimator with two real coefficients, $P(c_1,c_2)$ is a quadratic surface. If $\mathbf{R}$ is positive definite, the surface is bowl-shaped and has one unique minimum. If $\mathbf{R}$ is not positive definite, the surface may be flat or saddle-shaped, and the minimization problem may become ill-posed.

> ![Figure 1.2](./CourseADSP2026/Fig/Chapter_6/fig_1_2_textbook_fig_6_4_p266.png)
>
> *Figure 1.2 (Textbook Fig. 6.4, p. 266): Quadratic error-performance surfaces. The positive-definite case gives a unique bowl-shaped minimum; the indefinite case does not define a proper MMSE optimum.*

This figure is pedagogically important because it shows why the correlation matrix matters. The normal equations may be written down algebraically, but a stable and unique solution requires $\mathbf{R}$ to be positive definite or at least nonsingular on the relevant subspace.

## 1.4 Deriving the Normal Equations

To minimize $P(\mathbf{c})$, differentiate with respect to the complex coefficient vector. The result is

$$
\boxed{\mathbf{R}\mathbf{c}_o=\mathbf{d}}.
$$

These are the **normal equations**. The optimum linear estimator is therefore

$$
\boxed{\mathbf{c}_o=\mathbf{R}^{-1}\mathbf{d}}
$$

when $\mathbf{R}$ is nonsingular.

The corresponding minimum MSE is

$$
\boxed{P_o=P_y-\mathbf{d}^H\mathbf{c}_o}
$$

or equivalently

$$
\boxed{P_o=P_y-\mathbf{d}^H\mathbf{R}^{-1}\mathbf{d}}.
$$

This formula has a very clear interpretation:

$$
\text{minimum error power}
=	ext{desired signal power}-\text{power explained by the optimum linear estimate}.
$$

If $\mathbf{x}$ carries no information about $y$, then $\mathbf{d}=\mathbf{0}$ and $P_o=P_y$. The filter cannot help.

If $y$ is perfectly linearly determined by $\mathbf{x}$, then the second term can equal $P_y$ and $P_o=0$.

## 1.5 Orthogonality Principle

The same result can be understood geometrically. At the optimum, the error must be orthogonal to every data component:

$$
\boxed{E\{\mathbf{x}e_o^\ast\}=\mathbf{0}}.
$$

Since

$$
e_o=y-\mathbf{c}_o^H\mathbf{x},
$$

we obtain

$$
E\{\mathbf{x}y^\ast\}-E\{\mathbf{x}\mathbf{x}^H\}\mathbf{c}_o=\mathbf{0},
$$

which is exactly

$$
\mathbf{d}-\mathbf{R}\mathbf{c}_o=\mathbf{0}.
$$

Thus the normal equations and the orthogonality principle are the same statement in two forms.

> ![Figure 1.3](./CourseADSP2026/Fig/Chapter_6/fig_1_3_textbook_fig_6_9_p273.png)
>
> *Figure 1.3 (Textbook Fig. 6.9, p. 273): Geometric interpretation of the orthogonality principle. The optimum estimate is the projection of the desired response onto the subspace spanned by the data variables.*

The picture is easiest to understand in Euclidean geometry. The desired vector $y$ is projected onto the plane spanned by the data vectors. The residual $e_o$ is perpendicular to that plane. In random signal processing, the “inner product” is an expected correlation:

$$
\langle u,v\rangle=E\{u v^\ast\}.
$$

So orthogonality means uncorrelatedness in the second-order sense.

## 1.6 Two Essential Properties to Remember

The linear MMSE estimator has two central properties:

**Property 1: It uses only second-order statistics.**  
The solution depends on $\mathbf{R}$ and $\mathbf{d}$, not on the full probability distribution.

**Property 2: Its error is orthogonal to the data.**  
The filter has extracted all linear information about $y$ that is present in $\mathbf{x}$.

These two facts will reappear in every section:

- FIR Wiener filtering uses a data vector made of delayed samples.
- IIR Wiener filtering uses spectral factorization to handle infinitely many samples.
- Matched filtering uses the same principle to maximize SNR.
- Kalman filtering uses a recursive version of the same projection idea.

---

# §2 FIR Wiener Filters

> 📖 Textbook §6.4 (Optimum Finite Impulse Response Filters); §6.5 (Linear Prediction)

## 2.1 From Linear Estimator to FIR Filter

An FIR Wiener filter is just a linear MMSE estimator whose data vector is made from delayed samples of an input signal.

For an $M$-tap FIR filter, define

$$
\mathbf{x}(n)=[x(n),x(n-1),\ldots,x(n-M+1)]^T.
$$

The estimate is

$$
\boxed{\hat y(n)=\mathbf{h}^H\mathbf{x}(n)=\sum_{k=0}^{M-1}h^\ast(k)x(n-k)}.
$$

The error is

$$
\boxed{e(n)=y(n)-\hat y(n)}.
$$

The filter design problem is

$$
\boxed{\min_{\mathbf{h}} E\{|y(n)-\mathbf{h}^H\mathbf{x}(n)|^2\}}.
$$

By the result from §1, the optimum coefficients satisfy

$$
\boxed{\mathbf{R}\mathbf{h}_o=\mathbf{d}},
$$

where

$$
\mathbf{R}=E\{\mathbf{x}(n)\mathbf{x}^H(n)\},
$$

and

$$
\mathbf{d}=E\{\mathbf{x}(n)y^\ast(n)\}.
$$

> ![Figure 2.1](./CourseADSP2026/Fig/Chapter_6/fig_2_1_textbook_fig_6_10_p278.png)
>
> *Figure 2.1 (Textbook Fig. 6.10, p. 278): General optimum filtering problem. The filter produces an estimate of the desired response and the error is minimized in the MMSE sense.*

## 2.2 Stationary FIR Wiener-Hopf Equations

If $x(n)$ and $y(n)$ are jointly wide-sense stationary, then the entries of $\mathbf{R}$ and $\mathbf{d}$ do not depend on absolute time. They depend only on lags.

The correlation matrix has Toeplitz form:

$$
\mathbf{R}=\begin{bmatrix}
r_x(0) & r_x(1) & \cdots & r_x(M-1) \\
r_x^\ast(1) & r_x(0) & \cdots & r_x(M-2) \\
\vdots & \vdots & \ddots & \vdots \\
r_x^\ast(M-1) & r_x^\ast(M-2) & \cdots & r_x(0)
\end{bmatrix}.
$$

The right-hand-side vector is

$$
\mathbf{d}=\begin{bmatrix}
r_{xy}(0) \\
r_{xy}(1) \\
\vdots \\
r_{xy}(M-1)
\end{bmatrix}
$$

up to the cross-correlation convention used. The essential point is that $\mathbf{d}$ stores the correlation between the available input samples and the desired response.

The normal equations

$$
\boxed{\mathbf{R}\mathbf{h}_o=\mathbf{d}}
$$

are called the **Wiener-Hopf equations**.

The minimum MSE is

$$
\boxed{P_o=P_y-\mathbf{d}^H\mathbf{h}_o}.
$$

## 2.3 Design and Implementation Are Separate

A very important practical distinction is:

| Step | What is used? | Output |
|------|---------------|--------|
| Design | Known or estimated second-order statistics $\mathbf{R}$ and $\mathbf{d}$ | Optimum coefficients $\mathbf{h}_o$ |
| Implementation | Input samples $x(n)$ | Estimated signal $\hat y(n)$ |

During actual filtering, the desired response $y(n)$ is not required. It is required only for training, modeling, or statistical estimation.

> ![Figure 2.2](./CourseADSP2026/Fig/Chapter_6/fig_2_2_textbook_fig_6_11_p280.png)
>
> *Figure 2.2 (Textbook Fig. 6.11, p. 280): Design and implementation of a time-varying optimum FIR filter. The a priori statistics determine the coefficients; the filtering structure then operates on the input signal.*

For a stationary problem, $\mathbf{R}$ and $\mathbf{d}$ are fixed. We solve once and then use the resulting FIR filter.

For a nonstationary problem, the moments may change with time. Then the optimum coefficients also change with time:

$$
\mathbf{R}(n)\mathbf{h}_o(n)=\mathbf{d}(n).
$$

This is one motivation for adaptive filtering and Kalman filtering.

## 2.4 Why Filter Order Matters

Increasing the FIR order gives the filter more degrees of freedom. In general, the MMSE cannot increase when more useful data samples are allowed. However, longer filters also require more computation and more reliable estimates of the correlation matrix.

The textbook example shows that as the FIR order $M$ increases:

- the MMSE decreases;
- the processing gain increases;
- after a certain order, improvement becomes small.

> ![Figure 2.3](./CourseADSP2026/Fig/Chapter_6/fig_2_3_textbook_fig_6_12_p283.png)
>
> *Figure 2.3 (Textbook Fig. 6.12, p. 283): MMSE and processing gain versus FIR filter order. The useful lesson is that filter order should be chosen by the accuracy-complexity tradeoff, not by making it as large as possible.*

The **processing gain** is often defined as an output SNR divided by an input SNR:

$$
\boxed{PG=\frac{\mathrm{SNR}_{\mathrm{out}}}{\mathrm{SNR}_{\mathrm{in}}}}.
$$

In dB form,

$$
PG_{\mathrm{dB}}=10\log_{10}\mathrm{SNR}_{\mathrm{out}}-10\log_{10}\mathrm{SNR}_{\mathrm{in}}.
$$

For teaching, it is useful to emphasize that MMSE and SNR are related but not identical. MMSE measures estimation error power. SNR measures useful signal power relative to disturbance power. In many filtering problems, improving one improves the other, but the exact interpretation depends on how the desired response is defined.

## 2.5 Frequency-Domain Interpretation of FIR Wiener Filtering

For stationary processes, the MSE can also be interpreted in the frequency domain. If a noncausal filter were allowed, the optimum frequency response would have the intuitive form

$$
\boxed{H_{nc}(e^{j\omega})=\frac{R_{yx}(e^{j\omega})}{R_x(e^{j\omega})}}.
$$

This says:

> At each frequency, pass the part of $x$ that is correlated with $y$, and suppress the part that is not useful for estimating $y$.

For additive noise filtering, where

$$
x(n)=y(n)+v(n)
$$

and $y(n)$ and $v(n)$ are uncorrelated, we have

$$
R_x(e^{j\omega})=R_y(e^{j\omega})+R_v(e^{j\omega}),
$$

and

$$
R_{yx}(e^{j\omega})=R_y(e^{j\omega}).
$$

Therefore the noncausal Wiener filter becomes

$$
\boxed{H_{nc}(e^{j\omega})=\frac{R_y(e^{j\omega})}{R_y(e^{j\omega})+R_v(e^{j\omega})}}.
$$

This formula is extremely intuitive:

- if signal power is much larger than noise power, $H_{nc}\approx 1$;
- if noise power is much larger than signal power, $H_{nc}\approx 0$;
- the filter is not simply a fixed low-pass or band-pass filter; it depends on the signal and noise spectra.

> ![Figure 2.4](./CourseADSP2026/Fig/Chapter_6/fig_2_4_textbook_fig_6_14_p284.png)
>
> *Figure 2.4 (Textbook Fig. 6.14, p. 284): PSD of input signal, magnitude response of the optimum filter, and PSD of the filtered output. The optimum filter emphasizes the frequency region where the desired component is statistically reliable.*

## 2.6 FIR Wiener Filtering Algorithm

For a stationary FIR Wiener filter, the practical algorithm is:

1. Choose the filter length $M$.
2. Estimate or derive $r_x(l)$ for $l=0,1,\ldots,M-1$.
3. Estimate or derive $r_{xy}(l)$ for the required lags.
4. Form $\mathbf{R}$ and $\mathbf{d}$.
5. Solve

$$
\mathbf{R}\mathbf{h}_o=\mathbf{d}.
$$

6. Implement

$$
\hat y(n)=\mathbf{h}_o^H\mathbf{x}(n).
$$

7. Evaluate

$$
P_o=P_y-\mathbf{d}^H\mathbf{h}_o.
$$

The main numerical issue is the conditioning of $\mathbf{R}$. If the data samples are highly correlated or if too high an order is chosen from too little data, $\mathbf{R}$ may be ill-conditioned. This is one reason why regularization, diagonal loading, and adaptive algorithms become important in practice.

---

# §3 Important FIR Wiener Filtering Applications

> 📖 Textbook §6.4 (Optimum FIR Filters); §6.5 (Linear Prediction); §6.7–§6.8 (Inverse Filtering, Deconvolution, and Equalization)

The same Wiener-Hopf equation solves many different problems. The only thing that changes is the definition of the desired response $y(n)$ and the data vector $\mathbf{x}(n)$.

## 3.1 Application 1: Additive-Noise Filtering

Suppose

$$
\boxed{x(n)=d(n)+v(n)}
$$

where

- $d(n)$ is the desired clean signal;
- $v(n)$ is additive noise;
- $d(n)$ and $v(n)$ are uncorrelated.

We want

$$
y(n)=d(n).
$$

The FIR Wiener filter estimates $d(n)$ from noisy samples of $x(n)$:

$$
\hat d(n)=\sum_{k=0}^{M-1}h_o^\ast(k)x(n-k).
$$

The required correlations are

$$
r_x(l)=r_d(l)+r_v(l),
$$

and

$$
r_{xd}(l)=E\{x(n-l)d^\ast(n)\}=r_d(l)
$$

if $d$ and $v$ are uncorrelated.

Therefore, the normal equation is built from the noisy observation autocorrelation but the right-hand side comes from the clean signal correlation.

In practice, this is difficult because $r_d(l)$ and $r_v(l)$ are not always known. A model or a training period may be required.

## 3.2 Application 2: Linear Prediction

Linear prediction is a Wiener filtering problem where the desired response is a future sample.

For one-step prediction, define

$$
y(n)=x(n+1).
$$

Use the data vector

$$
\mathbf{x}(n)=[x(n),x(n-1),\ldots,x(n-M+1)]^T.
$$

Then

$$
\hat x(n+1)=\mathbf{h}_o^H\mathbf{x}(n).
$$

The right-hand-side vector becomes

$$
\mathbf{d}=E\{\mathbf{x}(n)x^\ast(n+1)\}.
$$

For a WSS process, its entries are determined by autocorrelation lags. For example, using a common convention,

$$
\mathbf{d}=\begin{bmatrix}
r_x(-1) \\
r_x(-2) \\
\vdots \\
r_x(-M)
\end{bmatrix}.
$$

For $\Delta$-step prediction,

$$
y(n)=x(n+\Delta),
$$

and the same form holds with shifted lags.

The textbook also distinguishes linear signal estimation, forward prediction, and backward prediction using the same data samples arranged in different roles.

> ![Figure 3.1](./CourseADSP2026/Fig/Chapter_6/fig_3_1_textbook_fig_6_16_p287.png)
>
> *Figure 3.1 (Textbook Fig. 6.16, p. 287): Linear signal estimation, forward linear prediction, and backward linear prediction. These are different choices of which sample is treated as the desired response.*

The key conceptual link is:

$$
\boxed{\text{prediction} = \text{Wiener filtering with a future desired response}.}
$$

This is why Chapter 3 linear prediction and Chapter 6 Wiener filtering are not separate subjects. Linear prediction is one of the most important special cases of optimum filtering.

## 3.3 Application 3: Noise Cancellation with an Auxiliary Reference

A common noise-cancellation setup has a primary observation

$$
x_0(n)=d(n)+v_1(n),
$$

where $d(n)$ is the desired signal and $v_1(n)$ is noise. We also observe a reference signal

$$
u(n)=v_2(n),
$$

which is correlated with $v_1(n)$ but ideally uncorrelated with $d(n)$.

The idea is not to estimate $d(n)$ directly. Instead, we estimate the noise $v_1(n)$ from the reference $u(n)$:

$$
\hat v_1(n)=\sum_{k=0}^{M-1}h^\ast(k)u(n-k).
$$

Then subtract it from the primary signal:

$$
e(n)=x_0(n)-\hat v_1(n).
$$

If $u(n)$ is uncorrelated with $d(n)$, minimizing

$$
E\{|x_0(n)-\hat v_1(n)|^2\}
$$

causes the filter to remove the component of $x_0(n)$ that is linearly predictable from the reference. This is the basis of adaptive noise cancellation.

The Wiener-Hopf equation is still

$$
\mathbf{R}_u\mathbf{h}_o=\mathbf{d}_{ux_0},
$$

where

$$
\mathbf{R}_u=E\{\mathbf{u}(n)\mathbf{u}^H(n)\}
$$

and

$$
\mathbf{d}_{ux_0}=E\{\mathbf{u}(n)x_0^\ast(n)\}.
$$

This example is important because it shows a major practical strength of Wiener filtering: sometimes the desired clean signal is never observed directly, but a useful reference signal is available.

## 3.4 Application 4: Deconvolution and MMSE Equalization

Suppose the desired input passes through a distorting LTI system:

$$
x(n)=g(n)*d(n)+v(n).
$$

Here:

- $d(n)$ is the original source;
- $g(n)$ is the channel or blur impulse response;
- $v(n)$ is additive noise;
- $x(n)$ is the observed distorted signal.

The goal of deconvolution is to estimate $d(n)$ from $x(n)$.

If there were no noise and if $G(z)$ were exactly invertible, we could use the inverse system

$$
H(z)=\frac{1}{G(z)}.
$$

But in practice this may fail because:

1. $G(z)$ may have zeros near or outside the unit circle.
2. The inverse may be unstable or noncausal.
3. The inverse may greatly amplify noise at frequencies where $G(e^{j\omega})$ is small.

The MMSE approach solves a more realistic problem:

$$
\boxed{\hat d(n-D)=\sum_k h^\ast(k)x(n-k)}
$$

where $D$ is a delay chosen to make the inverse response more realizable.

> ![Figure 3.2](./CourseADSP2026/Fig/Chapter_6/fig_4_3_textbook_fig_6_24_p307.png)
>
> *Figure 3.2 (Textbook Fig. 6.24, p. 307): Optimum inverse system modeling. The delay $D$ is introduced because the best inverse response may be noncausal or may need time alignment.*

The corresponding Wiener-Hopf equation is still

$$
\mathbf{R}_x\mathbf{h}_o=\mathbf{d},
$$

but now $\mathbf{d}$ contains correlations between the received distorted signal and the delayed desired source.

### Zero-Forcing Equalizer versus MMSE Equalizer

A **zero-forcing equalizer** tries to satisfy

$$
H(z)G(z)\approx z^{-D}.
$$

This removes intersymbol interference if the channel model is exact and noise is negligible.

An **MMSE equalizer** minimizes

$$
E\{|a(n-D)-\hat a(n)|^2\}.
$$

It accepts a controlled amount of residual distortion if that avoids excessive noise amplification.

This is the key practical difference:

| Equalizer | Main Objective | Weakness |
|----------|----------------|----------|
| Zero-forcing | Remove channel distortion exactly | Can strongly amplify noise |
| MMSE / Wiener | Balance residual ISI and noise enhancement | Needs statistical information |

## 3.5 Data Transmission and ISI

In digital communications, the received sample can contain contributions from neighboring transmitted symbols. This is called **intersymbol interference** (ISI).

The textbook uses a baseband pulse amplitude modulation model.

> ![Figure 3.3](./CourseADSP2026/Fig/Chapter_6/fig_4_4_textbook_fig_6_27_p310.png)
>
> *Figure 3.3 (Textbook Fig. 6.27, p. 310): Baseband PAM communication model and input symbol sequence. A channel, receiving filter, sampler, and detector together determine the equivalent discrete-time problem.*

After sampling and noise whitening, a useful equivalent model is

$$
\boxed{x(n)=\sum_{k=0}^{L}a_k h_r(n-k)+v(n)}.
$$

This is a finite-memory channel with white Gaussian noise.

> ![Figure 3.4](./CourseADSP2026/Fig/Chapter_6/fig_4_5_textbook_fig_6_31_p314.png)
>
> *Figure 3.4 (Textbook Fig. 6.31, p. 314): Equivalent discrete-time channel model with ISI and white Gaussian noise.*

A linear equalizer is then an FIR filter that estimates the desired symbol from received samples.

> ![Figure 3.5](./CourseADSP2026/Fig/Chapter_6/fig_4_6_textbook_fig_6_32_p315.png)
>
> *Figure 3.5 (Textbook Fig. 6.32, p. 315): Equalizer-based receiver model. The equalizer can be viewed in continuous time or in the equivalent discrete-time model.*

During a training phase, the receiver may know the transmitted symbol sequence. Then the desired response can be chosen as a delayed version of the training symbol:

$$
y(n)=a(n-D).
$$

The equalizer output is

$$
\hat y(n)=\sum_{k=-M}^{M}c^\ast(k)x(n-k).
$$

The error is

$$
e(n)=a(n-D)-\hat y(n).
$$

> ![Figure 3.6](./CourseADSP2026/Fig/Chapter_6/fig_4_7_textbook_fig_6_33_p317.png)
>
> *Figure 3.6 (Textbook Fig. 6.33, p. 317): Data communication model used to design an MMSE equalizer with a training sequence and delay.*

The delay $D$ is not a cosmetic detail. It determines which part of the overall channel response the equalizer tries to align with the transmitted symbol. A poor delay can force the equalizer to implement an unnecessarily difficult inverse.

---

# §4 Optimum IIR Wiener Filters

> 📖 Textbook §6.6 (Optimum Infinite Impulse Response Filters); §6.7 (Inverse Filtering and Deconvolution)

FIR Wiener filters use a finite number of data samples. IIR Wiener filters allow, at least conceptually, infinitely many samples. This can improve performance, but it introduces causality and spectral-factorization issues.

## 4.1 Noncausal IIR Wiener Filter: Performance Upper Bound

If the filter is allowed to be noncausal, the problem is easiest in the frequency domain. For jointly WSS processes, the optimum noncausal IIR filter is

$$
\boxed{H_{nc}(e^{j\omega})=\frac{R_{yx}(e^{j\omega})}{R_x(e^{j\omega})}}.
$$

Equivalently in the $z$ domain,

$$
\boxed{H_{nc}(z)=\frac{R_{yx}(z)}{R_x(z)}}.
$$

The MMSE is

$$
\boxed{P_{nc}=r_y(0)-\frac{1}{2\pi}\int_{-\pi}^{\pi}\frac{|R_{yx}(e^{j\omega})|^2}{R_x(e^{j\omega})}\,d\omega}.
$$

This is a theoretical performance limit for linear filtering because the noncausal filter may use future samples. It may not be physically realizable in real time, but it tells us how well any linear filter could do if causality were not a constraint.

## 4.2 Why the Causal IIR Filter Is Harder

A causal IIR Wiener filter may only use present and past samples:

$$
x(n),x(n-1),x(n-2),\ldots
$$

The direct frequency-domain ratio

$$
\frac{R_{yx}(z)}{R_x(z)}
$$

may contain noncausal terms, unstable factors, or both. Therefore, the causal solution requires spectral factorization.

The main idea is:

1. Factor the input spectrum into a minimum-phase causal system and its conjugate reciprocal.
2. Use the inverse of the causal factor to whiten the input.
3. Design the best causal filter for the whitened input.
4. Cascade the whitening and causal filtering parts.

## 4.3 Spectral Factorization Design

Assume that the input spectrum can be written as

$$
\boxed{R_x(z)=\sigma_x^2 H_x(z)H_x^\ast(1/z^\ast)}
$$

where $H_x(z)$ is causal, stable, and minimum phase.

The corresponding whitening filter is

$$
\frac{1}{H_x(z)}.
$$

The whitened process is

$$
w(n)=\frac{1}{H_x(z)}x(n).
$$

The causal optimum IIR filter has the form

$$
\boxed{H_c(z)=\frac{1}{\sigma_x^2 H_x(z)}
\left[\frac{R_{yx}(z)}{H_x^\ast(1/z^\ast)}\right]_+.}
$$

The notation $[\cdot]_+$ means: keep only the causal part of the Laurent series.

> ![Figure 4.1](./CourseADSP2026/Fig/Chapter_6/fig_4_1_textbook_fig_6_18_p298.png)
>
> *Figure 4.1 (Textbook Fig. 6.18, p. 298): Causal IIR Wiener filter design using spectral factorization. The input is first whitened, then a causal optimum filter is applied.*

## 4.4 Causal versus Noncausal IIR Filters

The noncausal optimum filter can use both past and future information. The causal optimum filter can use only current and past samples.

The difference is visible in the decomposition of the term

$$
\frac{R_{yx}(z)}{H_x^\ast(1/z^\ast)}.
$$

The noncausal filter uses the full two-sided expression; the causal filter keeps only the causal part.

> ![Figure 4.2](./CourseADSP2026/Fig/Chapter_6/fig_4_2_textbook_fig_6_19_p299.png)
>
> *Figure 4.2 (Textbook Fig. 6.19, p. 299): Comparison of causal and noncausal IIR optimum filters. The causal filter is realizable in real time; the noncausal filter provides a useful performance benchmark.*

A common source of confusion is the phrase “optimum filter.” We must always ask:

- optimum among FIR filters of length $M$?
- optimum among all causal filters?
- optimum among all noncausal filters?
- optimum under the assumed second-order statistics?

These are different optimization classes. Their best achievable MMSE values are generally different.

## 4.5 Linear Prediction Using the Infinite Past

For one-step prediction, the desired response is

$$
y(n)=x(n+1).
$$

If infinitely many past samples are allowed, the prediction problem is closely related to whitening and innovations.

The prediction error is the part of $x(n+1)$ that cannot be linearly predicted from the infinite past. This unpredictable component is the **innovation**.

For a purely regular stochastic process, the innovation has nonzero variance. This means that even with the entire infinite past, the next sample cannot be predicted perfectly.

For an AR($p$) process, the optimum predictor is finite order:

$$
x(n)+\sum_{k=1}^{p}a_k x(n-k)=w(n).
$$

The whitening filter is

$$
A(z)=1+\sum_{k=1}^{p}a_k z^{-k}.
$$

Therefore the prediction error is exactly the white innovation $w(n)$.

The key conclusion is:

$$
\boxed{\text{A prediction error filter is a whitening filter for an AR process.}}
$$

This connects Chapter 3 linear prediction, Chapter 4 AR modeling, Chapter 5 spectral estimation, and Chapter 6 Wiener filtering.

## 4.6 IIR Deconvolution and Inverse Filtering

The IIR theory is especially useful for inverse filtering. If a signal passes through a system $G(z)$ and is corrupted by noise, a perfect inverse is usually not the best practical solution.

The noncausal Wiener inverse filter has the general form

$$
\boxed{H_{nc}(z)=z^{-D}\frac{R_{yx}(z)}{R_x(z)}}
$$

where $D$ is a delay used to align the recovered signal.

The causal version again requires the causal part after spectral factorization.

The practical teaching point is:

> **Inverse filtering is dangerous when a channel has spectral nulls. Wiener deconvolution avoids infinite noise amplification by balancing inverse response and noise suppression.**

---

# §5 Matched Filters and Eigenfilters

> 📖 Textbook §6.9 (Matched Filters and Eigenfilters)

Matched filtering is often taught in deterministic signal detection. In this textbook, it is connected to optimum linear filtering and second-order statistics.

## 5.1 Deterministic Signal in Colored Noise

Suppose the observation vector is

$$
\mathbf{x}=\mathbf{s}+\mathbf{v},
$$

where

- $\mathbf{s}$ is a known deterministic signal vector;
- $\mathbf{v}$ is zero-mean noise with covariance $\mathbf{R}_v$.

A linear filter output is

$$
z=\mathbf{c}^H\mathbf{x}.
$$

The output signal amplitude is

$$
\mathbf{c}^H\mathbf{s},
$$

and the output noise power is

$$
\mathbf{c}^H\mathbf{R}_v\mathbf{c}.
$$

The output SNR is therefore

$$
\mathrm{SNR}(\mathbf{c})=\frac{|\mathbf{c}^H\mathbf{s}|^2}{\mathbf{c}^H\mathbf{R}_v\mathbf{c}}.
$$

The filter that maximizes this SNR is proportional to

$$
\boxed{\mathbf{c}_o=\mathbf{R}_v^{-1}\mathbf{s}}.
$$

If the noise is white, $\mathbf{R}_v=\sigma_v^2\mathbf{I}$, then

$$
\mathbf{c}_o\propto \mathbf{s}.
$$

This is the familiar matched filter: match the filter to the signal.

If the noise is colored, the optimum filter first accounts for the noise covariance. It is better understood as a **whitened matched filter**.

> ![Figure 5.1](./CourseADSP2026/Fig/Chapter_6/fig_5_1_textbook_fig_6_35_p321.png)
>
> *Figure 5.1 (Textbook Fig. 6.35, p. 321): Signal and optimum matched-filter impulse responses in colored noise. When noise is highly correlated, the optimum filter shape can differ strongly from the signal shape.*

## 5.2 Random Signal in Noise and Eigenfilters

If the signal is random, the signal power at the filter output is

$$
\mathbf{c}^H\mathbf{R}_s\mathbf{c},
$$

and the noise power is

$$
\mathbf{c}^H\mathbf{R}_v\mathbf{c}.
$$

The output SNR is a ratio of quadratic forms:

$$
\boxed{\mathrm{SNR}(\mathbf{c})=rac{\mathbf{c}^H\mathbf{R}_s\mathbf{c}}{\mathbf{c}^H\mathbf{R}_v\mathbf{c}}.}
$$

If $\mathbf{R}_v=\sigma_v^2\mathbf{I}$, maximizing this ratio reduces to choosing the eigenvector of $\mathbf{R}_s$ associated with the largest eigenvalue.

If $\mathbf{R}_v$ is not white, the problem becomes a generalized eigenvalue problem:

$$
\boxed{\mathbf{R}_s\mathbf{c}=\lambda \mathbf{R}_v\mathbf{c}.}
$$

The best filter is the generalized eigenvector corresponding to the largest generalized eigenvalue.

> ![Figure 5.2](./CourseADSP2026/Fig/Chapter_6/fig_5_2_textbook_fig_6_36_p323.png)
>
> *Figure 5.2 (Textbook Fig. 6.36, p. 323): Geometric interpretation of eigenfilter optimization. The optimum direction maximizes one quadratic form subject to another quadratic constraint.*

## 5.3 Interference Rejection Filters

The textbook also compares matched filtering, linear prediction error filtering, and binomial filters for rejecting interference.

> ![Figure 5.3](./CourseADSP2026/Fig/Chapter_6/fig_5_3_textbook_fig_6_37_p324.png)
>
> *Figure 5.3 (Textbook Fig. 6.37, p. 324): Frequency responses of matched, prediction-error, and binomial interference-rejection filters. Different filters embody different assumptions about the interference and desired signal.*

The main lesson is not that one filter is always best. The lesson is that the optimum structure depends on the statistical model:

- matched filter: strongest when the desired waveform is known;
- eigenfilter: useful when signal and noise correlation matrices are known;
- prediction-error filter: useful when interference is predictable and the desired component is less predictable;
- binomial filter: simple high-pass-like rejection of low-frequency or stationary clutter.

---

# §6 Discrete Kalman Filtering

> 📖 Textbook §7.8 (Kalman Filter Algorithm)

Wiener filtering assumes that the relevant second-order statistics are known. For stationary processes, the optimum filter may be fixed. But many real signals are not stationary. Examples include:

- radar target tracking;
- vehicle navigation;
- time-varying communication channels;
- sensor fusion;
- financial time series;
- biological or physiological measurements.

Kalman filtering addresses this situation by using a **state-space model**. Instead of assuming a fixed stationary covariance, it assumes that the signal evolves according to a dynamic model.

## 6.1 Why Wiener Filtering Is Not Enough

A Wiener filter is usually designed from correlation functions such as

$$
r_x(l),\quad r_{yx}(l).
$$

For a WSS process, these depend only on lag. But if the process is nonstationary, correlations can change with time:

$$
r_x(n,l),\quad r_{yx}(n,l).
$$

A fixed Wiener filter may then be inappropriate.

Kalman filtering replaces the stationary-correlation viewpoint with a recursive dynamic model:

1. predict the current state from the previous state;
2. compare the predicted observation with the actual observation;
3. correct the prediction using a gain chosen by MMSE principles.

This is why the Kalman filter can be viewed as a time-recursive MMSE estimator.

## 6.2 State-Space Model

Use the following state model:

$$
\boxed{\mathbf{s}(n)=\mathbf{A}(n-1)\mathbf{s}(n-1)+\mathbf{B}(n)\boldsymbol{\eta}(n)}
$$

and the observation model:

$$
\boxed{\mathbf{y}(n)=\mathbf{C}(n)\mathbf{s}(n)+\mathbf{v}(n)}.
$$

Here:

- $\mathbf{s}(n)$ is the hidden state;
- $\mathbf{y}(n)$ is the measurement;
- $\boldsymbol{\eta}(n)$ is process noise;
- $\mathbf{v}(n)$ is observation noise;
- $\mathbf{A}(n)$ describes how the state evolves;
- $\mathbf{C}(n)$ describes how the state is observed.

Assume

$$
E\{\boldsymbol{\eta}(n)\boldsymbol{\eta}^H(n)\}=\mathbf{Q}(n),
$$

and

$$
E\{\mathbf{v}(n)\mathbf{v}^H(n)\}=\mathbf{R}_v(n).
$$

The process noise and observation noise are usually assumed uncorrelated with each other and with the initial state.

> ![Figure 6.1](./CourseADSP2026/Fig/Chapter_6/fig_6_1_textbook_fig_7_11_p384.png)
>
> *Figure 6.1 (Textbook Fig. 7.11, p. 384): Kalman filter model and algorithm. The filter alternates between model-based prediction and measurement-based correction.*

## 6.3 Prediction Step

At time $n-1$, suppose we already have the best estimate after using all observations up to time $n-1$:

$$
\hat{\mathbf{s}}(n-1|n-1).
$$

The one-step prediction is

$$
\boxed{\hat{\mathbf{s}}(n|n-1)=\mathbf{A}(n-1)\hat{\mathbf{s}}(n-1|n-1)}.
$$

The prediction error is

$$
\tilde{\mathbf{s}}(n|n-1)=\mathbf{s}(n)-\hat{\mathbf{s}}(n|n-1).
$$

Its covariance is

$$
\boxed{\mathbf{P}(n|n-1)=\mathbf{A}(n-1)\mathbf{P}(n-1|n-1)\mathbf{A}^H(n-1)
+\mathbf{B}(n)\mathbf{Q}(n)\mathbf{B}^H(n).}
$$

This equation is a Riccati-type covariance prediction equation. It says:

- uncertainty from the previous time propagates through the dynamic model;
- new process noise adds uncertainty.

## 6.4 Innovation and Kalman Gain

Given the predicted state, the predicted measurement is

$$
\hat{\mathbf{y}}(n|n-1)=\mathbf{C}(n)\hat{\mathbf{s}}(n|n-1).
$$

The measurement residual, or innovation, is

$$
\boxed{\boldsymbol{\nu}(n)=\mathbf{y}(n)-\mathbf{C}(n)\hat{\mathbf{s}}(n|n-1)}.
$$

The innovation covariance is

$$
\boxed{\mathbf{S}(n)=\mathbf{C}(n)\mathbf{P}(n|n-1)\mathbf{C}^H(n)+\mathbf{R}_v(n)}.
$$

The cross-covariance between the state prediction error and the innovation is

$$
\mathbf{P}(n|n-1)\mathbf{C}^H(n).
$$

The Kalman gain is therefore

$$
\boxed{\mathbf{K}(n)=\mathbf{P}(n|n-1)\mathbf{C}^H(n)\mathbf{S}^{-1}(n)}.
$$

This is the same structure as a Wiener filter:

$$
\boxed{\text{gain} = \text{cross-covariance} \times \text{inverse observation covariance}.}
$$

So the Kalman gain is not a heuristic tuning factor. It is the MMSE-optimal linear gain for correcting the prediction using the current innovation.

## 6.5 Update Step

The corrected state estimate is

$$
\boxed{\hat{\mathbf{s}}(n|n)=\hat{\mathbf{s}}(n|n-1)+\mathbf{K}(n)\boldsymbol{\nu}(n)}.
$$

Equivalently,

$$
\boxed{\hat{\mathbf{s}}(n|n)=\hat{\mathbf{s}}(n|n-1)+\mathbf{K}(n)
\left[\mathbf{y}(n)-\mathbf{C}(n)\hat{\mathbf{s}}(n|n-1)\right].}
$$

The a posteriori covariance update is

$$
\boxed{\mathbf{P}(n|n)=\left[\mathbf{I}-\mathbf{K}(n)\mathbf{C}(n)\right]\mathbf{P}(n|n-1).}
$$

A numerically more stable Joseph-form update is often written as

$$
\mathbf{P}(n|n)=
[\mathbf{I}-\mathbf{K}(n)\mathbf{C}(n)]\mathbf{P}(n|n-1)[\mathbf{I}-\mathbf{K}(n)\mathbf{C}(n)]^H
+\mathbf{K}(n)\mathbf{R}_v(n)\mathbf{K}^H(n).
$$

The textbook derivation minimizes a scalar measure of covariance, often

$$
\mathrm{tr}\{\mathbf{P}(n|n)\},
$$

which is the sum of posterior state estimation error variances.

## 6.6 Complete Kalman Filter Recursion

A compact Kalman filtering loop is:

### Initialization

Choose

$$
\boxed{\hat{\mathbf{s}}(0|0)=E\{\mathbf{s}(0)\}}
$$

and

$$
\boxed{\mathbf{P}(0|0)=E\{[\mathbf{s}(0)-\hat{\mathbf{s}}(0|0)][\mathbf{s}(0)-\hat{\mathbf{s}}(0|0)]^H\}}.
$$

If no observation has yet been processed, one may instead initialize $\hat{\mathbf{s}}(0|-1)$ and $\mathbf{P}(0|-1)$.

### For each $n=1,2,\ldots$

Predict:

$$
\hat{\mathbf{s}}(n|n-1)=\mathbf{A}(n-1)\hat{\mathbf{s}}(n-1|n-1),
$$

$$
\mathbf{P}(n|n-1)=\mathbf{A}(n-1)\mathbf{P}(n-1|n-1)\mathbf{A}^H(n-1)+\mathbf{B}(n)\mathbf{Q}(n)\mathbf{B}^H(n).
$$

Compute innovation:

$$
\boldsymbol{\nu}(n)=\mathbf{y}(n)-\mathbf{C}(n)\hat{\mathbf{s}}(n|n-1),
$$

$$
\mathbf{S}(n)=\mathbf{C}(n)\mathbf{P}(n|n-1)\mathbf{C}^H(n)+\mathbf{R}_v(n).
$$

Compute gain:

$$
\mathbf{K}(n)=\mathbf{P}(n|n-1)\mathbf{C}^H(n)\mathbf{S}^{-1}(n).
$$

Update:

$$
\hat{\mathbf{s}}(n|n)=\hat{\mathbf{s}}(n|n-1)+\mathbf{K}(n)\boldsymbol{\nu}(n),
$$

$$
\mathbf{P}(n|n)=\left[\mathbf{I}-\mathbf{K}(n)\mathbf{C}(n)\right]\mathbf{P}(n|n-1).
$$

This recursion is computationally attractive because it does not require storing all past data. All information from the past is summarized by two quantities:

$$
\hat{\mathbf{s}}(n|n),\qquad \mathbf{P}(n|n).
$$

## 6.7 Scalar Example: Kalman Estimation of an AR(1) Process

Consider the scalar state model

$$
\boxed{s(n)=a s(n-1)+\eta(n)}
$$

and observation model

$$
\boxed{y(n)=s(n)+v(n)}.
$$

Assume

$$
\eta(n)\sim WGN(0,q),\qquad v(n)\sim WGN(0,r),
$$

and $\eta(n)$ and $v(n)$ are uncorrelated.

The Kalman equations become scalar:

Prediction:

$$
\hat s(n|n-1)=a\hat s(n-1|n-1),
$$

$$
P(n|n-1)=a^2P(n-1|n-1)+q.
$$

Gain:

$$
K(n)=\frac{P(n|n-1)}{P(n|n-1)+r}.
$$

Update:

$$
\hat s(n|n)=\hat s(n|n-1)+K(n)[y(n)-\hat s(n|n-1)],
$$

$$
P(n|n)=[1-K(n)]P(n|n-1).
$$

This example makes the meaning of $K(n)$ clear:

- if measurement noise $r$ is large, $K(n)$ is small, so the filter trusts the model more;
- if prediction uncertainty $P(n|n-1)$ is large, $K(n)$ is large, so the filter trusts the new measurement more;
- if the system is time invariant, $K(n)$ often converges to a steady-state value.

When the gain converges, the Kalman filter becomes equivalent to a fixed recursive Wiener filter. Before convergence, it is time-varying.

## 6.8 Kalman Filter Examples from the Textbook

The textbook first demonstrates Kalman filtering on an autoregressive process. The observations are noisy, but the state model allows the filter to recover a smoother estimate.

> ![Figure 6.2](./CourseADSP2026/Fig/Chapter_6/fig_6_2_textbook_fig_7_12_p385.png)
>
> *Figure 6.2 (Textbook Fig. 7.12, p. 385): Estimation of an AR process using a Kalman filter. The filter tracks the underlying signal despite noisy observations.*

The corresponding Kalman gains and covariance converge toward steady values.

> ![Figure 6.3](./CourseADSP2026/Fig/Chapter_6/fig_6_3_textbook_fig_7_13_p385.png)
>
> *Figure 6.3 (Textbook Fig. 7.13, p. 385): Kalman gain values and estimation-error covariance. In a time-invariant model, the filter often approaches steady-state behavior.*

A second example considers position and velocity estimation. The state contains both position and velocity, while the measurement observes only noisy position. The filter can still infer velocity because the dynamic model couples position and velocity.

> ![Figure 6.4](./CourseADSP2026/Fig/Chapter_6/fig_6_4_textbook_fig_7_14_p386.png)
>
> *Figure 6.4 (Textbook Fig. 7.14, p. 386): Estimation of position and velocity using a Kalman filter. Even when velocity is not directly observed, it can be inferred through the state model.*

The Kalman gain and covariance trajectories show a transient period followed by approximate steady state.

> ![Figure 6.5](./CourseADSP2026/Fig/Chapter_6/fig_6_5_textbook_fig_7_15_p387.png)
>
> *Figure 6.5 (Textbook Fig. 7.15, p. 387): Kalman gains and estimation-error variances. The posterior covariance decreases after each observation and then grows again during prediction.*

## 6.9 Kalman Filter Interpretation

The Kalman filter can be understood through three complementary viewpoints.

### Viewpoint 1: Recursive Wiener Filter

At each time $n$, the Kalman filter performs a linear MMSE correction:

$$
\hat{\mathbf{s}}(n|n)=\hat{\mathbf{s}}(n|n-1)+\mathbf{K}(n)\boldsymbol{\nu}(n).
$$

The gain has the Wiener form:

$$
\mathbf{K}=\text{cross-covariance}\times \text{inverse covariance}.
$$

### Viewpoint 2: Model-Based Predictor plus Measurement Corrector

The prediction step trusts the dynamics:

$$
\text{state model} \quad \Longrightarrow \quad \hat{\mathbf{s}}(n|n-1).
$$

The update step trusts the measurement according to its reliability:

$$
\text{innovation} \quad \Longrightarrow \quad \text{correction}.
$$

### Viewpoint 3: Information Compression

All past measurements are summarized by

$$
\hat{\mathbf{s}}(n|n),\quad \mathbf{P}(n|n).
$$

Therefore the filter does not need to store the full observation history.

## 6.10 Practical Strengths and Limitations

| Strength | Explanation |
|----------|-------------|
| Recursive | Processes one observation at a time |
| Time-varying | Handles nonstationary state uncertainty |
| Model-based | Uses known dynamics to infer unobserved variables |
| Uncertainty-aware | Gain automatically balances model confidence and measurement confidence |
| Memory-efficient | Stores only state estimate and covariance |

However, Kalman filtering also depends on assumptions:

| Limitation | Consequence |
|------------|-------------|
| Wrong state model | Biased or unstable estimates |
| Wrong noise covariance | Gain becomes too aggressive or too conservative |
| Nonlinear dynamics | Standard linear Kalman filter is not directly applicable |
| Non-Gaussian noise | MMSE linear estimate may not be fully optimal in a nonlinear sense |

The standard Kalman filter is exactly optimal for linear Gaussian state-space models. Without Gaussianity, it is still the optimum linear MMSE recursive estimator under the specified covariance assumptions.

---

# §7 Chapter Summary, Method Selection, and Figure Checklist

## 7.1 Core Equations

### Linear MMSE Estimator

$$
\hat y=\mathbf{c}^H\mathbf{x}
$$

$$
\mathbf{R}\mathbf{c}_o=\mathbf{d}
$$

$$
P_o=P_y-\mathbf{d}^H\mathbf{c}_o
$$

$$
E\{\mathbf{x}e_o^\ast\}=\mathbf{0}
$$

### FIR Wiener Filter

$$
\hat y(n)=\mathbf{h}^H\mathbf{x}(n)
$$

$$
\mathbf{R}\mathbf{h}_o=\mathbf{d}
$$

### Noncausal IIR Wiener Filter

$$
H_{nc}(e^{j\omega})=\frac{R_{yx}(e^{j\omega})}{R_x(e^{j\omega})}
$$

### Causal IIR Wiener Filter

$$
H_c(z)=\frac{1}{\sigma_x^2 H_x(z)}
\left[\frac{R_{yx}(z)}{H_x^\ast(1/z^\ast)}\right]_+
$$

### Kalman Filter

Prediction:

$$
\hat{\mathbf{s}}(n|n-1)=\mathbf{A}(n-1)\hat{\mathbf{s}}(n-1|n-1)
$$

$$
\mathbf{P}(n|n-1)=\mathbf{A}\mathbf{P}(n-1|n-1)\mathbf{A}^H+\mathbf{B}\mathbf{Q}\mathbf{B}^H
$$

Gain:

$$
\mathbf{K}(n)=\mathbf{P}(n|n-1)\mathbf{C}^H(n)
[\mathbf{C}(n)\mathbf{P}(n|n-1)\mathbf{C}^H(n)+\mathbf{R}_v(n)]^{-1}
$$

Update:

$$
\hat{\mathbf{s}}(n|n)=\hat{\mathbf{s}}(n|n-1)+\mathbf{K}(n)[\mathbf{y}(n)-\mathbf{C}(n)\hat{\mathbf{s}}(n|n-1)]
$$

$$
\mathbf{P}(n|n)=[\mathbf{I}-\mathbf{K}(n)\mathbf{C}(n)]\mathbf{P}(n|n-1)
$$

## 7.2 Method Selection Guide

| Situation | Recommended Viewpoint |
|----------|-----------------------|
| Known stationary correlations, finite memory desired | FIR Wiener filter |
| Need theoretical best linear performance with future samples allowed | Noncausal IIR Wiener filter |
| Need real-time infinite-duration optimum filtering | Causal IIR Wiener filter with spectral factorization |
| Need future-value estimation | Linear prediction as Wiener filtering |
| Need undo channel distortion in noise | MMSE deconvolution / equalization |
| Known deterministic waveform in colored noise | Matched filter $\mathbf{R}_v^{-1}\mathbf{s}$ |
| Random signal and noise covariance known | Eigenfilter / generalized eigenfilter |
| Nonstationary state with dynamic model | Kalman filter |

## 7.3 Conceptual Links to Previous Lectures

| Previous Topic | Link to This Chapter |
|---------------|---------------------|
| Autocorrelation and PSD | Determine $\mathbf{R}$, $\mathbf{d}$, and Wiener spectra |
| Linear prediction | Special case of FIR/IIR Wiener filtering |
| AR modeling | Prediction-error filter whitens AR processes |
| Spectral factorization | Required for causal IIR Wiener filtering |
| Parametric spectrum estimation | Can provide models used to design Wiener filters |
| Subspace/eigen methods | Related to eigenfilters and quadratic-form optimization |

## 7.4 Teaching Flow for a 2–3 Lecture Delivery

A clear teaching sequence is:

1. Start with the scalar linear estimator and the MSE expansion.
2. Derive $\mathbf{R}\mathbf{c}=\mathbf{d}$.
3. Explain the orthogonality principle geometrically.
4. Convert the estimator into an FIR filter by using delayed samples.
5. Apply the same equation to filtering, prediction, noise cancellation, and equalization.
6. Explain why noncausal IIR Wiener filtering is easy in the frequency domain.
7. Explain why causal IIR Wiener filtering needs spectral factorization.
8. Introduce Kalman filtering as recursive time-varying MMSE estimation.
9. Emphasize prediction-correction and covariance propagation.

## 7.5 Figure Checklist

All figures displayed in this lecture are screenshots extracted from the textbook PDF and stored in:

```text
./CourseADSP2026/Fig/Chapter_6/
```

| Lecture Figure | Textbook Figure | File |
|----------------|-----------------|------|
| Figure 0.1 | Fig. 6.1, p. 262 | `fig_0_1_textbook_fig_6_1_p262.png` |
| Figure 0.2 | Fig. 6.2, p. 263 | `fig_0_2_textbook_fig_6_2_p263.png` |
| Figure 1.1 | Fig. 6.3, p. 265 | `fig_1_1_textbook_fig_6_3_p265.png` |
| Figure 1.2 | Fig. 6.4, p. 266 | `fig_1_2_textbook_fig_6_4_p266.png` |
| Figure 1.3 | Fig. 6.9, p. 273 | `fig_1_3_textbook_fig_6_9_p273.png` |
| Figure 2.1 | Fig. 6.10, p. 278 | `fig_2_1_textbook_fig_6_10_p278.png` |
| Figure 2.2 | Fig. 6.11, p. 280 | `fig_2_2_textbook_fig_6_11_p280.png` |
| Figure 2.3 | Fig. 6.12, p. 283 | `fig_2_3_textbook_fig_6_12_p283.png` |
| Figure 2.4 | Fig. 6.14, p. 284 | `fig_2_4_textbook_fig_6_14_p284.png` |
| Figure 3.1 | Fig. 6.16, p. 287 | `fig_3_1_textbook_fig_6_16_p287.png` |
| Figure 3.2 | Fig. 6.24, p. 307 | `fig_4_3_textbook_fig_6_24_p307.png` |
| Figure 3.3 | Fig. 6.27, p. 310 | `fig_4_4_textbook_fig_6_27_p310.png` |
| Figure 3.4 | Fig. 6.31, p. 314 | `fig_4_5_textbook_fig_6_31_p314.png` |
| Figure 3.5 | Fig. 6.32, p. 315 | `fig_4_6_textbook_fig_6_32_p315.png` |
| Figure 3.6 | Fig. 6.33, p. 317 | `fig_4_7_textbook_fig_6_33_p317.png` |
| Figure 4.1 | Fig. 6.18, p. 298 | `fig_4_1_textbook_fig_6_18_p298.png` |
| Figure 4.2 | Fig. 6.19, p. 299 | `fig_4_2_textbook_fig_6_19_p299.png` |
| Figure 5.1 | Fig. 6.35, p. 321 | `fig_5_1_textbook_fig_6_35_p321.png` |
| Figure 5.2 | Fig. 6.36, p. 323 | `fig_5_2_textbook_fig_6_36_p323.png` |
| Figure 5.3 | Fig. 6.37, p. 324 | `fig_5_3_textbook_fig_6_37_p324.png` |
| Figure 6.1 | Fig. 7.11, p. 384 | `fig_6_1_textbook_fig_7_11_p384.png` |
| Figure 6.2 | Fig. 7.12, p. 385 | `fig_6_2_textbook_fig_7_12_p385.png` |
| Figure 6.3 | Fig. 7.13, p. 385 | `fig_6_3_textbook_fig_7_13_p385.png` |
| Figure 6.4 | Fig. 7.14, p. 386 | `fig_6_4_textbook_fig_7_14_p386.png` |
| Figure 6.5 | Fig. 7.15, p. 387 | `fig_6_5_textbook_fig_7_15_p387.png` |

---

## End-of-Chapter Takeaway

The entire chapter can be compressed into one sentence:

> **Wiener filtering finds the best linear estimate from known second-order statistics, while Kalman filtering performs the same MMSE idea recursively when a dynamic state model is available.**

The practical question is never only “what is the formula?” The practical question is:

1. What is the desired response?
2. What data are available to the estimator?
3. What statistics or model are known?
4. Is the filter required to be causal?
5. Is the process stationary or time-varying?
6. How much complexity and delay are acceptable?

Once these questions are answered, the correct filtering framework is usually clear.
