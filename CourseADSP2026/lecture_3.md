# Modern Digital Signal Processing
## Chapter 3: Linear Prediction and Lattice Filters

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005  
> Chapters covered: §6.5 (Linear Prediction) · Ch. 7 (Algorithms and Structures for Optimum Linear Filters, §7.1–§7.7) · §9.2.3 (Maximum Entropy Method / Burg Method) · §9.4.2 (Speech Modeling)

---

## Table of Contents

- [Notation and Variable Definitions](#notation-and-variable-definitions)
  - [Time Index, Data Length, and Model Order](#time-index-data-length-and-model-order)
  - [Signals, Predictors, and Errors](#signals-predictors-and-errors)
  - [Correlation Quantities](#correlation-quantities)
  - [Recursive and Lattice Parameters](#recursive-and-lattice-parameters)
  - [Model Types](#model-types)
- [§1 Basic Linear Prediction Model and the Autocorrelation Method](#1-basic-linear-prediction-model-and-the-autocorrelation-method)
  - [1.1 Why Linear Prediction Is Important](#11-why-linear-prediction-is-important)
  - [1.2 Linear Prediction as LMMSE Estimation](#12-linear-prediction-as-lmmse-estimation)
  - [1.3 Minimum Prediction Error Power](#13-minimum-prediction-error-power)
  - [1.4 The Autocorrelation Method for Finite Data](#14-the-autocorrelation-method-for-finite-data)
    - [Important Practical Consequence: Stability](#important-practical-consequence-stability)
    - [Weakness of the Autocorrelation Method](#weakness-of-the-autocorrelation-method)
  - [1.5 Windowing and Model Accuracy](#15-windowing-and-model-accuracy)
- [§2 Equivalence Between AR All-Pole Modeling and Linear Prediction](#2-equivalence-between-ar-all-pole-modeling-and-linear-prediction)
  - [2.1 The AR($p$) Model](#21-the-arp-model)
  - [2.2 Prediction Error Filter Equals Whitening Filter](#22-prediction-error-filter-equals-whitening-filter)
  - [2.3 Yule-Walker Equations from the AR Model](#23-yule-walker-equations-from-the-ar-model)
  - [2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction](#24-statistical-all-pole-modeling-vs-deterministic-linear-prediction)
    - [Statistical AR Modeling](#statistical-ar-modeling)
    - [Deterministic Linear Prediction](#deterministic-linear-prediction)
  - [2.5 Why All-Pole Models Are Useful](#25-why-all-pole-models-are-useful)
- [§3 Levinson-Durbin Recursive Algorithm](#3-levinson-durbin-recursive-algorithm)
  - [3.1 The Computational Problem](#31-the-computational-problem)
  - [3.2 The Order-Recursive View](#32-the-order-recursive-view)
  - [3.3 Levinson-Durbin Recursion Formulas](#33-levinson-durbin-recursion-formulas)
  - [3.4 Why the Error Recursion Makes Sense](#34-why-the-error-recursion-makes-sense)
  - [3.5 Reflection Coefficient as Partial Correlation](#35-reflection-coefficient-as-partial-correlation)
  - [3.6 Direct Form vs Recursive Structure](#36-direct-form-vs-recursive-structure)
  - [3.7 Algorithm Table](#37-algorithm-table)
  - [3.8 Simple First-Order Example](#38-simple-first-order-example)
- [§4 Three Equivalent Sets of Recursive Parameters](#4-three-equivalent-sets-of-recursive-parameters)
  - [4.1 Three Equivalent Descriptions](#41-three-equivalent-descriptions)
  - [4.2 Positive Definiteness, Reflection Coefficients, and Stability](#42-positive-definiteness-reflection-coefficients-and-stability)
  - [4.3 Boundary Case: Zero on the Unit Circle](#43-boundary-case-zero-on-the-unit-circle)
  - [4.4 Cholesky / LDL$^H$ Decomposition View](#44-cholesky--ldlh-decomposition-view)
  - [4.5 Autocorrelation Extension and Maximum Entropy](#45-autocorrelation-extension-and-maximum-entropy)
  - [4.6 Step-Up Recursion: Lattice to Direct Form](#46-step-up-recursion-lattice-to-direct-form)
  - [4.7 Step-Down Recursion: Direct Form to Lattice](#47-step-down-recursion-direct-form-to-lattice)
  - [4.8 Why Reflection Coefficients Are Numerically Attractive](#48-why-reflection-coefficients-are-numerically-attractive)
- [§5 Schur Recursive Algorithm](#5-schur-recursive-algorithm)
  - [5.1 Goal of the Schur Algorithm](#51-goal-of-the-schur-algorithm)
  - [5.2 Gapped Correlation Functions](#52-gapped-correlation-functions)
  - [5.3 Algorithmic Interpretation](#53-algorithmic-interpretation)
  - [5.4 Schur Algorithm Summary](#54-schur-algorithm-summary)
  - [5.5 Relationship to Levinson-Durbin](#55-relationship-to-levinson-durbin)
- [§6 General Levinson Recursion for Toeplitz Equations](#6-general-levinson-recursion-for-toeplitz-equations)
  - [6.1 Beyond Linear Prediction](#61-beyond-linear-prediction)
  - [6.2 The Optimum Nesting Property](#62-the-optimum-nesting-property)
  - [6.3 Partitioned Matrix Inversion View](#63-partitioned-matrix-inversion-view)
  - [6.4 General Levinson Recursion Idea](#64-general-levinson-recursion-idea)
  - [6.5 Complexity](#65-complexity)
- [§7 Covariance Algorithm for Linear Prediction](#7-covariance-algorithm-for-linear-prediction)
  - [7.1 Motivation](#71-motivation)
  - [7.2 Error Criterion](#72-error-criterion)
  - [7.3 Why the Matrix Is Not Toeplitz](#73-why-the-matrix-is-not-toeplitz)
  - [7.4 Advantages and Disadvantages](#74-advantages-and-disadvantages)
  - [7.5 When to Use the Covariance Method](#75-when-to-use-the-covariance-method)
- [§8 Forward/Backward Prediction and Lattice Filters](#8-forwardbackward-prediction-and-lattice-filters)
  - [8.1 Forward and Backward Prediction Errors](#81-forward-and-backward-prediction-errors)
  - [8.2 Lattice Recursions](#82-lattice-recursions)
  - [8.3 Intuition Behind the Lattice Stage](#83-intuition-behind-the-lattice-stage)
  - [8.4 All-Pole Lattice Synthesis](#84-all-pole-lattice-synthesis)
  - [8.5 Advantages of Lattice Filters](#85-advantages-of-lattice-filters)
  - [8.6 Lattice-Ladder Structure](#86-lattice-ladder-structure)
- [§9 Lattice Modeling and the Burg Algorithm](#9-lattice-modeling-and-the-burg-algorithm)
  - [9.1 Why Burg's Algorithm Was Introduced](#91-why-burgs-algorithm-was-introduced)
  - [9.2 Forward and Backward Error Criterion](#92-forward-and-backward-error-criterion)
  - [9.3 Burg Reflection Coefficient Update](#93-burg-reflection-coefficient-update)
  - [9.4 Burg Algorithm Steps](#94-burg-algorithm-steps)
  - [9.5 Burg, Forward Covariance, and Backward Covariance](#95-burg-forward-covariance-and-backward-covariance)
    - [Forward Lattice Covariance Method](#forward-lattice-covariance-method)
    - [Backward Lattice Covariance Method](#backward-lattice-covariance-method)
    - [Burg Method](#burg-method)
  - [9.6 Itakura-Saito / Geometric Mean Variant](#96-itakura-saito--geometric-mean-variant)
  - [9.7 Maximum Entropy Interpretation](#97-maximum-entropy-interpretation)
  - [9.8 Strengths and Weaknesses of Burg's Algorithm](#98-strengths-and-weaknesses-of-burgs-algorithm)
- [§10 Modified Covariance Algorithm](#10-modified-covariance-algorithm)
  - [10.1 Motivation](#101-motivation)
  - [10.2 Difference Between Burg and Modified Covariance](#102-difference-between-burg-and-modified-covariance)
  - [10.3 Modified Covariance Normal Equations](#103-modified-covariance-normal-equations)
  - [10.4 Advantages](#104-advantages)
  - [10.5 Disadvantages](#105-disadvantages)
  - [10.6 Comparison of Main Linear Prediction Estimation Methods](#106-comparison-of-main-linear-prediction-estimation-methods)
- [§11 Application Example: Linear Prediction in Speech Coding](#11-application-example-linear-prediction-in-speech-coding)
  - [11.1 Why Speech Is Predictable](#111-why-speech-is-predictable)
  - [11.2 LPC Model](#112-lpc-model)
  - [11.3 Three Coding Paradigms](#113-three-coding-paradigms)
    - [Waveform Coding](#waveform-coding)
    - [Parametric Coding](#parametric-coding)
    - [Hybrid Coding](#hybrid-coding)
  - [11.4 Why Reflection Coefficients Are Useful in Speech](#114-why-reflection-coefficients-are-useful-in-speech)
  - [11.5 Speech Spectrum and All-Pole Envelope](#115-speech-spectrum-and-all-pole-envelope)
  - [Chapter 3 Summary](#chapter-3-summary)
  - [Figure Source Checklist](#figure-source-checklist)
  - [Suggested Teaching Flow](#suggested-teaching-flow)

---

## Notation and Variable Definitions

All symbols used in this chapter are collected below. The notation follows the conventions of Chapters 1 and 2: boldface lower-case letters denote vectors, boldface upper-case letters denote matrices, and superscript $H$ denotes Hermitian transpose.

### Time Index, Data Length, and Model Order

| Symbol | Definition |
|--------|------------|
| $n$ | Discrete-time sample index |
| $N$ | Last sample index or data length parameter, depending on context |
| $p$ | Linear prediction / AR model order |
| $m$ | Intermediate recursive order, usually $m=0,1,\ldots,p$ |
| $k$ | Lag index or coefficient index |
| $l$ | Correlation lag index |
| $N_i, N_f$ | Initial and final indices in finite-data least-squares criteria |

### Signals, Predictors, and Errors

| Symbol | Definition |
|--------|------------|
| $x(n)$ | Discrete-time signal or WSS random process to be predicted/modelled |
| $\hat{x}(n)$ | Linear prediction estimate of $x(n)$ |
| $w(n)$ | White-noise excitation / innovation sequence |
| $e_p^f(n)$ | $p$-th order forward prediction error |
| $e_p^b(n)$ | $p$-th order backward prediction error |
| $\mathbf{x}_p(n)$ | Data vector, usually $[x(n-1),x(n-2),\ldots,x(n-p)]^T$ for forward prediction |
| $\mathbf{a}_p$ | Direct-form forward prediction coefficient vector $[a_1^{(p)},\ldots,a_p^{(p)}]^T$ |
| $\mathbf{b}_p$ | Direct-form backward prediction coefficient vector $[b_0^{(p)},\ldots,b_{p-1}^{(p)}]^T$ |
| $A_p(z)$ | Forward prediction error filter (PEF) |
| $B_p(z)$ | Backward prediction error filter (BPEF) |

### Correlation Quantities

| Symbol | Definition |
|--------|------------|
| $r_x(l)=E\left[x(n)x^{\ast}(n-l)\right]$ | Autocorrelation sequence of a WSS process |
| $\hat r_x(l)$ | Estimated autocorrelation sequence from finite data |
| $\mathbf{R}_p$ | $p\times p$ autocorrelation matrix, Hermitian Toeplitz in the WSS case |
| $\mathbf{r}_p=[r_x(1),\ldots,r_x(p)]^T$ | Correlation vector used in the Yule-Walker / prediction equations |
| $P_p$ | Minimum prediction error power at order $p$ |
| $\sigma_p^2$ | Another common notation for $P_p$; in AR modeling it is the residual variance |
| $\Phi(i,k)$ | Sample covariance term used in covariance and modified covariance methods |

### Recursive and Lattice Parameters

| Symbol | Definition |
|--------|------------|
| $\kappa_m$ or $k_m$ | Reflection coefficient / lattice coefficient / PARCOR coefficient at stage $m$ |
| $\alpha_m$ | Prediction error update numerator in some Levinson-Durbin derivations |
| $\mathbf{J}$ | Exchange matrix that reverses vector order |
| $\xi_m^f(l)$ | Forward gapped correlation function in the Schur algorithm |
| $\xi_m^b(l)$ | Backward gapped correlation function in the Schur algorithm |
| $k_m^c$ | Ladder coefficient in a lattice-ladder optimum FIR filter |

### Model Types

| Symbol | Definition |
|--------|------------|
| AR($p$) | Autoregressive model of order $p$, also called an all-pole model |
| AP($p$) | All-pole model of order $p$ |
| PEF | Prediction error filter; for an AR process it is the whitening filter |
| LPC | Linear predictive coding; speech coding method based on short-time all-pole modeling |

---

# §1 Basic Linear Prediction Model and the Autocorrelation Method

---

### Guiding Question

A seabed hydrophone records slowly varying pump noise. After mean removal, its measured autocorrelations are $r(0)=10$, $r(1)=6$, and $r(2)=3$. An embedded detector wants to predict the current sample from the preceding two samples so that unexpected target transients appear strongly in the residual. What are the two optimum prediction weights, what is the minimum residual power, and what percentage of the original power has prediction removed?

The random-signal tools from Lecture 2 describe these correlations, but they do not yet turn them into a finite-order predictor. Section 1 develops that LMMSE construction.

---

> 📖 Textbook §6.5 (Linear Prediction); §7.1 (Fundamentals of Order-Recursive Algorithms)

---

**§1 Basic Linear Prediction Model and the Autocorrelation Method**

- [1.1 Why Linear Prediction Is Important](#11-why-linear-prediction-is-important)
- <a href="#12-linear-prediction-as-lmmse-estimation" style="color: #a0a0a0;">1.2 Linear Prediction as LMMSE Estimation</a>
- <a href="#13-minimum-prediction-error-power" style="color: #a0a0a0;">1.3 Minimum Prediction Error Power</a>
- <a href="#14-the-autocorrelation-method-for-finite-data" style="color: #a0a0a0;">1.4 The Autocorrelation Method for Finite Data</a>
- <a href="#15-windowing-and-model-accuracy" style="color: #a0a0a0;">1.5 Windowing and Model Accuracy</a>

---

## 1.1 Why Linear Prediction Is Important

Linear prediction is one of the central ideas in modern digital signal processing. The basic question is simple:

> Can we estimate the current sample of a signal from a linear combination of its past samples?

Concrete application scenarios with numerical signals:

- **Speech coding:** Here $x(n)$ is one discrete-time speech sample after the microphone signal has been sampled and normalized. Physically, it is proportional to the short-time acoustic pressure at the microphone, or equivalently to the microphone voltage after A/D conversion. If the sampling rate is 8 kHz, then adjacent samples are only $1/8000=0.125$ ms apart, so voiced speech usually changes smoothly from one sample to the next.

  Suppose four consecutive normalized speech samples are
  $$x(n-3)=0.62,\quad x(n-2)=0.71,\quad x(n-1)=0.76,\quad x(n)=0.80.$$
  These numbers mean that the waveform amplitude is rising slowly during this short interval. A simple third-order predictor might use
  $$\hat{x}(n)=0.65x(n-1)+0.25x(n-2)+0.08x(n-3).$$
  Then
  $$\hat{x}(n)=0.65(0.76)+0.25(0.71)+0.08(0.62)=0.721,$$
  so the prediction error is
  $$e(n)=x(n)-\hat{x}(n)=0.80-0.721=0.079.$$
  The sample $x(n)=0.80$ itself is large because it contains both the predictable waveform shape and the new information. The residual $e(n)=0.079$ is smaller because the predictable part, $0.721$, has already been explained by past samples.

  This helps coding because the residual usually has smaller variance and a smaller amplitude range. For example, if raw normalized speech samples occupy roughly $[-1,1]$, but the prediction residual usually lies in $[-0.125,0.125]$, then the residual range is eight times narrower. With uniform quantization, using 5 bits for $[-0.125,0.125]$ gives step size
  $$\Delta_e=\frac{0.25}{2^5}=0.0078125,$$
  which is the same step size as using 8 bits for raw samples over $[-1,1]$:
  $$\Delta_x=\frac{2}{2^8}=0.0078125.$$
  So, in this simplified example, coding the residual can use about 5 bits per sample instead of 8 bits per sample for similar scalar quantization precision. The predictor coefficients are transmitted or updated much less frequently, usually once per short frame, not once per sample.

For a highly correlated signal, such as speech, radar clutter, narrowband interference, or a slowly varying sensor measurement, the answer is often yes. The current sample contains information that is already partly present in previous samples. Linear prediction extracts this redundancy.

A one-step $p$-th order forward predictor estimates $x(n)$ using the previous $p$ samples:

$$\hat{x}(n)=-\sum_{k=1}^{p} a_k^{\ast}x(n-k).$$

The corresponding prediction error is the residual left after prediction. We denote the $p$-th order forward prediction error by $e_p^f(n)$:

$$\boxed{e_p^f(n)=x(n)-\hat{x}(n)=x(n)+\sum_{k=1}^{p}a_k^{\ast}x(n-k)}$$

The prediction error filter (PEF) is the FIR filter that maps the original signal $x(n)$ to this residual error $e_p^f(n)$. We denote the z-domain transfer function of this order-$p$ error filter by $A_p(z)$. Equivalently, $A_p(z)$ is the polynomial whose impulse-response coefficients are $[1,a_1^\ast,\ldots,a_p^\ast]$, so filtering $x(n)$ by $A_p(z)$ produces the residual:

$$e_p^f(n)=A_p(z)x(n).$$

The minus sign in the predictor is a convention chosen so that this filter has the compact polynomial form

$$\boxed{A_p(z)=1+\sum_{k=1}^{p}a_k^{\ast}z^{-k}}.$$

This means that linear prediction can be viewed in two equivalent ways:

| Viewpoint | Meaning |
|----------|---------|
| Predictor | Use the past to estimate the current sample |
| Error filter | Filter the signal by $A_p(z)$ so that predictable components are removed |

The second viewpoint is especially important. If $A_p(z)$ removes all correlation that can be linearly predicted from the past, then its output is the innovation or residual part of the signal.

> ![Figure 1.1](./CourseADSP2026/Fig/Chapter_3/fig_1_1_textbook_fig_6_16_p283.png)
>
> *Figure 1.1 (Textbook Fig. 6.16, p. 283): Linear signal estimation, forward linear prediction, and backward linear prediction. This figure is the most useful starting point for this chapter because it shows that prediction is a special case of linear signal estimation.*

---

**§1 Basic Linear Prediction Model and the Autocorrelation Method**

- <a href="#11-why-linear-prediction-is-important" style="color: #a0a0a0;">1.1 Why Linear Prediction Is Important</a>
- [1.2 Linear Prediction as LMMSE Estimation](#12-linear-prediction-as-lmmse-estimation)
- <a href="#13-minimum-prediction-error-power" style="color: #a0a0a0;">1.3 Minimum Prediction Error Power</a>
- <a href="#14-the-autocorrelation-method-for-finite-data" style="color: #a0a0a0;">1.4 The Autocorrelation Method for Finite Data</a>
- <a href="#15-windowing-and-model-accuracy" style="color: #a0a0a0;">1.5 Windowing and Model Accuracy</a>

---

## 1.2 Linear Prediction as LMMSE Estimation

Chapter 2 introduced the linear minimum mean-square-error (LMMSE) principle. For linear prediction, the desired response is $x(n)$ and the data vector is

$$\mathbf{x}_p(n)=\begin{bmatrix}x(n-1)&x(n-2)&\cdots&x(n-p)\end{bmatrix}^T.$$

The prediction error is

$$e_p^f(n)=x(n)+\mathbf{a}_p^H\mathbf{x}_p(n).$$

The optimal coefficient vector is the one that minimizes

$$J(\mathbf{a}_p)=E\{\lvert e_p^f(n)\rvert^2\}.$$

Before writing the normal equations, it is useful to understand the geometry of this minimization. The previous samples

$$x(n-1),x(n-2),\ldots,x(n-p)$$

span the linear prediction space. The predictor chooses the point in this space that is closest to the desired sample $x(n)$ in the mean-square sense. The prediction error is the residual part left after this projection:

$$e_p^f(n)=x(n)-\hat{x}(n).$$

If the residual were still correlated with one of the samples used by the predictor, then that sample would still contain useful linear information about the residual. In that case, we could adjust the corresponding prediction coefficient and reduce the mean-square error further. Therefore, at the optimum, no such useful linear information can remain in the residual.

Equivalently, the predictor has already extracted all linearly predictable information from the previous $p$ samples. The leftover error must be perpendicular, in the correlation/inner-product sense, to each sample used in the prediction.

The LMMSE orthogonality principle says:

> The optimal prediction error is orthogonal to every sample used by the predictor.

Thus, for $i=1,2,\ldots,p$,

$$E\{e_p^f(n)x^{\ast}(n-i)\}=0.$$

Substituting the prediction error gives

$$E\left\{\left[x(n)+\sum_{k=1}^{p}a_k^{\ast}x(n-k)\right]x^{\ast}(n-i)\right\}=0.$$

For a WSS process, this becomes

$$r_x(i)+\sum_{k=1}^{p}a_k^{\ast}r_x(i-k)=0, \qquad i=1,2,\ldots,p.$$

In matrix form,

$$\boxed{\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p^{\ast}}$$

or, depending on whether the coefficient vector is conjugated in the chosen convention,

$$\boxed{\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p}.$$

The exact placement of complex conjugates is a notation issue. For real-valued signals, both forms reduce to the familiar Yule-Walker / Wiener-Hopf equations:

$$\begin{bmatrix}
r(0)&r(1)&\cdots&r(p-1)\\
r(1)&r(0)&\cdots&r(p-2)\\
\vdots&\vdots&\ddots&\vdots\\
r(p-1)&r(p-2)&\cdots&r(0)
\end{bmatrix}
\begin{bmatrix}a_1\\a_2\\\vdots\\a_p\end{bmatrix}
=-\begin{bmatrix}r(1)\\r(2)\\\vdots\\r(p)\end{bmatrix}.$$

The matrix is Toeplitz: each diagonal is constant. This Toeplitz structure is the reason why Levinson-Durbin recursion can solve the system much faster than generic Gaussian elimination.

---

**§1 Basic Linear Prediction Model and the Autocorrelation Method**

- <a href="#11-why-linear-prediction-is-important" style="color: #a0a0a0;">1.1 Why Linear Prediction Is Important</a>
- <a href="#12-linear-prediction-as-lmmse-estimation" style="color: #a0a0a0;">1.2 Linear Prediction as LMMSE Estimation</a>
- [1.3 Minimum Prediction Error Power](#13-minimum-prediction-error-power)
- <a href="#14-the-autocorrelation-method-for-finite-data" style="color: #a0a0a0;">1.4 The Autocorrelation Method for Finite Data</a>
- <a href="#15-windowing-and-model-accuracy" style="color: #a0a0a0;">1.5 Windowing and Model Accuracy</a>

---

## 1.3 Minimum Prediction Error Power

After the optimal coefficients are found, the minimum prediction error power is

$$\boxed{P_p=E\{\lvert e_p^f(n)\rvert^2\}=r_x(0)+\sum_{k=1}^{p}a_k^{\ast}r_x(k)}.$$

In vector form,

$$\boxed{P_p=r_x(0)+\mathbf{r}_p^H\mathbf{a}_p}.$$

This expression has a clear interpretation:

- $r_x(0)$ is the original signal power.
- $\mathbf{r}_p^H\mathbf{a}_p$ is the amount of predictable power removed by the predictor.
- $P_p$ is the leftover power that cannot be predicted linearly from the previous $p$ samples.

Strictly speaking, the second bullet above is a sign-sensitive statement. The actual predictable power removed by the optimal predictor is

$$-\mathbf{r}_p^H\mathbf{a}_p \ge 0.$$

Indeed, using the Wiener-Hopf equation

$$\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p,$$

we have

$$\mathbf{a}_p=-\mathbf{R}_p^{-1}\mathbf{r}_p,$$

and therefore

$$\mathbf{r}_p^H\mathbf{a}_p
=-\mathbf{r}_p^H\mathbf{R}_p^{-1}\mathbf{r}_p \le 0,$$

because the autocorrelation matrix $\mathbf{R}_p$ is positive definite, or at least positive semidefinite in the non-invertible case. Thus the prediction error power can be read as

$$P_p=r_x(0)-\left(-\mathbf{r}_p^H\mathbf{a}_p\right).$$

This does not mean that every coefficient $a_k^{\ast}$ is negative. The sign comes from the whole quadratic form. For example, in the real-valued first-order case,

$$a_1=-\frac{r(1)}{r(0)},$$

so

$$a_1r(1)=-\frac{r(1)^2}{r(0)}\le 0.$$

If $r(1)>0$, then $a_1<0$; if $r(1)<0$, then $a_1>0$. In both cases the product is nonpositive. What matters is not the sign of each coefficient, but the fact that the optimal predictor chooses coefficients that cancel the predictable correlation.

If the signal is very predictable, $P_p$ is small. If the signal is white noise, all nonzero-lag autocorrelations are zero.

To see this, for zero-mean white noise with variance $\sigma_x^2$,

$$E\left[x(n)x^{\ast}(m)\right]=\sigma_x^2\delta[n-m].$$

Therefore

$$r_x(l)=E\left[x(n)x^{\ast}(n-l)\right]
=\sigma_x^2\delta[l].$$

Hence

$$r_x(l)=0,\qquad l\ne 0.$$

Thus the autocorrelation vector used in the Wiener-Hopf equation is

$$\mathbf{r}_p=
\begin{bmatrix}
r_x(1)\\
r_x(2)\\
\vdots\\
r_x(p)
\end{bmatrix}
=\mathbf{0}.$$

Since

$$\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p,$$

we get $\mathbf{a}_p=\mathbf{0}$ when $\mathbf{R}_p$ is invertible. Consequently,

$$P_p=r_x(0).$$

In that case prediction cannot improve anything.

---

**§1 Basic Linear Prediction Model and the Autocorrelation Method**

- <a href="#11-why-linear-prediction-is-important" style="color: #a0a0a0;">1.1 Why Linear Prediction Is Important</a>
- <a href="#12-linear-prediction-as-lmmse-estimation" style="color: #a0a0a0;">1.2 Linear Prediction as LMMSE Estimation</a>
- <a href="#13-minimum-prediction-error-power" style="color: #a0a0a0;">1.3 Minimum Prediction Error Power</a>
- [1.4 The Autocorrelation Method for Finite Data](#14-the-autocorrelation-method-for-finite-data)
- <a href="#15-windowing-and-model-accuracy" style="color: #a0a0a0;">1.5 Windowing and Model Accuracy</a>

---

## 1.4 The Autocorrelation Method for Finite Data

In theory, the autocorrelation sequence is an ensemble average:

$$r_x(l)=E\left[x(n)x^{\ast}(n-l)\right].$$

In practice, we usually have only one finite data record:

$$x(0),x(1),\ldots,x(N).$$

The autocorrelation method estimates $r_x(l)$ by treating samples outside the observed interval as zero. A common estimate is

$$\boxed{\hat r_x(l)=\sum_{n=l}^{N}x(n)x^{\ast}(n-l),\qquad l=0,1,\ldots,p.}$$

Equivalently, this is the autocorrelation of a finite sequence that has been multiplied by a rectangular window.

The resulting normal equations are still Toeplitz:

Why is the matrix still Toeplitz? The key point is that, after zero extension, the finite data record can be viewed as a sequence defined for all integer time indices, with samples outside the observed interval set to zero:

$$x(n)=0,\qquad n<0\ \text{or}\ n>N.$$

For a $p$th-order predictor, the prediction error is

$$e(n)=x(n)+\sum_{k=1}^{p}a_kx(n-k).$$

The autocorrelation method minimizes the total squared error

$$J=\sum_n |e(n)|^2,$$

where the summation may be taken over all $n$ because samples outside the observed interval are zero. The normal equations come from the orthogonality conditions

$$\sum_n e(n)x^{\ast}(n-i)=0,\qquad i=1,\ldots,p.$$

Substituting $e(n)$ gives

$$
\sum_n x(n)x^{\ast}(n-i)
+\sum_{k=1}^{p}a_k\sum_n x(n-k)x^{\ast}(n-i)=0.
$$

The coefficient matrix is determined by the inner sums

$$\sum_n x(n-k)x^{\ast}(n-i).$$

Let $m=n-k$. Then

$$
\sum_n x(n-k)x^{\ast}(n-i)
=\sum_m x(m)x^{\ast}(m+k-i).
$$

This quantity depends only on the difference $i-k$, not on $i$ and $k$ separately. Therefore the $(i,k)$ entry of the normal-equation matrix has the form

$$[\hat{\mathbf{R}}_p]_{i,k}=\hat r_x(i-k),\qquad i,k=1,\ldots,p,$$

up to the usual conjugation convention for negative lags. Since all entries on the same diagonal have the same value of $i-k$, every diagonal is constant. Hence $\hat{\mathbf{R}}_p$ is Hermitian Toeplitz.

The hat notation here emphasizes that these are finite-sample quantities, not ensemble autocorrelations. The unknown vector

$$\hat{\mathbf{a}}_p=[\hat a_1,\hat a_2,\ldots,\hat a_p]^T$$

contains the prediction coefficients obtained by minimizing the sample squared error $J$. The right-hand side comes from the first term in the scalar normal equations:

$$
\sum_n x(n)x^{\ast}(n-i)=\hat r_x(i),\qquad i=1,\ldots,p.
$$

Thus

$$\hat{\mathbf{r}}_p=[\hat r_x(1),\hat r_x(2),\ldots,\hat r_x(p)]^T,$$

with the same conjugation convention used in $\hat{\mathbf{R}}_p$. The $i$-th scalar equation

$$
\hat r_x(i)+\sum_{k=1}^{p}\hat a_k\hat r_x(i-k)=0
$$

is therefore exactly the $i$-th row of the matrix equation

$$\hat{\mathbf{R}}_p\hat{\mathbf{a}}_p=-\hat{\mathbf{r}}_p.$$

This is the main advantage of the autocorrelation method: it preserves the Toeplitz structure, so the coefficients can be computed efficiently and stably by Levinson-Durbin recursion.

### Important Practical Consequence: Stability

For a positive definite autocorrelation matrix, the autocorrelation method produces a prediction error filter $A_p(z)$ that is minimum phase. This means all zeros of $A_p(z)$ are inside the unit circle.

The relationship between the autocorrelation matrix and $A_p(z)$ is indirect but very important. The autocorrelation method first forms the Toeplitz matrix

$$
\hat{\mathbf{R}}_p=
\begin{bmatrix}
\hat r_x(0)&\hat r_x(1)&\cdots&\hat r_x(p-1)\\
\hat r_x(1)&\hat r_x(0)&\cdots&\hat r_x(p-2)\\
\vdots&\vdots&\ddots&\vdots\\
\hat r_x(p-1)&\hat r_x(p-2)&\cdots&\hat r_x(0)
\end{bmatrix}
$$

from the estimated autocorrelation values. Then it solves the normal equations

$$\hat{\mathbf{R}}_p\hat{\mathbf{a}}_p=-\hat{\mathbf{r}}_p,$$

where

$$\hat{\mathbf{a}}_p=[\hat a_1,\hat a_2,\ldots,\hat a_p]^T.$$

These solved coefficients are exactly the coefficients used to build the prediction error filter:

$$A_p(z)=1+\sum_{k=1}^{p}\hat a_k z^{-k}.$$

So the flow is:

$$
\text{autocorrelation values}
\rightarrow \hat{\mathbf{R}}_p,\hat{\mathbf{r}}_p
\rightarrow \hat a_1,\ldots,\hat a_p
\rightarrow A_p(z).
$$

Positive definiteness is a property of $\hat{\mathbf{R}}_p$, but it controls the kind of filter that results after solving for the coefficients. The reason is not merely that the normal equations have a unique solution; the stronger reason is that positive definiteness keeps every intermediate prediction problem strictly nondegenerate.

Here is the logic, with the symbols unpacked first.

The Levinson-Durbin recursion is an efficient algorithm for solving the Toeplitz autocorrelation normal equations. Instead of solving only the final $p$-th order system, it builds the predictor one order at a time:

$$0\rightarrow 1\rightarrow 2\rightarrow \cdots \rightarrow p.$$

Here $A_p(z)$ is the $p$-th order prediction error filter. It is the filter that converts the original signal into the prediction residual:

$$e_p^f(n)=A_p(z)x(n).$$

For a $p$-th order predictor, this filter has the polynomial form

$$A_p(z)=1+\sum_{k=1}^{p}a_k^{(p)}z^{-k}.$$

The leading coefficient is $1$ because the prediction error always contains the current sample $x(n)$ itself:

$$e_p^f(n)=x(n)+\sum_{k=1}^{p}a_k^{(p)}x(n-k).$$

When $p=0$, no past samples are used. There is no predictor, so the prediction error is simply the original sample:

$$e_0^f(n)=x(n).$$

Therefore the filter that maps $x(n)$ to $e_0^f(n)$ is just the identity filter:

$$A_0(z)=1.$$

At intermediate order $m$, it has already found the best $m$-th order prediction error filter

$$A_m(z)=1+\sum_{k=1}^{m}a_k^{(m)}z^{-k}.$$

Its output is the $m$-th order prediction error

$$e_m^f(n)=A_m(z)x(n).$$

The corresponding minimum prediction error power is

$$P_m=E\{\lvert e_m^f(n)\rvert^2\}.$$

This is the same quantity introduced earlier as $P_p$, but with the final order $p$ replaced by an arbitrary intermediate order $m$. In other words, $P_p$ is the final error power after the full $p$-th order predictor is built, while

$$P_0,P_1,\ldots,P_p$$

are the error powers along the way.

The notation means:

| Symbol | Meaning |
|--------|---------|
| $m$ | current prediction order |
| $P_m$ | minimum prediction error power using an $m$-th order predictor |
| $P_{m-1}$ | minimum prediction error power from the previous order |
| $\kappa_m$ | reflection coefficient introduced when going from order $m-1$ to order $m$ |

For example, $P_0$ is the error power when no past sample is used. In the ensemble WSS case, $P_0=r_x(0)$. In the finite-data autocorrelation method, it is the zero-order residual energy of the windowed data. After adding one more prediction lag, the error power becomes $P_1$, then $P_2$, and so on.

The reflection coefficient $\kappa_m$ measures how much predictable correlation remains after the lower-order predictor has already removed the effects of lags $1,\ldots,m-1$. It is also called a lattice coefficient or PARCOR coefficient. If $\lvert\kappa_m\rvert$ is small, the new lag contributes little. If $\lvert\kappa_m\rvert$ is large, the new lag removes a large fraction of the remaining error power.

The Levinson-Durbin energy update is

$$P_m=P_{m-1}(1-\lvert\kappa_m\rvert^2),$$

which says:

> the order-$m$ prediction error power equals the order-$(m-1)$ error power times the remaining fraction $1-\lvert\kappa_m\rvert^2$.

So the new stage cannot increase the prediction error power. It reduces the previous error power by the fraction $\lvert\kappa_m\rvert^2$.

If the autocorrelation matrix and its leading principal submatrices are positive definite, then the minimum prediction error power at every order is strictly positive:

$$P_m>0,\qquad m=0,1,\ldots,p.$$

Since $P_{m-1}>0$, the update formula forces

$$1-\lvert\kappa_m\rvert^2>0
\quad\Longrightarrow\quad
\lvert\kappa_m\rvert<1.$$

This is the key stability condition. The remaining step uses a polynomial stability theorem, not just the energy formula. In the Levinson-Durbin step-up recursion, the new prediction error filter is formed from the previous one by

$$A_m(z)=A_{m-1}(z)+\kappa_m z^{-m}A_{m-1}^{\ast}(1/z^{\ast}).$$

Here $A_{m-1}^{\ast}(1/z^{\ast})$ is the conjugate-reversed version of the previous polynomial. This formula is another way of writing the coefficient update in Levinson-Durbin recursion.

This formula shows the direct relationship between $\kappa_m$ and $A_m(z)$. The coefficient of the newest delay term $z^{-m}$ in $A_m(z)$ is exactly the reflection coefficient:

$$a_m^{(m)}=\kappa_m.$$

So $\kappa_m$ is the new last coefficient added when the predictor order increases from $m-1$ to $m$. But it also changes the older coefficients through the same step-up recursion:

$$a_k^{(m)}=a_k^{(m-1)}+\kappa_m\left[a_{m-k}^{(m-1)}\right]^{\ast},\qquad k=1,\ldots,m-1.$$

Thus $\kappa_m$ is not separate from $A_m(z)$. It is one parameter that tells us how to grow the whole prediction error filter from order $m-1$ to order $m$.

For example, when $m=1$,

$$A_1(z)=1+\kappa_1z^{-1}.$$

When $m=2$,

$$A_2(z)=1+a_1^{(2)}z^{-1}+\kappa_2z^{-2},$$

where

$$a_1^{(2)}=a_1^{(1)}+\kappa_2\left[a_1^{(1)}\right]^{\ast}.$$

So each $\kappa_m$ becomes the newest endpoint coefficient of $A_m(z)$ and also determines how the previous coefficients are adjusted.

The Schur-Cohn theorem says:

> If $A_{m-1}(z)$ is minimum phase, then the step-up polynomial $A_m(z)$ is minimum phase if and only if $\lvert\kappa_m\rvert<1$.

So the phrase "without ever moving a zero outside the unit circle" means the following inductive argument:

1. $A_0(z)=1$ has no zeros, so it is trivially minimum phase.
2. Positive definiteness gives $\lvert\kappa_1\rvert<1$, so the Schur-Cohn theorem implies $A_1(z)$ is minimum phase.
3. Positive definiteness also gives $\lvert\kappa_2\rvert<1$, so the next step preserves minimum phase and $A_2(z)$ is minimum phase.
4. Repeating this argument up to order $p$ gives

$$A_1(z),A_2(z),\ldots,A_p(z)$$

all minimum phase. Hence $A_p(z)$ is minimum phase.

In short, the energy update gives $\lvert\kappa_m\rvert<1$, and the Schur-Cohn theorem translates $\lvert\kappa_m\rvert<1$ into the zero-location statement.

Another way to say the same thing is:

| Condition | Consequence |
|----------|-------------|
| $\hat{\mathbf{R}}_p$ positive definite | every finite-order prediction error power is positive |
| $P_m=P_{m-1}(1-\lvert\kappa_m\rvert^2)$ | every reflection coefficient satisfies $\lvert\kappa_m\rvert<1$ |
| $\lvert\kappa_m\rvert<1$ for all stages | the prediction error filter $A_p(z)$ is minimum phase |

If the matrix were only positive semidefinite, then some $P_m$ could become zero. In that boundary case $\lvert\kappa_m\rvert=1$, and the corresponding prediction error filter can have a zero on the unit circle. Positive definiteness excludes this boundary case.

Therefore, the corresponding all-pole synthesis filter

$$H_p(z)=\frac{1}{A_p(z)}$$

is stable.

This stability property is one reason why the autocorrelation method is widely used in speech LPC and AR spectrum estimation.

### Weakness of the Autocorrelation Method

The zero-extension assumption is not always physically accurate. If the true signal does not actually become zero outside the observed interval, then the rectangular window creates artificial boundary effects. These boundary effects can bias the estimated autocorrelation and degrade model accuracy, especially for short records.

This motivates the covariance and modified covariance methods, discussed later in §7 and §10.

---

**§1 Basic Linear Prediction Model and the Autocorrelation Method**

- <a href="#11-why-linear-prediction-is-important" style="color: #a0a0a0;">1.1 Why Linear Prediction Is Important</a>
- <a href="#12-linear-prediction-as-lmmse-estimation" style="color: #a0a0a0;">1.2 Linear Prediction as LMMSE Estimation</a>
- <a href="#13-minimum-prediction-error-power" style="color: #a0a0a0;">1.3 Minimum Prediction Error Power</a>
- <a href="#14-the-autocorrelation-method-for-finite-data" style="color: #a0a0a0;">1.4 The Autocorrelation Method for Finite Data</a>
- [1.5 Windowing and Model Accuracy](#15-windowing-and-model-accuracy)

---

## 1.5 Windowing and Model Accuracy

The autocorrelation method with zero extension is equivalent to using a rectangular data window. A rectangular window has strong sidelobes in frequency, which can create spectral leakage.

A smoother window, such as a Hamming window, reduces sidelobes but broadens mainlobes. In the linear prediction context, this creates the same bias-resolution tradeoff encountered in spectrum estimation:

| Window | Advantage | Disadvantage |
|--------|-----------|--------------|
| Rectangular | High resolution; simple; preserves more data energy | Strong sidelobes; boundary discontinuity |
| Hamming | Lower sidelobes; smoother finite record | Lower resolution; broadened spectral peaks |

The choice of window changes the estimated autocorrelation sequence and therefore changes the autoregressive (AR) model.

---

### Answer to the Guiding Question

**Question.** A seabed hydrophone records slowly varying pump noise with $r(0)=10$, $r(1)=6$, and $r(2)=3$. What are the optimum two-sample prediction weights, minimum residual power, and percentage of original power removed?

**Answer.**

For the real second-order prediction-error filter $e(n)=x(n)+a_1x(n-1)+a_2x(n-2)$, the normal equations are

$$\begin{bmatrix}10&6\\6&10\end{bmatrix}
\begin{bmatrix}a_1\\a_2\end{bmatrix}
=-\begin{bmatrix}6\\3\end{bmatrix}.$$

The determinant is $100-36=64$, giving

$$a_1=-\frac{42}{64}=-0.65625,\qquad a_2=\frac{6}{64}=0.09375.$$

Because $e=x-\hat x$, the actual predictor is

$$\hat x(n)=0.65625x(n-1)-0.09375x(n-2).$$

The minimum residual power is

$$P_2=r(0)+a_1r(1)+a_2r(2)
=10-0.65625(6)+0.09375(3)=6.34375.$$

Thus the removed predictable power is $10-6.34375=3.65625$, or $3.65625/10=36.5625\%$. **Numerically, the prediction weights are $0.65625$ and $-0.09375$, the residual power is $6.34375$, and prediction removes $36.56\%$ of the hydrophone-noise power.**

---

# §2 Equivalence Between AR All-Pole Modeling and Linear Prediction

---

### Guiding Question

Low-frequency machinery sound at a passive sonar is modeled by $x(n)=0.75x(n-1)+w(n)$, where the innovation $w(n)$ is white with variance $4$. Determine the stationary signal power, lag-one autocorrelation, optimum one-step prediction coefficient, prediction-error power, and the causal whitening filter. Also calculate the PSD at $\omega=0$ and $\omega=\pi$ and their ratio.

Section 1 can fit a predictor from correlations, but it does not yet explain why the same coefficients define an all-pole acoustic generator and a whitening filter. Section 2 establishes that equivalence.

---

> 📖 Textbook §6.5 (Linear Prediction); §9.2 (Estimation of All-Pole Models)

---

**§2 Equivalence Between AR All-Pole Modeling and Linear Prediction**

- [2.1 The AR($p$) Model](#21-the-arp-model)
- <a href="#22-prediction-error-filter-equals-whitening-filter" style="color: #a0a0a0;">2.2 Prediction Error Filter Equals Whitening Filter</a>
- <a href="#23-yule-walker-equations-from-the-ar-model" style="color: #a0a0a0;">2.3 Yule-Walker Equations from the AR Model</a>
- <a href="#24-statistical-all-pole-modeling-vs-deterministic-linear-prediction" style="color: #a0a0a0;">2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction</a>
- <a href="#25-why-all-pole-models-are-useful" style="color: #a0a0a0;">2.5 Why All-Pole Models Are Useful</a>

---

## 2.1 The AR($p$) Model

An autoregressive process of order $p$ is defined by

$$\boxed{x(n)+\sum_{k=1}^{p}a_k x(n-k)=w(n)}$$

where $w(n)$ is white noise with variance $\sigma_w^2$.

In $z$-domain form,

$$A_p(z)x(n) \longleftrightarrow w(n), \qquad A_p(z)=1+\sum_{k=1}^{p}a_k z^{-k}.$$

The corresponding synthesis model is

$$\boxed{x(n)=H_p(z)w(n),\qquad H_p(z)=\frac{1}{A_p(z)}.}$$

Because $H_p(z)$ has only poles, AR modeling is also called all-pole modeling.

---

**§2 Equivalence Between AR All-Pole Modeling and Linear Prediction**

- <a href="#21-the-arp-model" style="color: #a0a0a0;">2.1 The AR($p$) Model</a>
- [2.2 Prediction Error Filter Equals Whitening Filter](#22-prediction-error-filter-equals-whitening-filter)
- <a href="#23-yule-walker-equations-from-the-ar-model" style="color: #a0a0a0;">2.3 Yule-Walker Equations from the AR Model</a>
- <a href="#24-statistical-all-pole-modeling-vs-deterministic-linear-prediction" style="color: #a0a0a0;">2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction</a>
- <a href="#25-why-all-pole-models-are-useful" style="color: #a0a0a0;">2.5 Why All-Pole Models Are Useful</a>

---

## 2.2 Prediction Error Filter Equals Whitening Filter

Compare the AR equation with the linear prediction error:

$$e_p^f(n)=x(n)+\sum_{k=1}^{p}a_k x(n-k).$$

For an exact AR($p$) process,

$$\boxed{e_p^f(n)=w(n).}$$

Therefore, the prediction error is the white-noise innovation. The prediction error filter $A_p(z)$ whitens the AR process.

This is the key equivalence:

> For an AR($p$) process, the optimal $p$-th order linear prediction error filter is the whitening filter.

Thus the same coefficients appear in three places:

| Object | Same polynomial |
|--------|-----------------|
| AR difference equation | $A_p(z)x(n)=w(n)$ |
| Prediction error filter | $e_p^f(n)=A_p(z)x(n)$ |
| All-pole synthesis filter | $x(n)=\dfrac{1}{A_p(z)}w(n)$ |

---

**§2 Equivalence Between AR All-Pole Modeling and Linear Prediction**

- <a href="#21-the-arp-model" style="color: #a0a0a0;">2.1 The AR($p$) Model</a>
- <a href="#22-prediction-error-filter-equals-whitening-filter" style="color: #a0a0a0;">2.2 Prediction Error Filter Equals Whitening Filter</a>
- [2.3 Yule-Walker Equations from the AR Model](#23-yule-walker-equations-from-the-ar-model)
- <a href="#24-statistical-all-pole-modeling-vs-deterministic-linear-prediction" style="color: #a0a0a0;">2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction</a>
- <a href="#25-why-all-pole-models-are-useful" style="color: #a0a0a0;">2.5 Why All-Pole Models Are Useful</a>

---

## 2.3 Yule-Walker Equations from the AR Model

Starting from

$$x(n)+\sum_{k=1}^{p}a_kx(n-k)=w(n),$$

multiply both sides by $x^{\ast}(n-i)$ and take expectation. Since the white-noise innovation is orthogonal to past samples, for $i=1,2,\ldots,p$,

$$E\{w(n)x^{\ast}(n-i)\}=0.$$

Therefore,

$$r_x(i)+\sum_{k=1}^{p}a_k r_x(i-k)=0,\qquad i=1,2,\ldots,p.$$

These are exactly the same equations as the optimal linear prediction equations.

For $i=0$,

$$r_x(0)+\sum_{k=1}^{p}a_k r_x(-k)=\sigma_w^2.$$

The right-hand side is no longer zero because the multiplier is the current
sample $x^{\ast}(n)$, not a past sample. More explicitly, after multiplying by
$x^{\ast}(n-i)$ and taking expectation, the right-hand side is

$$E\{w(n)x^{\ast}(n-i)\}.$$

For $i=1,2,\ldots,p$, this term is zero because $w(n)$ is orthogonal to the
past samples $x(n-i)$. For $i=0$, however,

$$E\{w(n)x^{\ast}(n)\}$$

must be evaluated. From the AR equation,

$$x(n)=w(n)-\sum_{k=1}^{p}a_kx(n-k),$$

and therefore

$$x^{\ast}(n)=w^{\ast}(n)-\sum_{k=1}^{p}a_k^{\ast}x^{\ast}(n-k).$$

Hence

$$
\begin{aligned}
E\{w(n)x^{\ast}(n)\}
&=E\{|w(n)|^2\}
-\sum_{k=1}^{p}a_k^{\ast}E\{w(n)x^{\ast}(n-k)\}\\
&=\sigma_w^2.
\end{aligned}
$$

The second term is zero because $w(n)$ is orthogonal to all past samples
$x(n-k)$, while $E\{|w(n)|^2\}$ is exactly the white-noise variance
$\sigma_w^2$.

This gives the innovation variance:

$$\boxed{\sigma_w^2=P_p=r_x(0)+\sum_{k=1}^{p}a_k r_x^{\ast}(k).}$$

Here the word **innovation** means the new, unpredictable part of the current
sample. Rewriting the AR model as

$$x(n)=-\sum_{k=1}^{p}a_kx(n-k)+w(n),$$

the term $-\sum_{k=1}^{p}a_kx(n-k)$ is the prediction of $x(n)$ from its past
$p$ samples, while $w(n)$ is the part left over after prediction. Thus
$w(n)$ is the innovation. Its variance is

$$E\{|w(n)|^2\}=\sigma_w^2.$$

But the linear prediction error is

$$e_p^f(n)=x(n)+\sum_{k=1}^{p}a_kx(n-k).$$

For an exact AR($p$) process, $e_p^f(n)=w(n)$. Therefore the prediction-error
power is the same as the innovation variance:

$$P_p=E\{|e_p^f(n)|^2\}=E\{|w(n)|^2\}=\sigma_w^2.$$

In words, the innovation variance is the average power of the part of the
current sample that cannot be predicted from the past $p$ samples.

---

**§2 Equivalence Between AR All-Pole Modeling and Linear Prediction**

- <a href="#21-the-arp-model" style="color: #a0a0a0;">2.1 The AR($p$) Model</a>
- <a href="#22-prediction-error-filter-equals-whitening-filter" style="color: #a0a0a0;">2.2 Prediction Error Filter Equals Whitening Filter</a>
- <a href="#23-yule-walker-equations-from-the-ar-model" style="color: #a0a0a0;">2.3 Yule-Walker Equations from the AR Model</a>
- [2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction](#24-statistical-all-pole-modeling-vs-deterministic-linear-prediction)
- <a href="#25-why-all-pole-models-are-useful" style="color: #a0a0a0;">2.5 Why All-Pole Models Are Useful</a>

---

## 2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction

There are two related but distinct problems.

### Statistical AR Modeling

Here $x(n)$ is assumed to be a realization of a WSS random process. We model its second-order statistics by an AR process. The autocorrelation sequence $r_x(l)$ is the central object.

The model is good if the AR spectrum

$$\boxed{R_x(e^{j\omega})=\frac{\sigma_w^2}{\lvert A_p(e^{j\omega})\rvert^2}}$$

matches the true or estimated power spectral density.

Why does the AR spectrum have this form? Start from the AR($p$) difference
equation

$$x(n)+a_1x(n-1)+\cdots+a_px(n-p)=w(n),$$

or

$$A_p(z)x(n)=w(n),\qquad A_p(z)=1+\sum_{k=1}^{p}a_kz^{-k}.$$

Thus

$$x(n)=\frac{1}{A_p(z)}w(n).$$

So an AR process can be viewed as white noise $w(n)$ passed through the
all-pole synthesis filter

$$H_p(z)=\frac{1}{A_p(z)}.$$

For a WSS input passed through an LTI filter, the output PSD is the input PSD
multiplied by the squared magnitude response:

$$R_x(e^{j\omega})=\lvert H_p(e^{j\omega})\rvert^2R_w(e^{j\omega}).$$

Since $w(n)$ is white noise, its PSD is flat:

$$R_w(e^{j\omega})=\sigma_w^2.$$

Therefore

$$R_x(e^{j\omega})
=\left\lvert\frac{1}{A_p(e^{j\omega})}\right\rvert^2\sigma_w^2
=\frac{\sigma_w^2}{\lvert A_p(e^{j\omega})\rvert^2}.$$

Intuitively, the white-noise input has equal power at all frequencies. The
filter $1/A_p(z)$ then reshapes that flat spectrum. Frequencies where
$\lvert A_p(e^{j\omega})\rvert$ is small are amplified, so the AR spectrum has
peaks there. Frequencies where $\lvert A_p(e^{j\omega})\rvert$ is large are
attenuated.

It is important not to confuse two different spectra:

| Quantity | How it is obtained | Meaning |
|----------|--------------------|---------|
| $\hat R_x(e^{j\omega})$ from a periodogram, Welch method, or correlation method | Estimated directly from the data | A nonparametric estimate of what the data spectrum looks like |
| $\hat R_x^{(AR)}(e^{j\omega})=\hat\sigma_w^2/\lvert \hat A_p(e^{j\omega})\rvert^2$ | Computed after fitting AR coefficients $\hat a_1,\ldots,\hat a_p$ and residual variance $\hat\sigma_w^2$ | The PSD implied by the fitted AR model |

Thus the AR spectrum is not automatically equal to the data-based PSD estimate.
It is a parametric model spectrum. A good AR model is one whose implied spectrum
matches the true PSD, or at least agrees well with a reliable data-based PSD
estimate. In other words, we use the comparison

$$\hat R_x^{(AR)}(e^{j\omega})\quad \text{versus}\quad \hat R_x(e^{j\omega})$$

as a diagnostic for whether the AR assumption is reasonable.

> **中文理解.** 这句话不是说“AR 谱本来就是从数据估计出来的 PSD，所以一定接近”。更准确地说，periodogram/Welch 等方法是直接从数据画出的 PSD 估计；AR 谱则是先假设数据满足 AR($p$) 模型，估计出 $\hat a_k$ 和 $\hat\sigma_w^2$，再由模型公式算出的 PSD。两条谱线接近，才说明这个 AR 模型对数据的谱结构解释得比较好。

### Deterministic Linear Prediction

Here $x(n)$ is treated as a finite deterministic data record. We choose coefficients to minimize a finite sum of squared errors. The central object is not the ensemble autocorrelation but the data matrix formed from the samples.

The two formulations become equivalent when the finite-sample correlation estimates are used as approximations to ensemble correlations. This is why the same algorithms appear in both AR spectrum estimation and deterministic linear prediction.

---

**§2 Equivalence Between AR All-Pole Modeling and Linear Prediction**

- <a href="#21-the-arp-model" style="color: #a0a0a0;">2.1 The AR($p$) Model</a>
- <a href="#22-prediction-error-filter-equals-whitening-filter" style="color: #a0a0a0;">2.2 Prediction Error Filter Equals Whitening Filter</a>
- <a href="#23-yule-walker-equations-from-the-ar-model" style="color: #a0a0a0;">2.3 Yule-Walker Equations from the AR Model</a>
- <a href="#24-statistical-all-pole-modeling-vs-deterministic-linear-prediction" style="color: #a0a0a0;">2.4 Statistical All-Pole Modeling vs Deterministic Linear Prediction</a>
- [2.5 Why All-Pole Models Are Useful](#25-why-all-pole-models-are-useful)

---

## 2.5 Why All-Pole Models Are Useful

All-pole models are especially effective for signals with spectral peaks. A pole near the unit circle creates a sharp spectral peak at the corresponding frequency. Therefore, an all-pole model can represent narrowband resonances with relatively few parameters.

Typical examples include:

- speech vocal-tract resonances,
- radar clutter resonances,
- narrowband interference,
- geophysical time series,
- biomedical oscillatory signals.

> ![Figure 2.1](./CourseADSP2026/Fig/Chapter_3/fig_9_3_textbook_fig_9_12_p452.png)
>
> *Figure 2.1 (Textbook Fig. 9.12, p. 452): Spectral-envelope matching property of all-pole models. All-pole models do not merely interpolate sample values; they provide a smooth resonant envelope that can capture spectral peaks.*

---

### Answer to the Guiding Question

**Question.** Low-frequency machinery sound is modeled by $x(n)=0.75x(n-1)+w(n)$ with white-innovation variance $4$. Find its power, lag-one autocorrelation, optimum prediction and whitening parameters, and its PSD values at $0$ and $\pi$.

**Answer.**

This is a stable AR(1) process because $|0.75|<1$. Its stationary power satisfies $r(0)=0.75^2r(0)+4$, so

$$r(0)=\frac{4}{1-0.75^2}=\frac{64}{7}\approx9.143,$$

and $r(1)=0.75r(0)=48/7\approx6.857$. The optimum predictor is $\hat x(n)=0.75x(n-1)$, and its error is exactly $w(n)$, with power $4$. Hence the prediction-error/whitening filter is

$$A(z)=1-0.75z^{-1},$$

while the acoustic synthesis filter is $1/A(z)$. The PSD is

$$R_x(e^{j\omega})=\frac{4}{|1-0.75e^{-j\omega}|^2}.$$

Therefore

$$R_x(e^{j0})=\frac4{0.25^2}=64,\qquad
R_x(e^{j\pi})=\frac4{1.75^2}=\frac{64}{49}\approx1.306.$$

Their ratio is exactly $49$. **Numerically, $r(0)=9.143$, $r(1)=6.857$, the predictor weight is $0.75$, residual power is $4$, the whitener coefficient is $-0.75$, and the low-to-high-frequency PSD ratio is $49$.**

---

# §3 Levinson-Durbin Recursive Algorithm

---

### Guiding Question

An underwater vehicle increases the order of a predictor for self-noise from one to two. Calibration gives $r(0)=10$, $r(1)=6$, and $r(2)=5$. Using an order-recursive calculation, find both reflection coefficients, the first- and second-order direct prediction-error-filter coefficients, and the residual powers after each stage. How much residual power does the second stage remove?

Solving a new matrix system at every order would ignore the Toeplitz structure learned in §1. The Levinson-Durbin recursion in this section produces all requested numbers stage by stage.

---

> 📖 Textbook §7.4 (Algorithms of Levinson and Levinson-Durbin); §7.1.3

---

**§3 Levinson-Durbin Recursive Algorithm**

- [3.1 The Computational Problem](#31-the-computational-problem)
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.1 The Computational Problem

The prediction coefficients are obtained from the Toeplitz system

$$\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p.$$

A generic matrix solver requires $O(p^3)$ operations. But $\mathbf{R}_p$ is not a generic matrix. It is Hermitian Toeplitz:

$$\mathbf{R}_p=\begin{bmatrix}
r(0)&r(1)&\cdots&r(p-1)\\
r^{\ast}(1)&r(0)&\cdots&r(p-2)\\
\vdots&\vdots&\ddots&\vdots\\
r^{\ast}(p-1)&r^{\ast}(p-2)&\cdots&r(0)
\end{bmatrix}.$$

Levinson-Durbin recursion exploits this structure and solves all orders $1,2,\ldots,p$ in $O(p^2)$ operations.

The idea is not merely to solve one equation. It solves a nested sequence of problems:

$$\mathbf{R}_1\mathbf{a}_1=-\mathbf{r}_1,$$

$$\mathbf{R}_2\mathbf{a}_2=-\mathbf{r}_2,$$

$$\cdots$$

$$\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p.$$

For the order-$m$ equation,

$$\mathbf{R}_m\mathbf{a}_m=-\mathbf{r}_m,$$

the shapes are

$$
\underbrace{\mathbf{R}_m}_{m\times m}
\underbrace{\mathbf{a}_m}_{m\times 1}
=
-\underbrace{\mathbf{r}_m}_{m\times 1}.
$$

More explicitly,

$$\mathbf{a}_m=[a_1^{(m)},a_2^{(m)},\ldots,a_m^{(m)}]^T,$$

and

$$\mathbf{r}_m=[r(1),r(2),\ldots,r(m)]^T.$$

Thus the left-hand side $\mathbf{R}_m\mathbf{a}_m$ is an $m\times 1$ vector, matching the right-hand side $-\mathbf{r}_m$. In the first-order case, $\mathbf{R}_1$ is a $1\times 1$ matrix and $\mathbf{r}_1=[r(1)]$ is a one-entry vector, so the equation looks scalar:

$$r(0)a_1=-r(1).$$

Each solution is built from the previous-order solution.

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- [3.2 The Order-Recursive View](#32-the-order-recursive-view)
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.2 The Order-Recursive View

Let

$$A_m(z)=1+\sum_{k=1}^{m}a_k^{(m)}z^{-k}$$

be the $m$-th order prediction error filter. The superscript $(m)$ means that the coefficient belongs to the order-$m$ predictor; it is not an exponent.

Suppose we already know $A_{m-1}(z)$ and the prediction error power $P_{m-1}$. The question is:

> What single new coefficient should be added to obtain the best order-$m$ predictor?

The new coefficient is the reflection coefficient

$$\kappa_m=a_m^{(m)}.$$

It measures how much correlation remains between the forward and backward prediction errors at order $m-1$.

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- [3.3 Levinson-Durbin Recursion Formulas](#33-levinson-durbin-recursion-formulas)
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.3 Levinson-Durbin Recursion Formulas

Initialize:

$$P_0=r(0).$$

For $m=1,2,\ldots,p$, compute the prediction-error correlation numerator:

$$\alpha_m=r(m)+\sum_{k=1}^{m-1}a_k^{(m-1)}r(m-k).$$

Then the reflection coefficient is

$$\boxed{\kappa_m=-\frac{\alpha_m}{P_{m-1}}.}$$

Update the coefficients:

$$\boxed{a_m^{(m)}=\kappa_m}$$

and for $k=1,2,\ldots,m-1$,

$$\boxed{a_k^{(m)}=a_k^{(m-1)}+\kappa_m\left[a_{m-k}^{(m-1)}\right]^{\ast}.}$$

Update the prediction error power:

$$\boxed{P_m=P_{m-1}(1-\lvert\kappa_m\rvert^2).}$$

These equations are the core of the Levinson-Durbin algorithm.

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- [3.4 Why the Error Recursion Makes Sense](#34-why-the-error-recursion-makes-sense)
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.4 Why the Error Recursion Makes Sense

The update

$$P_m=P_{m-1}(1-\lvert\kappa_m\rvert^2)$$

is extremely informative.

If $\lvert\kappa_m\rvert$ is close to zero, the new stage adds little predictive power, so $P_m\approx P_{m-1}$.

If $\lvert\kappa_m\rvert$ is large, the new lag explains a significant part of the residual correlation, so the error power decreases strongly.

If $\lvert\kappa_m\rvert<1$, then $P_m>0$. This connects numerical recursion, signal predictability, and filter stability.

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- [3.5 Reflection Coefficient as Partial Correlation](#35-reflection-coefficient-as-partial-correlation)
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.5 Reflection Coefficient as Partial Correlation

The reflection coefficient is also called the partial autocorrelation coefficient (PARCOR). It measures the correlation between two samples after the effects of the intermediate samples have already been removed.

For example, $\kappa_3$ is not simply the correlation between $x(n)$ and $x(n-3)$. It is the remaining correlation between them after the information in $x(n-1)$ and $x(n-2)$ has been accounted for.

This interpretation is why reflection coefficients are useful for model-order selection.

Suppose the signal is truly generated by an AR($p$) model:

$$x(n)+a_1x(n-1)+a_2x(n-2)+\cdots+a_px(n-p)=w(n),$$

where $w(n)$ is the innovation, or the part of $x(n)$ that cannot be predicted from the past. This equation says that, after lags $1,\ldots,p$ have been used, there is no additional linear predictive information left in older samples.

The reflection coefficient $\kappa_m$ asks a more refined question than the ordinary autocorrelation:

> After the effects of lags $1,\ldots,m-1$ have already been removed, does lag $m$ still add new predictive information?

For an ideal AR($p$) process, the answer can be nonzero for $m\le p$, because those lags are part of the true AR model. But for $m>p$, the answer should be zero: once the first $p$ lags have been accounted for, lag $p+1$, lag $p+2$, and so on do not add any new partial correlation. Therefore,

$$\kappa_m\ne 0 \quad \text{for some } m\le p,$$

but

$$\kappa_m=0 \quad \text{for all } m>p$$

in the ideal infinite-data AR($p$) case.

This gives a practical order-selection rule. Compute the sequence

$$\kappa_1,\kappa_2,\kappa_3,\ldots$$

and look for the point after which the coefficients are no longer significantly different from zero. The last significant reflection coefficient suggests the AR order. For example, if $\kappa_1,\ldots,\kappa_4$ are significant but $\kappa_5,\kappa_6,\ldots$ are small, then AR(4) is a natural candidate.

In real finite data, the coefficients after the true order will not be exactly zero because of sample variability and noise. Thus $p$ is not known in advance; it is estimated from the data by checking where the PARCOR sequence effectively cuts off, often together with criteria such as prediction error power, AIC/BIC, or residual whiteness.

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- [3.6 Direct Form vs Recursive Structure](#36-direct-form-vs-recursive-structure)
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.6 Direct Form vs Recursive Structure

The direct-form predictor uses $p$ taps at once. The recursive approach builds the predictor stage by stage. At every stage, the newly introduced reflection coefficient corrects both the forward and backward prediction errors.

This stage-wise interpretation leads naturally to lattice filters.

The sentence above is important. It means that the Levinson-Durbin recursion is not just an efficient way to compute a final list of direct-form coefficients. It also suggests a physical filter structure. Since the predictor is grown one order at a time, the implementation can also be built one section at a time. Each new section introduces one reflection coefficient $\kappa_m$, removes one more layer of residual correlation, and produces a higher-order prediction error. A cascade of these local order-update sections is exactly the idea behind a lattice filter.

Equivalently, compare two ways of thinking about the same predictor:

| View | What happens |
|------|--------------|
| Direct-form view | Use all $p$ coefficients $a_1^{(p)},\ldots,a_p^{(p)}$ at once to form the final prediction error |
| Order-recursive view | Start from order $0$, add one new coefficient $\kappa_m$ at each stage, and update the error signals |
| Lattice-filter view | Implement each order update as one local lattice section controlled by $\kappa_m$ |

Thus the reflection coefficient has two meanings at the same time:

1. statistically, $\kappa_m$ measures the remaining partial correlation after lower-order lags have already been accounted for;
2. structurally, $\kappa_m$ is the parameter of the $m$-th lattice section.

This is why the word "stage" matters. The $m$-th stage does not redesign the whole predictor. It only asks:

> After stages $1,\ldots,m-1$ have removed the predictable structure up to lag $m-1$, how much residual correlation remains at lag $m$?

The answer is encoded by $\kappa_m$. If $\kappa_m$ is small, the new stage contributes little. If $\kappa_m$ is large in magnitude, the new stage removes a substantial part of the remaining prediction error power.

> ![Figure 3.1](./CourseADSP2026/Fig/Chapter_3/fig_3_1_textbook_fig_7_1_p337.png)
>
> *Figure 3.1 (Textbook Fig. 7.1, p. 337): Orthogonal order-recursive structure for linear MMSE estimation. This figure is useful for understanding why order-recursive algorithms can update an estimator by adding one orthogonalized component at a time.*

Figure 3.1 shows the same idea in a more general linear-estimation setting. The original inputs $x_1,x_2,x_3,x_4$ are usually correlated with one another. The block labeled **Decorrelator** converts them into innovation variables $w_1,w_2,w_3,w_4$. An innovation is the part of a variable that cannot be linearly predicted from the variables that have already been processed.

For example:

- $w_1=x_1$ is the first available component.
- $w_2$ is the part of $x_2$ that remains after the component predictable from $x_1$ has been removed.
- $w_3$ is the part of $x_3$ that remains after the components predictable from $x_1$ and $x_2$ have been removed.
- $w_4$ is the new information in $x_4$ after the earlier inputs have already been accounted for.

So the decorrelator performs a step-by-step orthogonalization:

$$x_1,x_2,x_3,x_4
\quad\longrightarrow\quad
w_1,w_2,w_3,w_4.$$

The notation in Figure 3.1 uses the textbook's generic LMMSE-estimation notation. In this chapter, we should translate it back to the linear-prediction notation already used above:

$$
y \longleftrightarrow x(n),\qquad
x_i \longleftrightarrow x(n-i),\qquad
\hat y_i \longleftrightarrow \hat{x}_i(n),\qquad
\mathbf{d}_i \longleftrightarrow \mathbf{r}_i.
$$

Thus $\hat y_i$ in the figure is not a new signal introduced for this chapter. It is the estimate of the desired sample after the first $i$ orthogonalized input components have been used. In our notation, this is the order-$i$ prediction

$$
\hat{x}_i(n)=-\sum_{k=1}^{i}a_k^{(i)\ast}x(n-k),
\qquad
e_i^f(n)=x(n)-\hat{x}_i(n).
$$

Similarly, the figure's $\mathbf{d}$ is the generic cross-correlation vector between the input vector and the desired response. For the order-$i$ prediction problem, the desired response is $x(n)$ and the input vector is

$$
\mathbf{x}_i(n)=\begin{bmatrix}x(n-1)&\cdots&x(n-i)\end{bmatrix}^T,
$$

so the same information is the prediction correlation vector

$$
\mathbf{r}_i=\begin{bmatrix}r_x(1)&r_x(2)&\cdots&r_x(i)\end{bmatrix}^T,
$$

up to the conjugation convention. The reason our prediction normal equation is written as $\mathbf{R}_i\mathbf{a}_i=-\mathbf{r}_i$ rather than $\mathbf{R}_i\mathbf{c}_i=\mathbf{d}_i$ is that $\mathbf{a}_i$ is the prediction-error coefficient vector, while the actual predictor coefficients are $-\mathbf{a}_i$.

| Notation | Meaning |
|----------|---------|
| $x_i$ | Generic input component; for prediction, read this as a lagged sample such as $x(n-i)$ |
| $w_i$ | Orthogonalized version of the input component; conceptually, the new information left after earlier lags have been removed |
| $\hat y_i$ | Generic running estimate; for prediction, read this as the order-$i$ predictor output $\hat{x}_i(n)$ |
| $\mathbf{R}$ | Input autocorrelation or covariance matrix |
| $\mathbf{d}$ | Generic cross-correlation vector; for prediction, this carries the same correlation information as $\mathbf{r}_i=[r_x(1),\ldots,r_x(i)]^T$ |
| $\mathbf{R}=\mathbf{L}\mathbf{D}\mathbf{L}^H$ | LDL$^H$ factorization of the correlation matrix |
| $\mathbf{B}=\mathbf{L}^{-1}$ | Matrix representation of the decorrelator |
| $\mathbf{k}=\mathbf{D}^{-1}\mathbf{B}\mathbf{d}$ | Optimum linear-combiner coefficients after decorrelation |
| $b_j^{(m)}$ | Coefficients used inside the decorrelator at stage $m$ |
| $k_i^\ast$ | Complex conjugate of the $i$-th combiner coefficient |
| $(\cdot)^\ast$ | Complex conjugate |
| $(\cdot)^H$ | Hermitian transpose, or conjugate transpose |

The small processing element in the lower-left corner implements

$$y_{\text{out}}=b x_{\text{in}}+a_{\text{in}}.$$

It multiplies one input by a coefficient $b$, adds another input, and passes the result forward. Repeating this simple local operation creates the triangular decorrelator. This triangular, order-by-order structure is the conceptual ancestor of the lattice filter.

> ![Figure 3.2](./CourseADSP2026/Fig/Chapter_3/fig_3_2_textbook_fig_7_2_p341.png)
>
> *Figure 3.2 (Textbook Fig. 7.2, p. 341): Gram-Schmidt orthogonalization. Levinson-type recursions can be interpreted as specialized orthogonalization procedures adapted to Toeplitz correlation matrices.*

Figure 3.2 gives the geometric version of the same story. In ordinary Gram-Schmidt orthogonalization, a vector is decomposed into two parts:

1. its projection onto the subspace already spanned by previous orthogonal vectors;
2. the leftover component that is orthogonal to that subspace.

The leftover component becomes the next innovation. In the $m=2$ drawing, $x_2$ is split into a component along $w_1$ and a new orthogonal component $w_2$. In the $m=3$ drawing, $x_3$ is split into components along $w_1$ and $w_2$, plus a new orthogonal component $w_3$.

Levinson-Durbin can be viewed as a specialized Gram-Schmidt procedure for Toeplitz autocorrelation matrices. Lattice filters then turn this order-recursive orthogonalization into a signal-flow structure. Later, the all-zero lattice recursions will make this explicit:

$$e_m^f(n)=e_{m-1}^f(n)+\kappa_m^{\ast}e_{m-1}^b(n-1),$$

$$e_m^b(n)=e_{m-1}^b(n-1)+\kappa_m e_{m-1}^f(n).$$

These equations say that the $m$-th lattice stage combines the previous forward and backward prediction errors using one parameter $\kappa_m$. The goal is local decorrelation: after the update, the order-$m$ errors contain less predictable structure than the order-$(m-1)$ errors.

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- [3.7 Algorithm Table](#37-algorithm-table)
- <a href="#38-simple-first-order-example" style="color: #a0a0a0;">3.8 Simple First-Order Example</a>

---

## 3.7 Algorithm Table

| Step | Operation |
|------|-----------|
| 1 | Set $P_0=r(0)$ and start with no prediction coefficients |
| 2 | For order $m$, compute $\alpha_m=r(m)+\sum_{k=1}^{m-1}a_k^{(m-1)}r(m-k)$ |
| 3 | Compute reflection coefficient $\kappa_m=-\alpha_m/P_{m-1}$ |
| 4 | Update direct-form coefficients by $a_k^{(m)}=a_k^{(m-1)}+\kappa_m[a_{m-k}^{(m-1)}]^\ast$ |
| 5 | Set $a_m^{(m)}=\kappa_m$ |
| 6 | Update $P_m=P_{m-1}(1-\lvert\kappa_m\rvert^2)$ |
| 7 | Continue until $m=p$ |

---

**§3 Levinson-Durbin Recursive Algorithm**

- <a href="#31-the-computational-problem" style="color: #a0a0a0;">3.1 The Computational Problem</a>
- <a href="#32-the-order-recursive-view" style="color: #a0a0a0;">3.2 The Order-Recursive View</a>
- <a href="#33-levinson-durbin-recursion-formulas" style="color: #a0a0a0;">3.3 Levinson-Durbin Recursion Formulas</a>
- <a href="#34-why-the-error-recursion-makes-sense" style="color: #a0a0a0;">3.4 Why the Error Recursion Makes Sense</a>
- <a href="#35-reflection-coefficient-as-partial-correlation" style="color: #a0a0a0;">3.5 Reflection Coefficient as Partial Correlation</a>
- <a href="#36-direct-form-vs-recursive-structure" style="color: #a0a0a0;">3.6 Direct Form vs Recursive Structure</a>
- <a href="#37-algorithm-table" style="color: #a0a0a0;">3.7 Algorithm Table</a>
- [3.8 Simple First-Order Example](#38-simple-first-order-example)

---

## 3.8 Simple First-Order Example

For $p=1$, the normal equation is

$$r(0)a_1=-r(1).$$

Thus

$$a_1=-\frac{r(1)}{r(0)}.$$

The first reflection coefficient is

$$\kappa_1=a_1=-\frac{r(1)}{r(0)}.$$

The first-order prediction error power is

$$P_1=r(0)(1-\lvert\kappa_1\rvert^2)=r(0)-\frac{\lvert r(1)\rvert^2}{r(0)}.$$

This result says: the more strongly $x(n)$ is correlated with $x(n-1)$, the more the prediction error power is reduced.

---

### Answer to the Guiding Question

**Question.** For self-noise with $r(0)=10$, $r(1)=6$, and $r(2)=5$, find the order-one and order-two Levinson-Durbin parameters and error powers, and quantify the second stage's benefit.

**Answer.**

Initialize $P_0=10$. At order one,

$$\kappa_1=-\frac{r(1)}{P_0}=-0.6,\qquad
a_1^{(1)}=-0.6,$$

$$P_1=P_0(1-\kappa_1^2)=10(1-0.36)=6.4.$$

At order two, the unexplained lag-two correlation is

$$\alpha_2=r(2)+a_1^{(1)}r(1)=5-0.6(6)=1.4,$$

so

$$\kappa_2=-\frac{1.4}{6.4}=-0.21875.$$

The step-up update gives

$$a_1^{(2)}=-0.6+(-0.21875)(-0.6)=-0.46875,\qquad
a_2^{(2)}=-0.21875.$$

Finally,

$$P_2=6.4(1-0.21875^2)=6.09375.$$

The second stage removes $P_1-P_2=0.30625$ power units, which is $4.785\%$ of the order-one residual. **Numerically, $(\kappa_1,\kappa_2)=(-0.6,-0.21875)$, the final polynomial is $1-0.46875z^{-1}-0.21875z^{-2}$, and the error powers are $10\to6.4\to6.09375$.**

---

# §4 Three Equivalent Sets of Recursive Parameters

---

### Guiding Question

A compact lattice model for correlated ocean ambient noise has initial power $P_0=8$ and reflection coefficients $\kappa_1=-0.5$, $\kappa_2=0.25$. Convert these lattice parameters into the second-order direct-form prediction polynomial, calculate $P_1$ and $P_2$, and locate both polynomial zeros numerically. Is the corresponding all-pole noise generator stable?

Levinson-Durbin outputs several parameter sets, but the preceding section does not yet show how to convert among reflection coefficients, direct coefficients, error powers, and stable polynomial roots. Section 4 makes those descriptions interchangeable.

---

> 📖 Textbook §7.1–§7.3, §7.5, §7.7

---

**§4 Three Equivalent Sets of Recursive Parameters**

- [4.1 Three Equivalent Descriptions](#41-three-equivalent-descriptions)
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.1 Three Equivalent Descriptions

A stable minimum-phase prediction error filter can be represented in at least three equivalent ways:

1. direct-form coefficients $a_1,\ldots,a_p$,
2. reflection coefficients $\kappa_1,\ldots,\kappa_p$,
3. prediction error powers $P_0,P_1,\ldots,P_p$.

The same prediction model can be converted from one representation to another.

> ![Figure 4.1](./CourseADSP2026/Fig/Chapter_3/fig_4_1_textbook_fig_7_4_p356.png)
>
> *Figure 4.1 (Textbook Fig. 7.4, p. 356): Direct-form structure for computing forward and backward prediction errors.*

In this figure, the lower branch computes the backward prediction error. The coefficients labeled $b_0^{(m)\ast},b_1^{(m)\ast},\ldots,b_{m-1}^{(m)\ast}$ are the conjugated entries of the direct-form backward coefficient vector

$$\mathbf{b}_m=[b_0^{(m)},b_1^{(m)},\ldots,b_{m-1}^{(m)}]^T.$$

Thus the lower output is

$$e_m^b(n)=x(n-m)+\sum_{k=0}^{m-1}b_k^{(m)\ast}x(n-k).$$

The rightmost coefficient $1$ on the lower branch multiplies the oldest sample $x(n-m)$. It plays the same role as the leading coefficient $1$ in the forward prediction error filter. For a WSS process with the usual optimum forward/backward predictors, the backward coefficients are the conjugate-reversed version of the forward coefficients:

$$\mathbf{b}_m=\mathbf{J}\mathbf{a}_m^{\ast}.$$

> ![Figure 4.2](./CourseADSP2026/Fig/Chapter_3/fig_4_2_textbook_fig_7_7_p361.png)
>
> *Figure 4.2 (Textbook Fig. 7.7, p. 361): Equivalent representations for minimum-phase linear prediction error filters. Autocorrelation values, reflection coefficients, direct-form coefficients, and error powers are different parameterizations of the same object.*

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- [4.2 Positive Definiteness, Reflection Coefficients, and Stability](#42-positive-definiteness-reflection-coefficients-and-stability)
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.2 Positive Definiteness, Reflection Coefficients, and Stability

For a valid WSS process, the autocorrelation matrix must be nonnegative definite. If it is positive definite, the Levinson-Durbin recursion gives

$$P_m>0,\qquad m=0,1,\ldots,p.$$

Since

$$P_m=P_{m-1}(1-\lvert\kappa_m\rvert^2),$$

positive prediction error power implies

$$\boxed{\lvert\kappa_m\rvert<1.}$$

This condition has a filter interpretation:

> The prediction error filter $A_p(z)$ is minimum phase if and only if all reflection coefficients satisfy $\lvert\kappa_m\rvert<1$.

This is extremely useful because checking stability through roots of $A_p(z)$ can be expensive and numerically sensitive. Checking reflection coefficients is simple.

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- [4.3 Boundary Case: Zero on the Unit Circle](#43-boundary-case-zero-on-the-unit-circle)
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.3 Boundary Case: Zero on the Unit Circle

If $\lvert\kappa_m\rvert=1$ for some stage, then

$$P_m=0.$$

This means that the signal becomes perfectly predictable at order $m$. The corresponding prediction error filter has a zero on the unit circle. In theory this may occur for deterministic sinusoidal components. In practice, for noisy data, exact equality is rare, but values very close to one indicate strong predictability or numerical ill-conditioning.

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- [4.4 Cholesky / LDL$^H$ Decomposition View](#44-cholesky--ldlh-decomposition-view)
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.4 Cholesky / LDL$^H$ Decomposition View

The autocorrelation matrix has a factorization

$$\boxed{\mathbf{R}_p=\mathbf{L}_p\mathbf{D}_p\mathbf{L}_p^H}$$

where $\mathbf{L}_p$ is lower triangular and $\mathbf{D}_p$ is diagonal.

The diagonal entries of $\mathbf{D}_p$ are prediction error powers:

$$\mathbf{D}_p=\mathrm{diag}\{P_0,P_1,\ldots,P_{p-1}\}$$

up to conventions of indexing.

This means the Levinson-Durbin algorithm can be interpreted as a structured Cholesky factorization of a Toeplitz matrix.

The key equivalence is:

$$\mathbf{R}_p>0 \quad\Longleftrightarrow\quad P_m>0\ \text{for all }m \quad\Longleftrightarrow\quad \lvert\kappa_m\rvert<1\ \text{for all }m.$$

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- [4.5 Autocorrelation Extension and Maximum Entropy](#45-autocorrelation-extension-and-maximum-entropy)
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.5 Autocorrelation Extension and Maximum Entropy

Suppose we know only

$$r(0),r(1),\ldots,r(p).$$

How should we extend the autocorrelation sequence to $r(p+1),r(p+2),\ldots$?

There are infinitely many possible extensions. However, a valid extension must preserve nonnegative definiteness of every enlarged Toeplitz matrix.

Levinson-Durbin gives a natural way to think about this. When we extend from order $p$ to order $p+1$, we are effectively choosing a new reflection coefficient $\kappa_{p+1}$. The extension is valid if

$$\lvert\kappa_{p+1}\rvert\le 1.$$

The maximum entropy extension chooses

$$\kappa_{p+1}=\kappa_{p+2}=\cdots=0.$$

This is equivalent to choosing an AR($p$) model. In other words, the AR($p$) model is the least committed extension: it preserves the known correlations but assumes no additional partial correlation beyond order $p$.

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- [4.6 Step-Up Recursion: Lattice to Direct Form](#46-step-up-recursion-lattice-to-direct-form)
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.6 Step-Up Recursion: Lattice to Direct Form

Given reflection coefficients $\kappa_1,\ldots,\kappa_p$, we can recover direct-form coefficients using the step-up recursion.

Initialize $A_0(z)=1$ and $P_0=r(0)$. For $m=1,2,\ldots,p$,

$$a_m^{(m)}=\kappa_m,$$

$$a_k^{(m)}=a_k^{(m-1)}+\kappa_m[a_{m-k}^{(m-1)}]^\ast,\qquad k=1,\ldots,m-1.$$

This is exactly the coefficient update used inside Levinson-Durbin.

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- [4.7 Step-Down Recursion: Direct Form to Lattice](#47-step-down-recursion-direct-form-to-lattice)
- <a href="#48-why-reflection-coefficients-are-numerically-attractive" style="color: #a0a0a0;">4.8 Why Reflection Coefficients Are Numerically Attractive</a>

---

## 4.7 Step-Down Recursion: Direct Form to Lattice

Given a stable direct-form polynomial $A_p(z)$, we can recover the reflection coefficients by reversing the step-up recursion.

At order $m$, the last coefficient is

$$\kappa_m=a_m^{(m)}.$$

Then for $k=1,\ldots,m-1$,

$$\boxed{a_k^{(m-1)}=\frac{a_k^{(m)}-\kappa_m[a_{m-k}^{(m)}]^\ast}{1-\lvert\kappa_m\rvert^2}.}$$

This recursion is also a stability test. If at any stage $\lvert\kappa_m\rvert\ge 1$, the polynomial is not minimum phase.

---

**§4 Three Equivalent Sets of Recursive Parameters**

- <a href="#41-three-equivalent-descriptions" style="color: #a0a0a0;">4.1 Three Equivalent Descriptions</a>
- <a href="#42-positive-definiteness-reflection-coefficients-and-stability" style="color: #a0a0a0;">4.2 Positive Definiteness, Reflection Coefficients, and Stability</a>
- <a href="#43-boundary-case-zero-on-the-unit-circle" style="color: #a0a0a0;">4.3 Boundary Case: Zero on the Unit Circle</a>
- <a href="#44-cholesky--ldlh-decomposition-view" style="color: #a0a0a0;">4.4 Cholesky / LDL$^H$ Decomposition View</a>
- <a href="#45-autocorrelation-extension-and-maximum-entropy" style="color: #a0a0a0;">4.5 Autocorrelation Extension and Maximum Entropy</a>
- <a href="#46-step-up-recursion-lattice-to-direct-form" style="color: #a0a0a0;">4.6 Step-Up Recursion: Lattice to Direct Form</a>
- <a href="#47-step-down-recursion-direct-form-to-lattice" style="color: #a0a0a0;">4.7 Step-Down Recursion: Direct Form to Lattice</a>
- [4.8 Why Reflection Coefficients Are Numerically Attractive](#48-why-reflection-coefficients-are-numerically-attractive)

---

## 4.8 Why Reflection Coefficients Are Numerically Attractive

Reflection coefficients are bounded for stable models:

$$\lvert\kappa_m\rvert<1.$$

This boundedness makes them more robust under quantization than direct-form coefficients. A small perturbation in direct-form coefficients can move roots across the unit circle. A small perturbation in reflection coefficients is less likely to break stability as long as the perturbed values remain inside the unit disk.

This is one of the practical reasons why lattice filters are common in speech processing and adaptive filtering.

---

### Answer to the Guiding Question

**Question.** A lattice ambient-noise model has $P_0=8$, $\kappa_1=-0.5$, and $\kappa_2=0.25$. Convert it to a direct polynomial, calculate its stage powers and zeros, and test stability.

**Answer.**

At order one, $a_1^{(1)}=\kappa_1=-0.5$. The order-two step-up recursion gives

$$a_1^{(2)}=a_1^{(1)}+\kappa_2a_1^{(1)}=-0.5+0.25(-0.5)=-0.625,$$

$$a_2^{(2)}=\kappa_2=0.25.$$

Thus

$$A_2(z)=1-0.625z^{-1}+0.25z^{-2}.$$

The error powers are

$$P_1=8(1-0.5^2)=6,$$

$$P_2=6(1-0.25^2)=5.625.$$

Multiplying $A_2(z)=0$ by $z^2$ gives $z^2-0.625z+0.25=0$. Its zeros are

$$z_{1,2}=0.3125\pm j0.3903,$$

each with magnitude $\sqrt{0.3125^2+0.3903^2}=0.5$. Both zeros lie inside the unit circle; equivalently, both reflection-coefficient magnitudes are below one. **Numerically, the direct coefficients are $(-0.625,0.25)$, the powers are $8\to6\to5.625$, the zeros are $0.3125\pm j0.3903$, and the reciprocal all-pole generator is stable.**

---

# §5 Schur Recursive Algorithm

---

### Guiding Question

A fixed-point processor in an underwater acoustic sensor will implement a two-stage lattice prewhitener and therefore needs reflection coefficients directly, not a direct-form predictor. Its measured autocorrelation sequence is $r(0)=4$, $r(1)=2$, $r(2)=1.5$. What two reflection coefficients and two stage error powers should be loaded, and does each stage satisfy the strict stability test?

The conversion formulas of §4 work once direct coefficients are available. Section 5 instead develops a Schur recursion aimed directly at the bounded lattice parameters needed here.

---

> 📖 Textbook §7.6 (Algorithm of Schür)

---

**§5 Schur Recursive Algorithm**

- [5.1 Goal of the Schur Algorithm](#51-goal-of-the-schur-algorithm)
- <a href="#52-gapped-correlation-functions" style="color: #a0a0a0;">5.2 Gapped Correlation Functions</a>
- <a href="#53-algorithmic-interpretation" style="color: #a0a0a0;">5.3 Algorithmic Interpretation</a>
- <a href="#54-schur-algorithm-summary" style="color: #a0a0a0;">5.4 Schur Algorithm Summary</a>
- <a href="#55-relationship-to-levinson-durbin" style="color: #a0a0a0;">5.5 Relationship to Levinson-Durbin</a>

---

## 5.1 Goal of the Schur Algorithm

The Levinson-Durbin algorithm computes both:

- direct-form prediction coefficients $a_k^{(m)}$,
- reflection coefficients $\kappa_m$.

The Schur algorithm focuses on computing the reflection coefficients directly from the autocorrelation sequence without explicitly computing all direct-form coefficients.

This is useful when the final implementation is a lattice filter, because the lattice filter only needs the reflection coefficients.

---

**§5 Schur Recursive Algorithm**

- <a href="#51-goal-of-the-schur-algorithm" style="color: #a0a0a0;">5.1 Goal of the Schur Algorithm</a>
- [5.2 Gapped Correlation Functions](#52-gapped-correlation-functions)
- <a href="#53-algorithmic-interpretation" style="color: #a0a0a0;">5.3 Algorithmic Interpretation</a>
- <a href="#54-schur-algorithm-summary" style="color: #a0a0a0;">5.4 Schur Algorithm Summary</a>
- <a href="#55-relationship-to-levinson-durbin" style="color: #a0a0a0;">5.5 Relationship to Levinson-Durbin</a>

---

## 5.2 Gapped Correlation Functions

The Schur algorithm introduces two auxiliary sequences:

$$\xi_m^f(l)=E\{x(n-l)[e_m^f(n)]^{\ast}\},$$

$$\xi_m^b(l)=E\{x(n-l)[e_m^b(n)]^{\ast}\}.$$

They are called gapped functions because orthogonality creates intervals of zeros. For example, the forward prediction error is orthogonal to the samples used by the forward predictor.

The key recursion is

$$\xi_m^f(l)=\xi_{m-1}^f(l)+\kappa_{m-1}^{\ast}\xi_{m-1}^b(l-1),$$

$$\xi_m^b(l)=\kappa_{m-1}\xi_{m-1}^f(l)+\xi_{m-1}^b(l-1).$$

The reflection coefficient is then obtained by

$$\boxed{\kappa_m=-\frac{\xi_m^f(m+1)}{\xi_m^b(m)}.}$$

---

**§5 Schur Recursive Algorithm**

- <a href="#51-goal-of-the-schur-algorithm" style="color: #a0a0a0;">5.1 Goal of the Schur Algorithm</a>
- <a href="#52-gapped-correlation-functions" style="color: #a0a0a0;">5.2 Gapped Correlation Functions</a>
- [5.3 Algorithmic Interpretation](#53-algorithmic-interpretation)
- <a href="#54-schur-algorithm-summary" style="color: #a0a0a0;">5.4 Schur Algorithm Summary</a>
- <a href="#55-relationship-to-levinson-durbin" style="color: #a0a0a0;">5.5 Relationship to Levinson-Durbin</a>

---

## 5.3 Algorithmic Interpretation

At stage $m$, the Schur algorithm asks:

> After removing all predictable structure up to order $m$, what residual forward/backward correlation remains at the next lag?

That residual correlation determines the next reflection coefficient.

If the remaining correlation is small, $\kappa_m$ is small and the next lattice stage contributes little.

If the remaining correlation is large, the next stage is important.

---

**§5 Schur Recursive Algorithm**

- <a href="#51-goal-of-the-schur-algorithm" style="color: #a0a0a0;">5.1 Goal of the Schur Algorithm</a>
- <a href="#52-gapped-correlation-functions" style="color: #a0a0a0;">5.2 Gapped Correlation Functions</a>
- <a href="#53-algorithmic-interpretation" style="color: #a0a0a0;">5.3 Algorithmic Interpretation</a>
- [5.4 Schur Algorithm Summary](#54-schur-algorithm-summary)
- <a href="#55-relationship-to-levinson-durbin" style="color: #a0a0a0;">5.5 Relationship to Levinson-Durbin</a>

---

## 5.4 Schur Algorithm Summary

| Step | Operation |
|------|-----------|
| 1 | Input $r(0),r(1),\ldots,r(p)$ |
| 2 | Initialize $\xi_0^f(l)=\xi_0^b(l)=r(l)$ |
| 3 | Compute $\kappa_0=-\xi_0^f(1)/\xi_0^b(0)$ |
| 4 | Use the lattice recursion to update $\xi_m^f(l)$ and $\xi_m^b(l)$ |
| 5 | At each order, compute $\kappa_m=-\xi_m^f(m+1)/\xi_m^b(m)$ |
| 6 | Stop at the desired order |

> ![Figure 5.1](./CourseADSP2026/Fig/Chapter_3/fig_5_1_textbook_fig_7_8_p366.png)
>
> *Figure 5.1 (Textbook Fig. 7.8, p. 366): Tree decomposition for Schur-algorithm computations.*

> ![Figure 5.2](./CourseADSP2026/Fig/Chapter_3/fig_5_2_textbook_fig_7_9_p367.png)
>
> *Figure 5.2 (Textbook Fig. 7.9, p. 367): Superlattice organization of the Schur algorithm. The input is the autocorrelation sequence; the output is the lattice-parameter sequence.*

> ![Figure 5.3](./CourseADSP2026/Fig/Chapter_3/fig_5_3_textbook_fig_7_10_p368.png)
>
> *Figure 5.3 (Textbook Fig. 7.10, p. 368): Superladder structure used in the extended Schur algorithm for lattice-ladder filtering.*

---

**§5 Schur Recursive Algorithm**

- <a href="#51-goal-of-the-schur-algorithm" style="color: #a0a0a0;">5.1 Goal of the Schur Algorithm</a>
- <a href="#52-gapped-correlation-functions" style="color: #a0a0a0;">5.2 Gapped Correlation Functions</a>
- <a href="#53-algorithmic-interpretation" style="color: #a0a0a0;">5.3 Algorithmic Interpretation</a>
- <a href="#54-schur-algorithm-summary" style="color: #a0a0a0;">5.4 Schur Algorithm Summary</a>
- [5.5 Relationship to Levinson-Durbin](#55-relationship-to-levinson-durbin)

---

## 5.5 Relationship to Levinson-Durbin

Both algorithms solve the same underlying Toeplitz prediction problem. The difference is what they compute along the way.

| Algorithm | Main output | Best suited for |
|----------|-------------|-----------------|
| Levinson-Durbin | Direct coefficients and reflection coefficients | Direct-form AR modeling, LPC coefficient computation |
| Schur | Reflection coefficients directly | Lattice implementations, stability testing, fixed-point systems |

The Schur algorithm has good numerical properties because its internal quantities remain bounded by the signal power under positive-definite autocorrelation assumptions.

---

### Answer to the Guiding Question

**Question.** A lattice prewhitener is designed directly from $r(0)=4$, $r(1)=2$, and $r(2)=1.5$. Find its reflection coefficients, stage error powers, and stability result.

**Answer.**

The first Schur/Levinson reflection coefficient is the normalized negative lag-one correlation:

$$\kappa_1=-\frac{2}{4}=-0.5.$$

The remaining error power is

$$P_1=4(1-0.5^2)=3.$$

After the lag-one component is removed, the residual lag-two numerator is

$$\alpha_2=r(2)+\kappa_1r(1)=1.5-1=0.5.$$

Therefore

$$\kappa_2=-\frac{0.5}{3}=-\frac16\approx-0.1667,$$

and

$$P_2=3\left(1-\frac1{36}\right)=\frac{35}{12}\approx2.9167.$$

The strict lattice stability checks are $|\kappa_1|=0.5<1$ and $|\kappa_2|=0.1667<1$, so both stages pass. **The processor should be loaded with reflection coefficients $-0.5$ and $-0.1667$; the power sequence is $4\to3\to2.9167$, and the resulting model is stable.**

---

# §6 General Levinson Recursion for Toeplitz Equations

---

### Guiding Question

Two adjacent hydrophone samples form the input to an optimum short acoustic-noise canceller. Their covariance and their cross-correlation with the desired clean pressure sample are

$$\mathbf R=\begin{bmatrix}4&2\\2&4\end{bmatrix},\qquad \mathbf d=\begin{bmatrix}3\\1\end{bmatrix},$$

and the desired-sample power is $5$. Find the two optimum FIR weights, the estimate produced when the current input vector is $[2,1]^T$, and the minimum MSE. How does the arithmetic complexity scale if this Toeplitz problem grows to order $p$?

The earlier recursions solve the special prediction right-hand side built from autocorrelations. This acoustic estimator has an arbitrary cross-correlation vector, which requires the general Levinson viewpoint of §6.

---

> 📖 Textbook §7.3 (Order-Recursive Algorithms for Optimum FIR Filters); §7.7 (Triangularization and Inversion of Toeplitz Matrices)

---

**§6 General Levinson Recursion for Toeplitz Equations**

- [6.1 Beyond Linear Prediction](#61-beyond-linear-prediction)
- <a href="#62-the-optimum-nesting-property" style="color: #a0a0a0;">6.2 The Optimum Nesting Property</a>
- <a href="#63-partitioned-matrix-inversion-view" style="color: #a0a0a0;">6.3 Partitioned Matrix Inversion View</a>
- <a href="#64-general-levinson-recursion-idea" style="color: #a0a0a0;">6.4 General Levinson Recursion Idea</a>
- <a href="#65-complexity" style="color: #a0a0a0;">6.5 Complexity</a>

---

## 6.1 Beyond Linear Prediction

The Levinson-Durbin recursion solves the special Toeplitz system

$$\mathbf{R}_p\mathbf{a}_p=-\mathbf{r}_p,$$

where the right-hand side is related to the autocorrelation sequence itself.

In many applications, however, we need to solve a more general Toeplitz system:

$$\boxed{\mathbf{R}_p\mathbf{c}_p=\mathbf{d}_p}$$

where $\mathbf{d}_p$ is an arbitrary cross-correlation vector.

Examples include:

- optimum FIR Wiener filtering,
- ARMA parameter estimation,
- linear equalization,
- smoothing and interpolation,
- lattice-ladder filtering.

---

**§6 General Levinson Recursion for Toeplitz Equations**

- <a href="#61-beyond-linear-prediction" style="color: #a0a0a0;">6.1 Beyond Linear Prediction</a>
- [6.2 The Optimum Nesting Property](#62-the-optimum-nesting-property)
- <a href="#63-partitioned-matrix-inversion-view" style="color: #a0a0a0;">6.3 Partitioned Matrix Inversion View</a>
- <a href="#64-general-levinson-recursion-idea" style="color: #a0a0a0;">6.4 General Levinson Recursion Idea</a>
- <a href="#65-complexity" style="color: #a0a0a0;">6.5 Complexity</a>

---

## 6.2 The Optimum Nesting Property

The key condition behind order-recursive algorithms is the optimum nesting property. When the filter order increases from $m$ to $m+1$, the new correlation matrix contains the old one as a principal submatrix:

$$\mathbf{R}_{m+1}=\begin{bmatrix}
\mathbf{R}_m & \mathbf{q}_m\\
\mathbf{q}_m^H & r(0)
\end{bmatrix}.$$

This nesting makes it possible to express $\mathbf{R}_{m+1}^{-1}$ in terms of $\mathbf{R}_{m}^{-1}$ plus a rank-one correction.

---

**§6 General Levinson Recursion for Toeplitz Equations**

- <a href="#61-beyond-linear-prediction" style="color: #a0a0a0;">6.1 Beyond Linear Prediction</a>
- <a href="#62-the-optimum-nesting-property" style="color: #a0a0a0;">6.2 The Optimum Nesting Property</a>
- [6.3 Partitioned Matrix Inversion View](#63-partitioned-matrix-inversion-view)
- <a href="#64-general-levinson-recursion-idea" style="color: #a0a0a0;">6.4 General Levinson Recursion Idea</a>
- <a href="#65-complexity" style="color: #a0a0a0;">6.5 Complexity</a>

---

## 6.3 Partitioned Matrix Inversion View

For a block matrix

$$\mathbf{R}_{m+1}=\begin{bmatrix}
\mathbf{R}_m & \mathbf{q}_m\\
\mathbf{q}_m^H & \gamma
\end{bmatrix},$$

its inverse can be written using the Schur complement

$$S_m=\gamma-\mathbf{q}_m^H\mathbf{R}_m^{-1}\mathbf{q}_m.$$

The quantity $S_m$ is exactly an error power. In prediction language, it is the part of the new sample that cannot be predicted from the previous $m$ samples.

Thus the matrix inversion lemma and the prediction interpretation are the same idea seen from two angles.

---

**§6 General Levinson Recursion for Toeplitz Equations**

- <a href="#61-beyond-linear-prediction" style="color: #a0a0a0;">6.1 Beyond Linear Prediction</a>
- <a href="#62-the-optimum-nesting-property" style="color: #a0a0a0;">6.2 The Optimum Nesting Property</a>
- <a href="#63-partitioned-matrix-inversion-view" style="color: #a0a0a0;">6.3 Partitioned Matrix Inversion View</a>
- [6.4 General Levinson Recursion Idea](#64-general-levinson-recursion-idea)
- <a href="#65-complexity" style="color: #a0a0a0;">6.5 Complexity</a>

---

## 6.4 General Levinson Recursion Idea

To solve

$$\mathbf{R}_m\mathbf{c}_m=\mathbf{d}_m,$$

recursively, the general Levinson algorithm tracks two types of quantities:

1. prediction vectors associated with the Toeplitz matrix,
2. filter vectors associated with the arbitrary right-hand side $\mathbf{d}_m$.

When the order increases, the prediction recursion provides the correction direction, and the cross-correlation recursion determines how much of that direction must be added to the current optimum filter.

This is why many optimum FIR filtering structures contain both:

- a lattice part, which orthogonalizes the input data,
- a ladder part, which combines orthogonalized components to estimate the desired response.

---

**§6 General Levinson Recursion for Toeplitz Equations**

- <a href="#61-beyond-linear-prediction" style="color: #a0a0a0;">6.1 Beyond Linear Prediction</a>
- <a href="#62-the-optimum-nesting-property" style="color: #a0a0a0;">6.2 The Optimum Nesting Property</a>
- <a href="#63-partitioned-matrix-inversion-view" style="color: #a0a0a0;">6.3 Partitioned Matrix Inversion View</a>
- <a href="#64-general-levinson-recursion-idea" style="color: #a0a0a0;">6.4 General Levinson Recursion Idea</a>
- [6.5 Complexity](#65-complexity)

---

## 6.5 Complexity

| Method | Complexity for order $p$ | Uses Toeplitz structure? |
|--------|--------------------------|--------------------------|
| Generic Gaussian elimination | $O(p^3)$ | No |
| Levinson / Levinson-Durbin | $O(p^2)$ | Yes |
| Schur recursion | $O(p^2)$ | Yes |
| Fast specialized Toeplitz solvers | Often below $O(p^2)$ in special settings | Yes |

The main message is that Toeplitz structure should never be ignored in large prediction/filtering problems.

---

### Answer to the Guiding Question

**Question.** For $\mathbf R=[[4,2],[2,4]]$, $\mathbf d=[3,1]^T$, and desired power $5$, find the optimum two-tap acoustic estimator, its output for $[2,1]^T$, the minimum MSE, and the relevant complexity scaling.

**Answer.**

The optimum weights solve $\mathbf R\mathbf c=\mathbf d$. Since $\det\mathbf R=12$,

$$\mathbf c=\frac1{12}\begin{bmatrix}4&-2\\-2&4\end{bmatrix}
\begin{bmatrix}3\\1\end{bmatrix}
=\begin{bmatrix}5/6\\-1/6\end{bmatrix}
\approx\begin{bmatrix}0.8333\\-0.1667\end{bmatrix}.$$

For the observed input vector,

$$\hat d=\mathbf c^T\begin{bmatrix}2\\1\end{bmatrix}
=\frac56(2)-\frac16(1)=1.5.$$

At the optimum, the minimum MSE is

$$J_{\min}=\sigma_d^2-\mathbf d^T\mathbf c
=5-\left(3\cdot\frac56-\frac16\right)
=\frac83\approx2.667.$$

A generic order-$p$ solve costs $O(p^3)$, whereas exploiting the Toeplitz structure with general Levinson recursion costs $O(p^2)$. **The numerical weights are $(0.8333,-0.1667)$, the current estimate is $1.5$, and the minimum MSE is $2.667$.**

---

# §7 Covariance Algorithm for Linear Prediction

---

### Guiding Question

Only four samples, $x(0),\ldots,x(3)=[1,2,3,4]$, are available from a short active-sonar echo. Fit the first-order prediction-error model $e(n)=x(n)+a_1x(n-1)$ using only indices for which both samples were actually observed. What coefficient minimizes the valid-data squared error, what is that minimum error, and would the corresponding first-order all-pole generator be stable?

The autocorrelation method used earlier implicitly extends a finite record with zeros. Section 7 asks what changes when a short acoustic record is fitted only on its valid interval.

---

> 📖 Textbook §7.1; §9.2.1 (Direct Structures)

---

**§7 Covariance Algorithm for Linear Prediction**

- [7.1 Motivation](#71-motivation)
- <a href="#72-error-criterion" style="color: #a0a0a0;">7.2 Error Criterion</a>
- <a href="#73-why-the-matrix-is-not-toeplitz" style="color: #a0a0a0;">7.3 Why the Matrix Is Not Toeplitz</a>
- <a href="#74-advantages-and-disadvantages" style="color: #a0a0a0;">7.4 Advantages and Disadvantages</a>
- <a href="#75-when-to-use-the-covariance-method" style="color: #a0a0a0;">7.5 When to Use the Covariance Method</a>

---

## 7.1 Motivation

The autocorrelation method assumes samples outside the data record are zero. This is convenient because it preserves Toeplitz structure and guarantees stable all-pole models. But the zero-extension assumption can distort short data records.

The covariance method avoids this by evaluating the prediction error only over indices where all required samples are available.

For a $p$-th order predictor and data $x(0),x(1),\ldots,x(N)$, use

$$n=p,p+1,\ldots,N.$$

Then every term $x(n-k)$ for $k=1,\ldots,p$ lies inside the observed data interval.

---

**§7 Covariance Algorithm for Linear Prediction**

- <a href="#71-motivation" style="color: #a0a0a0;">7.1 Motivation</a>
- [7.2 Error Criterion](#72-error-criterion)
- <a href="#73-why-the-matrix-is-not-toeplitz" style="color: #a0a0a0;">7.3 Why the Matrix Is Not Toeplitz</a>
- <a href="#74-advantages-and-disadvantages" style="color: #a0a0a0;">7.4 Advantages and Disadvantages</a>
- <a href="#75-when-to-use-the-covariance-method" style="color: #a0a0a0;">7.5 When to Use the Covariance Method</a>

---

## 7.2 Error Criterion

The covariance-method criterion is

$$\boxed{E_p=\sum_{n=p}^{N}\left\lvert x(n)+\sum_{k=1}^{p}a_k^{\ast}x(n-k)\right\rvert^2.}$$

This is a deterministic least-squares problem.

Differentiate with respect to the prediction coefficients to obtain the normal equations:

$$\sum_{k=1}^{p}\Phi(i,k)a_k=-\Phi(i,0),\qquad i=1,2,\ldots,p,$$

where

$$\boxed{\Phi(i,k)=\sum_{n=p}^{N}x(n-i)x^{\ast}(n-k).}$$

In matrix form,

$$\boxed{\boldsymbol{\Phi}\mathbf{a}_p=-\boldsymbol{\phi}.}$$

---

**§7 Covariance Algorithm for Linear Prediction**

- <a href="#71-motivation" style="color: #a0a0a0;">7.1 Motivation</a>
- <a href="#72-error-criterion" style="color: #a0a0a0;">7.2 Error Criterion</a>
- [7.3 Why the Matrix Is Not Toeplitz](#73-why-the-matrix-is-not-toeplitz)
- <a href="#74-advantages-and-disadvantages" style="color: #a0a0a0;">7.4 Advantages and Disadvantages</a>
- <a href="#75-when-to-use-the-covariance-method" style="color: #a0a0a0;">7.5 When to Use the Covariance Method</a>

---

## 7.3 Why the Matrix Is Not Toeplitz

The covariance matrix entries depend on both $i$ and $k$:

$$\Phi(i,k)=\sum_{n=p}^{N}x(n-i)x^{\ast}(n-k).$$

Although $x(n-i)$ and $x(n-k)$ differ by lag $i-k$, the summation limits do not shift with the lag. Therefore the result is not simply a function of $i-k$.

So, unlike the autocorrelation method, the covariance method usually produces a non-Toeplitz matrix.

---

**§7 Covariance Algorithm for Linear Prediction**

- <a href="#71-motivation" style="color: #a0a0a0;">7.1 Motivation</a>
- <a href="#72-error-criterion" style="color: #a0a0a0;">7.2 Error Criterion</a>
- <a href="#73-why-the-matrix-is-not-toeplitz" style="color: #a0a0a0;">7.3 Why the Matrix Is Not Toeplitz</a>
- [7.4 Advantages and Disadvantages](#74-advantages-and-disadvantages)
- <a href="#75-when-to-use-the-covariance-method" style="color: #a0a0a0;">7.5 When to Use the Covariance Method</a>

---

## 7.4 Advantages and Disadvantages

| Method | Advantage | Disadvantage |
|--------|-----------|--------------|
| Autocorrelation method | Toeplitz; efficient; guarantees stable AR model | Imposes zero extension / windowing |
| Covariance method | Uses only valid data; often better for short records | Non-Toeplitz; more computation; no automatic stability guarantee |

The covariance method is attractive when the data record is short and boundary distortion is serious. But because it does not guarantee a minimum-phase prediction error filter, one must check stability if the coefficients are used as an all-pole synthesis model.

---

**§7 Covariance Algorithm for Linear Prediction**

- <a href="#71-motivation" style="color: #a0a0a0;">7.1 Motivation</a>
- <a href="#72-error-criterion" style="color: #a0a0a0;">7.2 Error Criterion</a>
- <a href="#73-why-the-matrix-is-not-toeplitz" style="color: #a0a0a0;">7.3 Why the Matrix Is Not Toeplitz</a>
- <a href="#74-advantages-and-disadvantages" style="color: #a0a0a0;">7.4 Advantages and Disadvantages</a>
- [7.5 When to Use the Covariance Method](#75-when-to-use-the-covariance-method)

---

## 7.5 When to Use the Covariance Method

Use the covariance method when:

- the data record is short,
- boundary assumptions are unreliable,
- high-resolution spectrum estimation is desired,
- computational cost is acceptable,
- model stability can be checked or enforced separately.

Use the autocorrelation method when:

- stability is essential,
- fast Toeplitz algorithms are desired,
- the record is long enough that boundary effects are less important,
- LPC-style robust parameter estimation is needed.

---

### Answer to the Guiding Question

**Question.** Fit $e(n)=x(n)+a_1x(n-1)$ to the valid pairs of the four-sample echo $[1,2,3,4]$. Find $a_1$, the minimum error, and the stability of the implied all-pole model.

**Answer.**

The valid indices are $n=1,2,3$. Differentiating

$$E_1=\sum_{n=1}^{3}[x(n)+a_1x(n-1)]^2$$

gives

$$a_1=-\frac{\sum_{n=1}^{3}x(n)x(n-1)}{\sum_{n=1}^{3}x^2(n-1)}
=-\frac{2+6+12}{1+4+9}=-\frac{10}{7}\approx-1.4286.$$

The three residuals are

$$\left[2-\frac{10}{7},\ 3-\frac{20}{7},\ 4-\frac{30}{7}\right]
=\left[\frac47,\frac17,-\frac27\right],$$

so

$$E_{\min}=\frac{16+1+4}{49}=\frac37\approx0.4286.$$

However, $A(z)=1-(10/7)z^{-1}$ has a zero at $z=10/7\approx1.4286$. Its reciprocal all-pole generator therefore has a pole outside the unit circle and is unstable. **The covariance fit is numerically excellent—$a_1=-1.4286$ and error $0.4286$—but it illustrates that the method does not automatically guarantee a stable AR model.**

---

# §8 Forward/Backward Prediction and Lattice Filters

---

### Guiding Question

At one stage of a lattice prewhitener for hull-vibration noise, the incoming forward error is $e_0^f(n)=3$, the delayed backward error is $e_0^b(n-1)=1$, and the real reflection coefficient is $\kappa_1=-0.5$. What forward and backward errors leave the stage? If the incoming error power is $P_0=4$, what outgoing error power is predicted, and does this stage correspond to a stable model?

Direct-form coefficients do not reveal how a modular lattice updates errors in both time directions. Section 8 supplies the coupled forward/backward recursions needed to compute these numbers.

---

> 📖 Textbook §7.5 (Lattice Structures for Optimum FIR Filters and Predictors)

---

**§8 Forward/Backward Prediction and Lattice Filters**

- [8.1 Forward and Backward Prediction Errors](#81-forward-and-backward-prediction-errors)
- <a href="#82-lattice-recursions" style="color: #a0a0a0;">8.2 Lattice Recursions</a>
- <a href="#83-intuition-behind-the-lattice-stage" style="color: #a0a0a0;">8.3 Intuition Behind the Lattice Stage</a>
- <a href="#84-all-pole-lattice-synthesis" style="color: #a0a0a0;">8.4 All-Pole Lattice Synthesis</a>
- <a href="#85-advantages-of-lattice-filters" style="color: #a0a0a0;">8.5 Advantages of Lattice Filters</a>
- <a href="#86-lattice-ladder-structure" style="color: #a0a0a0;">8.6 Lattice-Ladder Structure</a>

---

## 8.1 Forward and Backward Prediction Errors

The forward prediction error estimates the present sample from the past:

$$\boxed{e_m^f(n)=x(n)+\sum_{k=1}^{m}a_k^{(m)\ast}x(n-k).}$$

The backward prediction error estimates the oldest sample from the newer samples:

$$\boxed{e_m^b(n)=x(n-m)+\sum_{k=0}^{m-1}b_k^{(m)\ast}x(n-k).}$$

For a WSS process, the forward and backward predictors are related by conjugate reversal:

$$\boxed{\mathbf{b}_m=\mathbf{J}\mathbf{a}_m^{\ast}.}$$

The forward and backward error powers are equal:

$$\boxed{E\{\lvert e_m^f(n)\rvert^2\}=E\{\lvert e_m^b(n)\rvert^2\}=P_m.}$$

---

**§8 Forward/Backward Prediction and Lattice Filters**

- <a href="#81-forward-and-backward-prediction-errors" style="color: #a0a0a0;">8.1 Forward and Backward Prediction Errors</a>
- [8.2 Lattice Recursions](#82-lattice-recursions)
- <a href="#83-intuition-behind-the-lattice-stage" style="color: #a0a0a0;">8.3 Intuition Behind the Lattice Stage</a>
- <a href="#84-all-pole-lattice-synthesis" style="color: #a0a0a0;">8.4 All-Pole Lattice Synthesis</a>
- <a href="#85-advantages-of-lattice-filters" style="color: #a0a0a0;">8.5 Advantages of Lattice Filters</a>
- <a href="#86-lattice-ladder-structure" style="color: #a0a0a0;">8.6 Lattice-Ladder Structure</a>

---

## 8.2 Lattice Recursions

The lattice structure is based on two coupled recursions:

$$\boxed{e_m^f(n)=e_{m-1}^f(n)+\kappa_m^{\ast}e_{m-1}^b(n-1)}$$

$$\boxed{e_m^b(n)=e_{m-1}^b(n-1)+\kappa_m e_{m-1}^f(n)}.$$

The initial condition is

$$e_0^f(n)=e_0^b(n)=x(n).$$

Each lattice stage removes the residual correlation between the current forward and backward errors.

> ![Figure 8.1](./CourseADSP2026/Fig/Chapter_3/fig_8_1_textbook_fig_7_5_p357.png)
>
> *Figure 8.1 (Textbook Fig. 7.5, p. 357): All-zero lattice structure for forward and backward prediction error filters.*

---

**§8 Forward/Backward Prediction and Lattice Filters**

- <a href="#81-forward-and-backward-prediction-errors" style="color: #a0a0a0;">8.1 Forward and Backward Prediction Errors</a>
- <a href="#82-lattice-recursions" style="color: #a0a0a0;">8.2 Lattice Recursions</a>
- [8.3 Intuition Behind the Lattice Stage](#83-intuition-behind-the-lattice-stage)
- <a href="#84-all-pole-lattice-synthesis" style="color: #a0a0a0;">8.4 All-Pole Lattice Synthesis</a>
- <a href="#85-advantages-of-lattice-filters" style="color: #a0a0a0;">8.5 Advantages of Lattice Filters</a>
- <a href="#86-lattice-ladder-structure" style="color: #a0a0a0;">8.6 Lattice-Ladder Structure</a>

---

## 8.3 Intuition Behind the Lattice Stage

At stage $m-1$, suppose we already removed all predictable structure up to lag $m-1$. The remaining correlation between $e_{m-1}^f(n)$ and $e_{m-1}^b(n-1)$ represents the new information at lag $m$.

The reflection coefficient $\kappa_m$ is chosen to remove this remaining correlation. After the update, the new errors satisfy a stronger orthogonality condition.

Thus each lattice stage performs a local decorrelation operation.

---

**§8 Forward/Backward Prediction and Lattice Filters**

- <a href="#81-forward-and-backward-prediction-errors" style="color: #a0a0a0;">8.1 Forward and Backward Prediction Errors</a>
- <a href="#82-lattice-recursions" style="color: #a0a0a0;">8.2 Lattice Recursions</a>
- <a href="#83-intuition-behind-the-lattice-stage" style="color: #a0a0a0;">8.3 Intuition Behind the Lattice Stage</a>
- [8.4 All-Pole Lattice Synthesis](#84-all-pole-lattice-synthesis)
- <a href="#85-advantages-of-lattice-filters" style="color: #a0a0a0;">8.5 Advantages of Lattice Filters</a>
- <a href="#86-lattice-ladder-structure" style="color: #a0a0a0;">8.6 Lattice-Ladder Structure</a>

---

## 8.4 All-Pole Lattice Synthesis

The prediction error filter maps $x(n)$ to the residual $e_p^f(n)$. The inverse system maps the residual back to $x(n)$:

$$x(n)=\frac{1}{A_p(z)}e_p^f(n).$$

This inverse can also be implemented in lattice form.

> ![Figure 8.2](./CourseADSP2026/Fig/Chapter_3/fig_8_2_textbook_fig_7_6_p359.png)
>
> *Figure 8.2 (Textbook Fig. 7.6, p. 359): All-pole lattice structure for recovering the input signal from the forward prediction error.*

---

**§8 Forward/Backward Prediction and Lattice Filters**

- <a href="#81-forward-and-backward-prediction-errors" style="color: #a0a0a0;">8.1 Forward and Backward Prediction Errors</a>
- <a href="#82-lattice-recursions" style="color: #a0a0a0;">8.2 Lattice Recursions</a>
- <a href="#83-intuition-behind-the-lattice-stage" style="color: #a0a0a0;">8.3 Intuition Behind the Lattice Stage</a>
- <a href="#84-all-pole-lattice-synthesis" style="color: #a0a0a0;">8.4 All-Pole Lattice Synthesis</a>
- [8.5 Advantages of Lattice Filters](#85-advantages-of-lattice-filters)
- <a href="#86-lattice-ladder-structure" style="color: #a0a0a0;">8.6 Lattice-Ladder Structure</a>

---

## 8.5 Advantages of Lattice Filters

Lattice filters have several important advantages:

1. **Modularity.** Increasing the model order adds one stage without redesigning the whole structure.
2. **Stability monitoring.** Stability is checked by $\lvert\kappa_m\rvert<1$.
3. **Numerical robustness.** Reflection coefficients are bounded for stable models.
4. **Order-recursive implementation.** The same structure naturally supports model-order selection.
5. **Reduced sensitivity to coefficient quantization.** Quantization of reflection coefficients is often safer than quantization of direct-form coefficients.

---

**§8 Forward/Backward Prediction and Lattice Filters**

- <a href="#81-forward-and-backward-prediction-errors" style="color: #a0a0a0;">8.1 Forward and Backward Prediction Errors</a>
- <a href="#82-lattice-recursions" style="color: #a0a0a0;">8.2 Lattice Recursions</a>
- <a href="#83-intuition-behind-the-lattice-stage" style="color: #a0a0a0;">8.3 Intuition Behind the Lattice Stage</a>
- <a href="#84-all-pole-lattice-synthesis" style="color: #a0a0a0;">8.4 All-Pole Lattice Synthesis</a>
- <a href="#85-advantages-of-lattice-filters" style="color: #a0a0a0;">8.5 Advantages of Lattice Filters</a>
- [8.6 Lattice-Ladder Structure](#86-lattice-ladder-structure)

---

## 8.6 Lattice-Ladder Structure

For general optimum FIR filtering, prediction errors alone are not enough. We also need to estimate a desired response $y(n)$. A lattice-ladder filter does this by:

- using the lattice section to generate orthogonalized backward prediction errors,
- using the ladder section to combine those errors to estimate $y(n)$.

The lattice part depends only on the input autocorrelation. The ladder coefficients depend on the cross-correlation between the input and the desired response.

This separation is conceptually powerful:

| Part | Role |
|------|------|
| Lattice | Decorrelates / orthogonalizes the input |
| Ladder | Projects the desired response onto the orthogonalized components |

---

### Answer to the Guiding Question

**Question.** A lattice stage receives $e_0^f(n)=3$, $e_0^b(n-1)=1$, $\kappa_1=-0.5$, and $P_0=4$. Find both output errors, the output power, and the stability result.

**Answer.**

For a real lattice stage,

$$e_1^f(n)=e_0^f(n)+\kappa_1e_0^b(n-1)
=3-0.5(1)=2.5,$$

$$e_1^b(n)=e_0^b(n-1)+\kappa_1e_0^f(n)
=1-0.5(3)=-0.5.$$

The stage error-power recursion gives

$$P_1=P_0(1-|\kappa_1|^2)=4(1-0.25)=3.$$

Since $|\kappa_1|=0.5<1$, the stage satisfies the strict stability condition. **Numerically, the outgoing forward error is $2.5$, the outgoing backward error is $-0.5$, the error power falls from $4$ to $3$, and the model is stable.** The two different instantaneous errors are not themselves power estimates; the power value is the ensemble or time-averaged recursion result.

---

# §9 Lattice Modeling and the Burg Algorithm

---

### Guiding Question

A very short hydrophone record gives paired zero-order forward and delayed-backward error vectors $\mathbf f=[2,1]$ and $\mathbf b=[1,2]$ at the first Burg stage. Find the Burg reflection coefficient, update both error vectors, and calculate the combined squared-error energy before and after the update. If the initial variance estimate is $5$, what is the updated variance, and is the resulting first-order model stable?

Section 8 explains a lattice when its coefficient is known. Section 9 shows how Burg's short-record criterion estimates that coefficient directly from forward and backward acoustic residuals.

---

> 📖 Textbook §9.2.3 (Maximum Entropy Method); §9.2 lattice AP estimation

---

**§9 Lattice Modeling and the Burg Algorithm**

- [9.1 Why Burg's Algorithm Was Introduced](#91-why-burgs-algorithm-was-introduced)
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.1 Why Burg's Algorithm Was Introduced

The autocorrelation method is stable but uses artificial zero extension. The covariance method uses the data more directly but does not guarantee stability.

Burg's algorithm attempts to combine useful features of both:

- it avoids explicit autocorrelation estimation,
- it uses both forward and backward prediction errors,
- it estimates reflection coefficients stage by stage,
- it guarantees a stable all-pole model because $\lvert\kappa_m\rvert<1$ under its update.

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- [9.2 Forward and Backward Error Criterion](#92-forward-and-backward-error-criterion)
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.2 Forward and Backward Error Criterion

At each order $m$, Burg's method minimizes the sum of forward and backward prediction error energies:

$$\boxed{E_m^{fb}=\sum_{n=N_i}^{N_f}\left(\lvert e_m^f(n)\rvert^2+\lvert e_m^b(n)\rvert^2\right).}$$

Using the lattice recursions,

$$e_m^f(n)=e_{m-1}^f(n)+\kappa_m^{\ast}e_{m-1}^b(n-1),$$

$$e_m^b(n)=e_{m-1}^b(n-1)+\kappa_m e_{m-1}^f(n),$$

only one new parameter $\kappa_m$ is optimized at stage $m$.

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- [9.3 Burg Reflection Coefficient Update](#93-burg-reflection-coefficient-update)
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.3 Burg Reflection Coefficient Update

Minimizing $E_m^{fb}$ with respect to $\kappa_m^{\ast}$ gives

$$\boxed{\kappa_m=-\frac{2\sum_n e_{m-1}^f(n)[e_{m-1}^b(n-1)]^{\ast}}{\sum_n \lvert e_{m-1}^f(n)\rvert^2+\sum_n \lvert e_{m-1}^b(n-1)\rvert^2}.}$$

For real-valued signals this becomes

$$\boxed{\kappa_m=-\frac{2\sum_n e_{m-1}^f(n)e_{m-1}^b(n-1)}{\sum_n [e_{m-1}^f(n)]^2+\sum_n [e_{m-1}^b(n-1)]^2}.}$$

By the Cauchy-Schwarz inequality, this estimate satisfies

$$\lvert\kappa_m\rvert\le 1.$$

For non-degenerate data, $\lvert\kappa_m\rvert<1$, so the resulting all-pole model is stable.

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- [9.4 Burg Algorithm Steps](#94-burg-algorithm-steps)
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.4 Burg Algorithm Steps

| Step | Operation |
|------|-----------|
| 1 | Initialize $e_0^f(n)=e_0^b(n)=x(n)$ |
| 2 | For $m=1,2,\ldots,p$, compute $\kappa_m$ from the forward/backward error formula |
| 3 | Update $e_m^f(n)$ and $e_m^b(n)$ using the lattice recursions |
| 4 | Update the residual variance, often by $\hat\sigma_m^2\approx\hat\sigma_{m-1}^2(1-\lvert\kappa_m\rvert^2)$ |
| 5 | After the final stage, convert reflection coefficients to direct AR coefficients if needed |

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- [9.5 Burg, Forward Covariance, and Backward Covariance](#95-burg-forward-covariance-and-backward-covariance)
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.5 Burg, Forward Covariance, and Backward Covariance

There are several related stage-wise lattice methods.

### Forward Lattice Covariance Method

Minimize only

$$\sum_n \lvert e_m^f(n)\rvert^2.$$

This gives a reflection coefficient based on forward error alone. It does not necessarily guarantee stability.

### Backward Lattice Covariance Method

Minimize only

$$\sum_n \lvert e_m^b(n)\rvert^2.$$

This is the backward counterpart and also does not necessarily guarantee stability.

### Burg Method

Minimize the combined forward/backward criterion:

$$\sum_n \left(\lvert e_m^f(n)\rvert^2+\lvert e_m^b(n)\rvert^2\right).$$

The combined criterion leads to a stable lattice model and is usually preferred for AR spectral estimation from short records.

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- [9.6 Itakura-Saito / Geometric Mean Variant](#96-itakura-saito--geometric-mean-variant)
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.6 Itakura-Saito / Geometric Mean Variant

Another estimate uses a geometric-mean normalization. In simplified notation,

$$\kappa_m^{IS}\propto -\frac{\beta_m^{fb}}{\sqrt{E_m^fE_m^b}}.$$

This can be viewed as replacing the arithmetic normalization in Burg's method with a geometric one. Both methods are designed to keep reflection coefficients within the unit disk under appropriate conditions.

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- [9.7 Maximum Entropy Interpretation](#97-maximum-entropy-interpretation)
- <a href="#98-strengths-and-weaknesses-of-burgs-algorithm" style="color: #a0a0a0;">9.8 Strengths and Weaknesses of Burg's Algorithm</a>

---

## 9.7 Maximum Entropy Interpretation

Burg's method is closely connected to the maximum entropy principle.

Suppose we know the first $p+1$ autocorrelation values

$$r(0),r(1),\ldots,r(p).$$

Among all regular Gaussian processes that match these known autocorrelation values, the AR($p$) process has maximum entropy.

The reason is that the entropy can be expressed in terms of reflection coefficients. For known correlations up to order $p$, the maximum entropy extension chooses

$$\kappa_m=0,\qquad m>p.$$

Thus no additional partial correlation is assumed beyond what is supported by the known data. This leads exactly to an AR($p$) all-pole model.

---

**§9 Lattice Modeling and the Burg Algorithm**

- <a href="#91-why-burgs-algorithm-was-introduced" style="color: #a0a0a0;">9.1 Why Burg's Algorithm Was Introduced</a>
- <a href="#92-forward-and-backward-error-criterion" style="color: #a0a0a0;">9.2 Forward and Backward Error Criterion</a>
- <a href="#93-burg-reflection-coefficient-update" style="color: #a0a0a0;">9.3 Burg Reflection Coefficient Update</a>
- <a href="#94-burg-algorithm-steps" style="color: #a0a0a0;">9.4 Burg Algorithm Steps</a>
- <a href="#95-burg-forward-covariance-and-backward-covariance" style="color: #a0a0a0;">9.5 Burg, Forward Covariance, and Backward Covariance</a>
- <a href="#96-itakura-saito--geometric-mean-variant" style="color: #a0a0a0;">9.6 Itakura-Saito / Geometric Mean Variant</a>
- <a href="#97-maximum-entropy-interpretation" style="color: #a0a0a0;">9.7 Maximum Entropy Interpretation</a>
- [9.8 Strengths and Weaknesses of Burg's Algorithm](#98-strengths-and-weaknesses-of-burgs-algorithm)

---

## 9.8 Strengths and Weaknesses of Burg's Algorithm

| Aspect | Burg Algorithm |
|--------|----------------|
| Main strength | Stable all-pole model without explicit autocorrelation estimation |
| Data usage | Uses both forward and backward errors |
| Resolution | Often high for short records |
| Implementation | Naturally lattice-based |
| Weakness | Can produce spectral line splitting |
| Weakness | Frequency estimates may be biased, especially at low SNR or with model mismatch |

> ![Figure 9.1](./CourseADSP2026/Fig/Chapter_3/fig_9_1_textbook_fig_9_9_p450.png)
>
> *Figure 9.1 (Textbook Fig. 9.9, p. 450): Data segment from an AR(4) process with autocorrelation, partial autocorrelation, and periodogram. The partial autocorrelation plot is useful for understanding AR order and lattice coefficients.*

> ![Figure 9.2](./CourseADSP2026/Fig/Chapter_3/fig_9_2_textbook_fig_9_10_p451.png)
>
> *Figure 9.2 (Textbook Fig. 9.10, p. 451): Comparison of theoretical and estimated AR spectra under different windowing assumptions.*

> ![Figure 9.3](./CourseADSP2026/Fig/Chapter_3/fig_9_4_textbook_fig_9_14_p463.png)
>
> *Figure 9.3 (Textbook Fig. 9.14, p. 463): Monte Carlo comparison of all-pole PSD estimation techniques using short data records. This is useful when discussing autocorrelation, covariance, modified covariance, and Burg estimates.*

---

### Answer to the Guiding Question

**Question.** At a Burg stage, $\mathbf f=[2,1]$ and $\mathbf b=[1,2]$. Find the reflection coefficient, updated errors, energy reduction, variance update, and stability.

**Answer.**

For real data, Burg's update is

$$\kappa=-\frac{2\mathbf f^T\mathbf b}{\|\mathbf f\|^2+\|\mathbf b\|^2}
=-\frac{2(2\cdot1+1\cdot2)}{(4+1)+(1+4)}=-0.8.$$

Using $\mathbf f_1=\mathbf f+\kappa\mathbf b$ and $\mathbf b_1=\mathbf b+\kappa\mathbf f$,

$$\mathbf f_1=[1.2,-0.6],\qquad \mathbf b_1=[-0.6,1.2].$$

The combined energy before the update is $5+5=10$. Afterward it is

$$1.2^2+(-0.6)^2+(-0.6)^2+1.2^2=3.6,$$

a reduction of $6.4$, or $64\%$. The variance recursion gives

$$P_1=5(1-0.8^2)=1.8.$$

Because $|\kappa|=0.8<1$, the first-order all-pole model is stable. **Numerically, $\kappa=-0.8$, the updated error vectors are $[1.2,-0.6]$ and $[-0.6,1.2]$, combined energy is $3.6$, and the variance estimate is $1.8$.**

---

# §10 Modified Covariance Algorithm

---

### Guiding Question

For a short underwater machinery-noise record, the final second-order forward and backward covariance accumulations are

$$\mathbf X_f^H\mathbf X_f=\begin{bmatrix}5&1\\1&3\end{bmatrix},\quad
\mathbf X_b^H\mathbf X_b=\begin{bmatrix}3&1\\1&5\end{bmatrix},\quad
\mathbf g=\begin{bmatrix}4\\2\end{bmatrix}.$$

The combined constant error-energy term is $C=10$. What globally optimum two-coefficient modified-covariance model results, what is its minimized combined error energy, and what are the two poles of its all-pole realization?

Burg's method makes a sequence of local lattice decisions. Section 10 instead optimizes all final-order coefficients together using both forward and backward errors.

---

> 📖 Textbook §9.2.1 (Modified Covariance Method); §7.3.2 (Lattice-Ladder Structure)

---

**§10 Modified Covariance Algorithm**

- [10.1 Motivation](#101-motivation)
- <a href="#102-difference-between-burg-and-modified-covariance" style="color: #a0a0a0;">10.2 Difference Between Burg and Modified Covariance</a>
- <a href="#103-modified-covariance-normal-equations" style="color: #a0a0a0;">10.3 Modified Covariance Normal Equations</a>
- <a href="#104-advantages" style="color: #a0a0a0;">10.4 Advantages</a>
- <a href="#105-disadvantages" style="color: #a0a0a0;">10.5 Disadvantages</a>
- <a href="#106-comparison-of-main-linear-prediction-estimation-methods" style="color: #a0a0a0;">10.6 Comparison of Main Linear Prediction Estimation Methods</a>

---

## 10.1 Motivation

The covariance method minimizes only the forward prediction error:

$$\sum_n \lvert e_p^f(n)\rvert^2.$$

But for finite data, forward and backward errors are not statistically identical. The modified covariance method uses both errors at the final order $p$:

$$\boxed{E_p^{fb}=\sum_{n=p}^{N}\left(\lvert e_p^f(n)\rvert^2+\lvert e_p^b(n)\rvert^2\right).}$$

This is similar in spirit to Burg's method, but the optimization is different.

---

**§10 Modified Covariance Algorithm**

- <a href="#101-motivation" style="color: #a0a0a0;">10.1 Motivation</a>
- [10.2 Difference Between Burg and Modified Covariance](#102-difference-between-burg-and-modified-covariance)
- <a href="#103-modified-covariance-normal-equations" style="color: #a0a0a0;">10.3 Modified Covariance Normal Equations</a>
- <a href="#104-advantages" style="color: #a0a0a0;">10.4 Advantages</a>
- <a href="#105-disadvantages" style="color: #a0a0a0;">10.5 Disadvantages</a>
- <a href="#106-comparison-of-main-linear-prediction-estimation-methods" style="color: #a0a0a0;">10.6 Comparison of Main Linear Prediction Estimation Methods</a>

---

## 10.2 Difference Between Burg and Modified Covariance

The distinction is crucial.

| Method | Optimization style | Parameter update |
|--------|--------------------|------------------|
| Burg | Sequential local minimization | One reflection coefficient at a time |
| Modified covariance | Global minimization at fixed order $p$ | Solve one full set of coefficients |

Burg's method is greedy: once it chooses $\kappa_1$, it keeps that decision while choosing $\kappa_2$, and so on.

The modified covariance method optimizes the whole $p$-th order coefficient vector at once using both forward and backward errors.

---

**§10 Modified Covariance Algorithm**

- <a href="#101-motivation" style="color: #a0a0a0;">10.1 Motivation</a>
- <a href="#102-difference-between-burg-and-modified-covariance" style="color: #a0a0a0;">10.2 Difference Between Burg and Modified Covariance</a>
- [10.3 Modified Covariance Normal Equations](#103-modified-covariance-normal-equations)
- <a href="#104-advantages" style="color: #a0a0a0;">10.4 Advantages</a>
- <a href="#105-disadvantages" style="color: #a0a0a0;">10.5 Disadvantages</a>
- <a href="#106-comparison-of-main-linear-prediction-estimation-methods" style="color: #a0a0a0;">10.6 Comparison of Main Linear Prediction Estimation Methods</a>

---

## 10.3 Modified Covariance Normal Equations

Let the forward error be

$$e_p^f(n)=x(n)+\sum_{k=1}^{p}a_k^{\ast}x(n-k).$$

For a WSS process, the backward error uses the conjugate-reversed coefficient structure. In finite-data least squares, the combined forward/backward criterion yields normal equations of the form

$$\boxed{(\mathbf{X}_f^H\mathbf{X}_f+\mathbf{X}_b^H\mathbf{X}_b)\mathbf{a}_p=-\mathbf{g}.}$$

Equivalently, using covariance sums,

$$\sum_{k=1}^{p}\left[\Phi_f(i,k)+\Phi_b(i,k)\right]a_k=-\left[\phi_f(i)+\phi_b(i)\right].$$

The precise matrix entries depend on how the data vectors are arranged, but the essential point is simple:

> The normal matrix contains both forward and backward covariance information, so it is generally not Toeplitz.

---

**§10 Modified Covariance Algorithm**

- <a href="#101-motivation" style="color: #a0a0a0;">10.1 Motivation</a>
- <a href="#102-difference-between-burg-and-modified-covariance" style="color: #a0a0a0;">10.2 Difference Between Burg and Modified Covariance</a>
- <a href="#103-modified-covariance-normal-equations" style="color: #a0a0a0;">10.3 Modified Covariance Normal Equations</a>
- [10.4 Advantages](#104-advantages)
- <a href="#105-disadvantages" style="color: #a0a0a0;">10.5 Disadvantages</a>
- <a href="#106-comparison-of-main-linear-prediction-estimation-methods" style="color: #a0a0a0;">10.6 Comparison of Main Linear Prediction Estimation Methods</a>

---

## 10.4 Advantages

The modified covariance method often gives high-resolution spectral estimates. It is especially useful when:

- the data record is short,
- spectral peaks are close together,
- boundary effects from zero extension are unacceptable,
- a full least-squares solution is computationally feasible.

Compared with the ordinary covariance method, the modified covariance method uses more error information and can reduce variance and spectral peak displacement.

---

**§10 Modified Covariance Algorithm**

- <a href="#101-motivation" style="color: #a0a0a0;">10.1 Motivation</a>
- <a href="#102-difference-between-burg-and-modified-covariance" style="color: #a0a0a0;">10.2 Difference Between Burg and Modified Covariance</a>
- <a href="#103-modified-covariance-normal-equations" style="color: #a0a0a0;">10.3 Modified Covariance Normal Equations</a>
- <a href="#104-advantages" style="color: #a0a0a0;">10.4 Advantages</a>
- [10.5 Disadvantages](#105-disadvantages)
- <a href="#106-comparison-of-main-linear-prediction-estimation-methods" style="color: #a0a0a0;">10.6 Comparison of Main Linear Prediction Estimation Methods</a>

---

## 10.5 Disadvantages

The modified covariance method is more expensive than the autocorrelation method because the normal matrix is not Toeplitz. It also does not automatically provide the simple stability guarantee of the autocorrelation method.

Practical implementations often use specialized algorithms, such as Marple-type algorithms, to solve the modified covariance equations efficiently.

---

**§10 Modified Covariance Algorithm**

- <a href="#101-motivation" style="color: #a0a0a0;">10.1 Motivation</a>
- <a href="#102-difference-between-burg-and-modified-covariance" style="color: #a0a0a0;">10.2 Difference Between Burg and Modified Covariance</a>
- <a href="#103-modified-covariance-normal-equations" style="color: #a0a0a0;">10.3 Modified Covariance Normal Equations</a>
- <a href="#104-advantages" style="color: #a0a0a0;">10.4 Advantages</a>
- <a href="#105-disadvantages" style="color: #a0a0a0;">10.5 Disadvantages</a>
- [10.6 Comparison of Main Linear Prediction Estimation Methods](#106-comparison-of-main-linear-prediction-estimation-methods)

---

## 10.6 Comparison of Main Linear Prediction Estimation Methods

| Method | Error criterion | Toeplitz? | Stable AR guaranteed? | Typical use |
|--------|----------------|-----------|-----------------------|-------------|
| Autocorrelation | Forward error with zero extension | Yes | Yes | Robust LPC, stable AR modeling |
| Covariance | Forward error on valid data interval | No | No | Short-record high-resolution modeling |
| Modified covariance | Forward + backward error at final order | No | Not automatic | High-resolution AR spectrum estimation |
| Burg | Sequential forward + backward lattice error | No explicit matrix | Yes | Stable short-record AR spectrum estimation |

---

### Answer to the Guiding Question

**Question.** Given the forward/backward covariance accumulations, $\mathbf g=[4,2]^T$, and $C=10$, find the globally optimum second-order modified-covariance model, minimized energy, and poles.

**Answer.**

The combined normal matrix is

$$\mathbf M=\mathbf X_f^H\mathbf X_f+\mathbf X_b^H\mathbf X_b
=\begin{bmatrix}8&2\\2&8\end{bmatrix}.$$

The optimum satisfies $\mathbf M\mathbf a=-\mathbf g$, so

$$\mathbf a=-\frac1{60}\begin{bmatrix}8&-2\\-2&8\end{bmatrix}
\begin{bmatrix}4\\2\end{bmatrix}
=\begin{bmatrix}-7/15\\-2/15\end{bmatrix}
\approx\begin{bmatrix}-0.4667\\-0.1333\end{bmatrix}.$$

For $J=C+2\mathbf g^T\mathbf a+\mathbf a^T\mathbf M\mathbf a$, the normal equation reduces the optimum value to

$$J_{\min}=C+\mathbf g^T\mathbf a
=10-\frac{32}{15}=\frac{118}{15}\approx7.867.$$

The all-pole denominator is $A(z)=1-(7/15)z^{-1}-(2/15)z^{-2}$. Its zeros, and hence the synthesis poles, are

$$z=\frac23\approx0.6667,\qquad z=-\frac15=-0.2.$$

**Numerically, the coefficient vector is $(-0.4667,-0.1333)$, the minimized combined error is $7.867$, and both poles lie inside the unit circle.**

---

# §11 Application Example: Linear Prediction in Speech Coding

---

### Guiding Question

An air-acoustic microphone analyzes a short, approximately stationary voiced segment for speaker-state sensing. Its measured power and lag-one autocorrelation are $r(0)=100$ and $r(1)=80$. For a first-order LPC analysis, find the prediction weight, prediction-error-filter coefficient, residual power and residual RMS. Also calculate the modeled spectral-envelope ratio between $\omega=0$ and $\omega=\pi$. How much smaller is the residual RMS than the original signal RMS?

Sections 1–10 develop prediction algorithms abstractly. Section 11 connects their coefficients, residual, and all-pole envelope to a concrete air-acoustic voice analysis.

---

> 📖 Textbook §9.4.2 (Speech Modeling); §1.4 adaptive filtering applications

---

**§11 Application Example: Linear Prediction in Speech Coding**

- [11.1 Why Speech Is Predictable](#111-why-speech-is-predictable)
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## 11.1 Why Speech Is Predictable

Speech is highly structured. Over short intervals, typically 10–30 ms, the vocal tract shape is approximately constant. During such a short-time frame, speech can be modeled as the output of an all-pole filter:

$$s(n)=\frac{G}{A_p(z)}u(n),$$

where:

- $u(n)$ is the excitation,
- $A_p(z)$ models the vocal-tract spectral envelope,
- $G$ is a gain factor.

Voiced sounds have quasi-periodic excitation due to vocal-fold vibration. Unvoiced sounds have noise-like excitation. In both cases, the vocal tract acts like a resonant filter.

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- [11.2 LPC Model](#112-lpc-model)
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## 11.2 LPC Model

Linear predictive coding estimates $A_p(z)$ from each short speech frame. The residual is

$$e(n)=A_p(z)s(n).$$

If the model is good, the residual contains mostly excitation information, while $A_p(z)$ contains the slowly varying vocal-tract envelope.

This decomposition is the foundation of LPC speech coding:

| Component | Meaning |
|-----------|---------|
| Prediction coefficients | Vocal-tract spectral envelope |
| Residual energy / gain | Excitation strength |
| Pitch period | Long-term periodicity for voiced speech |
| Voicing decision | Whether excitation is pulse-like or noise-like |

> ![Figure 11.1](./CourseADSP2026/Fig/Chapter_3/fig_11_1_textbook_fig_9_17_p465.png)
>
> *Figure 11.1 (Textbook Fig. 9.17, p. 465): Block diagram of an all-pole modeling processor for speech coding and recognition.*

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- [11.3 Three Coding Paradigms](#113-three-coding-paradigms)
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## 11.3 Three Coding Paradigms

### Waveform Coding

Waveform coding attempts to reproduce the waveform itself. Examples include PCM, DPCM, and ADPCM.

Linear prediction helps waveform coding by reducing sample-to-sample redundancy. Instead of coding $x(n)$ directly, we code the prediction error:

$$e(n)=x(n)-\hat{x}(n).$$

If the prediction is good, $e(n)$ has smaller variance and requires fewer bits.

### Parametric Coding

Parametric coding transmits model parameters rather than waveform samples. In LPC vocoders, the encoder transmits:

- prediction coefficients or reflection coefficients,
- gain,
- pitch period,
- voiced/unvoiced decision.

The decoder reconstructs speech by exciting the all-pole synthesis filter.

### Hybrid Coding

Hybrid coding combines waveform accuracy with model-based compression. Code-excited linear prediction (CELP) is the classic example conceptually, although detailed CELP standards and implementation details are beyond the scope of this lecture.

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- [11.4 Why Reflection Coefficients Are Useful in Speech](#114-why-reflection-coefficients-are-useful-in-speech)
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## 11.4 Why Reflection Coefficients Are Useful in Speech

Speech coders often transform LPC coefficients into more robust parameter sets before quantization. Reflection coefficients are useful because stability corresponds to

$$\lvert\kappa_m\rvert<1.$$

If quantization keeps all reflection coefficients inside the unit interval, then the decoded synthesis filter remains stable.

This is safer than directly quantizing the polynomial coefficients $a_k$, where small coefficient errors can destabilize the all-pole filter.

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- [11.5 Speech Spectrum and All-Pole Envelope](#115-speech-spectrum-and-all-pole-envelope)
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## 11.5 Speech Spectrum and All-Pole Envelope

The all-pole model captures the broad spectral envelope of speech, especially formants. It does not need to reproduce every fine spectral harmonic. This distinction is important:

- harmonics mainly come from the excitation,
- formant envelope mainly comes from the vocal tract filter.

Thus LPC separates source and filter in a computationally efficient way.

> ![Figure 11.2](./CourseADSP2026/Fig/Chapter_3/fig_9_5_textbook_fig_9_15_p464.png)
>
> *Figure 11.2 (Textbook Fig. 9.15, p. 464): Nonparametric PSD estimation using linear prediction prewhitening. The same idea appears in speech processing: remove predictable spectral envelope, process residual, then recolor if needed.*

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- [Chapter 3 Summary](#chapter-3-summary)
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## Chapter 3 Summary

| Topic | Key Idea | Why It Matters |
|------|----------|----------------|
| Linear prediction | Estimate $x(n)$ from past samples | Removes redundancy; foundation of LPC and AR modeling |
| Prediction error filter | $A_p(z)x(n)=e_p(n)$ | Converts modeling into filtering |
| AR equivalence | For AR($p$), prediction error is white noise | PEF is the whitening filter |
| Autocorrelation method | Uses zero-extended autocorrelation estimates | Toeplitz and stable, but can suffer boundary bias |
| Covariance method | Uses only valid data intervals | Better for short records, but non-Toeplitz and not automatically stable |
| Levinson-Durbin | Order-recursive solution of Toeplitz prediction equations | Reduces complexity from $O(p^3)$ to $O(p^2)$ |
| Reflection coefficients | Stage-wise partial correlations | Stability and lattice implementation |
| Schur algorithm | Computes reflection coefficients directly | Useful for lattice filters and fixed-point implementation |
| Lattice filters | Recursively update forward/backward errors | Modular, robust, stable parameterization |
| Burg algorithm | Sequentially minimizes forward/backward errors | Stable high-resolution AR modeling without explicit autocorrelation |
| Modified covariance | Globally minimizes forward/backward errors | High-resolution estimates, higher computation |
| Speech LPC | All-pole short-time speech model | Efficient coding and spectral-envelope representation |

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- [Figure Source Checklist](#figure-source-checklist)
- <a href="#suggested-teaching-flow" style="color: #a0a0a0;">Suggested Teaching Flow</a>

---

## Figure Source Checklist

All figures displayed in this lecture are rendered from the uploaded textbook PDF, *Statistical and Adaptive Signal Processing* by Manolakis, Ingle, and Kogon.

| Lecture Figure | Textbook Figure | Textbook Page | Role in Lecture |
|----------------|-----------------|---------------|-----------------|
| Figure 1.1 | Fig. 6.16 | p. 283 | Linear signal estimation, FLP, BLP |
| Figure 2.1 | Fig. 9.12 | p. 452 | All-pole spectral envelope |
| Figure 3.1 | Fig. 7.1 | p. 337 | Orthogonal order-recursive structure |
| Figure 3.2 | Fig. 7.2 | p. 341 | Gram-Schmidt interpretation |
| Figure 4.1 | Fig. 7.4 | p. 356 | Direct-form prediction errors |
| Figure 4.2 | Fig. 7.7 | p. 361 | Equivalent representations |
| Figure 5.1 | Fig. 7.8 | p. 366 | Schur tree decomposition |
| Figure 5.2 | Fig. 7.9 | p. 367 | Schur superlattice |
| Figure 5.3 | Fig. 7.10 | p. 368 | Schur superladder |
| Figure 8.1 | Fig. 7.5 | p. 357 | All-zero lattice |
| Figure 8.2 | Fig. 7.6 | p. 359 | All-pole lattice |
| Figure 9.1 | Fig. 9.9 | p. 450 | AR data and PACF |
| Figure 9.2 | Fig. 9.10 | p. 451 | Windowing effects in AR spectra |
| Figure 9.3 | Fig. 9.14 | p. 463 | Comparison of AP PSD estimation methods |
| Figure 11.1 | Fig. 9.17 | p. 465 | Speech all-pole modeling processor |
| Figure 11.2 | Fig. 9.15 | p. 464 | Prewhitening / postcoloring concept |

---

**§11 Application Example: Linear Prediction in Speech Coding**

- <a href="#111-why-speech-is-predictable" style="color: #a0a0a0;">11.1 Why Speech Is Predictable</a>
- <a href="#112-lpc-model" style="color: #a0a0a0;">11.2 LPC Model</a>
- <a href="#113-three-coding-paradigms" style="color: #a0a0a0;">11.3 Three Coding Paradigms</a>
- <a href="#114-why-reflection-coefficients-are-useful-in-speech" style="color: #a0a0a0;">11.4 Why Reflection Coefficients Are Useful in Speech</a>
- <a href="#115-speech-spectrum-and-all-pole-envelope" style="color: #a0a0a0;">11.5 Speech Spectrum and All-Pole Envelope</a>
- <a href="#chapter-3-summary" style="color: #a0a0a0;">Chapter 3 Summary</a>
- <a href="#figure-source-checklist" style="color: #a0a0a0;">Figure Source Checklist</a>
- [Suggested Teaching Flow](#suggested-teaching-flow)

---

## Suggested Teaching Flow

1. Start with Figure 1.1 and emphasize that prediction is a special case of linear estimation.
2. Derive the orthogonality condition and the Toeplitz normal equations.
3. Explain AR modeling as the same mathematics viewed as a whitening problem.
4. Introduce Levinson-Durbin as an efficient recursive solver, not as a mysterious formula.
5. Interpret reflection coefficients as partial correlations.
6. Use the stability equivalence $\lvert\kappa_m\rvert<1$ to motivate lattice filters.
7. Compare autocorrelation, covariance, modified covariance, and Burg methods.
8. End with speech LPC because it makes the whole chapter feel practical.

---

### Answer to the Guiding Question

**Question.** An air-acoustic microphone measures $r(0)=100$ and $r(1)=80$ for a voiced segment. Find the first-order LPC predictor, residual statistics, spectral-envelope ratio, and RMS reduction.

**Answer.**

The first-order normal equation is $r(0)a_1=-r(1)$, hence

$$a_1=-\frac{80}{100}=-0.8.$$

The prediction-error filter is therefore

$$A_1(z)=1-0.8z^{-1},$$

and the predictor itself is $\hat x(n)=0.8x(n-1)$. The residual power is

$$P_1=r(0)+a_1r(1)=100-0.8(80)=36.$$

Thus the signal RMS is $\sqrt{100}=10$, whereas the residual RMS is $\sqrt{36}=6$: prediction lowers RMS by $4$ units, or $40\%$. The corresponding all-pole spectral envelope is proportional to $1/|1-0.8e^{-j\omega}|^2$, so

$$\frac{R_x(e^{j0})}{R_x(e^{j\pi})}
=\frac{|1+0.8|^2}{|1-0.8|^2}
=\frac{1.8^2}{0.2^2}=81.$$

**Numerically, the predictor weight is $0.8$, the prediction-error coefficient is $-0.8$, residual power is $36$, residual RMS is $6$, the envelope ratio is $81$, and the residual RMS is $40\%$ below the original.** This is why the residual exposes excitation or anomalies while the coefficient represents the slowly varying acoustic envelope.

---
