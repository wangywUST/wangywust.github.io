# Modern Digital Signal Processing
## Chapter 7: Adaptive Filters — LMS, RLS, and Tracking

> 📖 Textbook: Manolakis, Ingle, Kogon — *Statistical and Adaptive Signal Processing*, Artech House, 2005  
> Chapters covered: Ch. 10 (Adaptive Filters, §10.1–§10.8)

---

## Table of Contents

1. [§0 Chapter Roadmap: From Optimum Filters to Adaptive Filters](#0-chapter-roadmap-from-optimum-filters-to-adaptive-filters)
2. [§1 Adaptive Filtering Framework and Typical Applications](#1-adaptive-filtering-framework-and-typical-applications)
3. [§2 Steepest Descent: Deterministic Gradient Adaptation](#2-steepest-descent-deterministic-gradient-adaptation)
4. [§3 Least-Mean-Square Adaptive Filters](#3-least-mean-square-adaptive-filters)
5. [§4 LMS Variants and Practical Extensions](#4-lms-variants-and-practical-extensions)
6. [§5 LMS Applications: Echo Cancelation, Noise Cancelation, and Equalization](#5-lms-applications-echo-cancelation-noise-cancelation-and-equalization)
7. [§6 Recursive Least-Squares Adaptive Filters](#6-recursive-least-squares-adaptive-filters)
8. [§7 Tracking, Algorithm Selection, and Figure Checklist](#7-tracking-algorithm-selection-and-figure-checklist)

---

## Notation and Variable Definitions

This chapter continues the notation of Chapters 3 and 6. In Chapter 6, the filter coefficients were usually assumed to be computed from known second-order statistics. In this chapter, the coefficients are updated automatically from observed data.

### Time, Order, and Adaptation Indices

| Symbol | Definition |
|--------|------------|
| $n$ | Discrete-time sample index |
| $M$ | Number of adaptive filter coefficients / FIR filter length |
| $k$ | Iteration index for deterministic steepest-descent analysis |
| $K$ | Projection order or number of recent input vectors in affine projection algorithms |
| $L$ | Sliding-window length in sliding-window RLS |
| $\lambda$ | RLS forgetting factor, usually $0<\lambda\le 1$ |
| $\mu$ | LMS or steepest-descent step-size parameter |
| $\delta$ | Positive regularization constant used to initialize RLS |

### Signals and Errors

| Symbol | Definition |
|--------|------------|
| $x(n)$ | Input signal sample |
| $\mathbf{x}(n)$ | Adaptive-filter input vector, often $[x(n),x(n-1),\ldots,x(n-M+1)]^T$ |
| $y(n)$ | Desired response / training signal |
| $\hat y(n)$ | Adaptive filter output |
| $e(n)=y(n)-\hat y(n)$ | A priori error, computed using old coefficients before the update |
| $\epsilon(n)$ | A posteriori error, computed after the coefficient update in some algorithms |
| $s(n)$ | Signal of interest in noise-cancelation examples |
| $v(n)$ | Noise, interference, or observation disturbance |
| $u(n)$ | Combined disturbance in echo-cancelation examples |

### Coefficients, Statistics, and Performance Measures

| Symbol | Definition |
|--------|------------|
| $\mathbf{c}(n)$ | Adaptive coefficient vector after processing sample $n$ |
| $\mathbf{c}(n-1)$ | Coefficient vector used to filter sample $n$ |
| $\mathbf{c}_o$ | Optimum Wiener coefficient vector for a stationary SOE |
| $\mathbf{c}_o(n)$ | Time-varying optimum coefficient vector in a nonstationary SOE |
| $\tilde{\mathbf{c}}(n)=\mathbf{c}_o-\mathbf{c}(n)$ | Coefficient error vector in a stationary SOE |
| $\mathbf{R}=E\{\mathbf{x}(n)\mathbf{x}^H(n)\}$ | Input correlation matrix |
| $\mathbf{d}=E\{\mathbf{x}(n)y^\ast(n)\}$ | Cross-correlation vector between input and desired response |
| $P(\mathbf{c})=E\{|y(n)-\mathbf{c}^H\mathbf{x}(n)|^2\}$ | Mean-square error surface |
| $P_o$ | Minimum mean-square error (MMSE) |
| $P(n)$ | Instantaneous or time-indexed MSE / learning-curve quantity |
| EMSE | Excess mean-square error above the Wiener MMSE |
| $\mathcal{M}$ | Misadjustment, usually EMSE divided by $P_o$ |
| MSD | Mean-square deviation of coefficients, $E\{\|\mathbf{c}(n)-\mathbf{c}_o(n)\|^2\}$ |

### RLS Quantities

| Symbol | Definition |
|--------|------------|
| $\hat{\mathbf{R}}(n)$ | Exponentially weighted sample correlation matrix |
| $\hat{\mathbf{d}}(n)$ | Exponentially weighted sample cross-correlation vector |
| $\mathbf{P}(n)=\hat{\mathbf{R}}^{-1}(n)$ | Inverse correlation matrix estimate used by RLS |
| $\mathbf{g}(n)$ | RLS gain vector |
| $\alpha(n)$ | RLS normalization scalar |
| QR-RLS | Numerically stable RLS implementation based on QR decomposition |
| FTF / FAEST | Fast transversal-filter / fast a posteriori error sequential technique variants |

---

# §0 Chapter Roadmap: From Optimum Filters to Adaptive Filters

> 📖 Textbook §10.1 (Typical Applications of Adaptive Filters); §10.2 (Principles of Adaptive Filters)

## 0.1 Why Adaptive Filtering Is Needed

In Chapter 6, the Wiener filter was derived under an ideal assumption: the relevant second-order statistics are known. For an FIR Wiener filter, the coefficient vector is obtained from the normal equation

$$
\mathbf{R}\mathbf{c}_o=\mathbf{d}.
$$

This is mathematically clean, but it is rarely the whole story in practice. In many real systems,

- the correlation matrix $\mathbf{R}$ is unknown;
- the cross-correlation vector $\mathbf{d}$ is unknown;
- the signal operating environment changes with time;
- the filter must process samples immediately as they arrive.

An **adaptive filter** solves this practical problem by updating the coefficients from data. Instead of designing a fixed filter once, the system repeatedly applies a rule of the form

$$
\boxed{\text{old coefficients} + \text{correction from new data} \longrightarrow \text{new coefficients}.}
$$

The correction is driven by an error signal. This is the central idea of the chapter.

## 0.2 The Conceptual Bridge from Chapter 6 to Chapter 7

The Wiener filter answers the question:

> If the statistics are known, what is the best linear filter?

Adaptive filtering answers the more practical question:

> If the statistics are unknown or changing, how can the filter move toward the best linear filter while operating on data?

Therefore, an adaptive filter is not a different goal from the Wiener filter. It is an algorithmic way of approaching or tracking the Wiener solution.

The relationship can be summarized as follows.

| Viewpoint | Known Statistics? | Coefficients | Main Equation |
|----------|-------------------|--------------|---------------|
| Wiener filtering | Yes | Fixed optimum $\mathbf{c}_o$ | $\mathbf{R}\mathbf{c}_o=\mathbf{d}$ |
| Block least squares | Estimated from a data block | Fixed within block | $\hat{\mathbf{R}}\hat{\mathbf{c}}=\hat{\mathbf{d}}$ |
| LMS | Not explicitly estimated | Updated sample by sample | $\mathbf{c}(n)=\mathbf{c}(n-1)+\text{error}\times\text{input}$ |
| RLS | Estimated recursively | Updated sample by sample | $\mathbf{c}(n)=\mathbf{c}(n-1)+\mathbf{g}(n)e^\ast(n)$ |

The big theme is that **adaptation replaces a priori statistical knowledge by data-driven updating**.

## 0.3 The Three Questions We Will Keep Asking

Throughout this chapter, every algorithm should be evaluated using three questions.

| Question | Meaning |
|----------|---------|
| Does it converge? | Does $\mathbf{c}(n)$ approach the optimum solution in a stationary environment? |
| How fast does it converge? | How many samples are needed before useful performance is obtained? |
| How well does it track? | If $\mathbf{c}_o(n)$ changes, can the algorithm follow it without excessive noise? |

These questions already reveal the main tradeoff. A large adaptation gain gives fast response but large coefficient fluctuations. A small adaptation gain gives low steady-state noise but slow adaptation.

## 0.4 Representative Adaptive-Filtering Applications

The textbook motivates adaptive filtering through echo cancelation, linear predictive coding, noise cancelation, and equalization.

> ![Figure 0.1](./CourseADSP2026/Fig/Chapter_7/fig_0_1_textbook_fig_10_3_p501.png)
>
> *Figure 0.1 (Textbook Fig. 10.3, p. 501): Principle of echo cancelation. The adaptive filter learns an echo-path model and subtracts an echo replica from the received signal.*

In echo cancelation, the desired signal is not a stationary textbook sequence with known statistics. The echo path may change from call to call or even during a call. A fixed filter designed in advance is therefore inadequate.

> ![Figure 0.2](./CourseADSP2026/Fig/Chapter_7/fig_0_2_textbook_fig_10_5_p505.png)
>
> *Figure 0.2 (Textbook Fig. 10.5, p. 505): Linear predictive coding system. The predictor should adapt because the correlation structure of speech, audio, and video changes with time.*

In linear predictive coding, the predictor estimates the current sample from past reconstructed samples. If the predictor is good, the prediction error has smaller variance than the original signal, so it can be quantized more efficiently.

> ![Figure 0.3](./CourseADSP2026/Fig/Chapter_7/fig_0_3_textbook_fig_10_6_p506.png)
>
> *Figure 0.3 (Textbook Fig. 10.6, p. 506): Adaptive noise cancelation using a reference input. The adaptive filter uses the reference noise measurement to remove a correlated noise component from the primary input.*

In noise cancelation, the filter does not directly observe the clean signal $s(n)$. Instead, it observes a primary input $s(n)+v_1(n)$ and a reference input $v_2(n)$ that is correlated with the noise but ideally uncorrelated with the signal of interest.

## 0.5 Chapter Map

The chapter will proceed in the following order.

1. First, we define the adaptive-filter architecture and the supervised error signal.
2. Then, we study steepest descent, the deterministic ancestor of LMS.
3. Next, we derive LMS by replacing exact gradients with instantaneous random gradients.
4. We then analyze convergence, step-size selection, EMSE, and misadjustment.
5. After that, we introduce practical LMS extensions such as NLMS, transform-domain LMS, block LMS, affine projection, leakage, and variable step-size ideas.
6. Finally, we study RLS, its fast variants, and tracking behavior in nonstationary environments.

---

# §1 Adaptive Filtering Framework and Typical Applications

> 📖 Textbook §10.1; §10.2

## 1.1 Basic Elements of an Adaptive Filter

An adaptive filter has three essential components.

| Component | Function |
|----------|----------|
| Adjustable filter | Computes $\hat y(n)$ from input data and current coefficients |
| Performance evaluation | Computes an error or cost signal that measures current performance |
| Adaptation algorithm | Updates coefficients using the input, error, and past coefficients |

> ![Figure 1.1](./CourseADSP2026/Fig/Chapter_7/fig_1_1_textbook_fig_10_7_p507.png)
>
> *Figure 1.1 (Textbook Fig. 10.7, p. 507): Basic elements of a general adaptive filter. The filter, performance evaluator, and adaptation algorithm form a closed adjustment loop.*

A fixed filter has only the first component. An adaptive filter has a feedback mechanism that changes the filter parameters while the system is running.

This structure is useful only when the performance signal carries information about how the coefficients should change. In supervised adaptation, this information is provided by a desired response.

## 1.2 Supervised Adaptive Filtering

The most common structure in this chapter is supervised adaptive filtering.

> ![Figure 1.2](./CourseADSP2026/Fig/Chapter_7/fig_1_2_textbook_fig_10_8_p508.png)
>
> *Figure 1.2 (Textbook Fig. 10.8, p. 508): Basic elements of a supervised adaptive filter. The error is computed by comparing the filter output with a desired response.*

At time $n$, the adaptive filter performs three steps.

First, it filters the input vector using the current coefficient vector:

$$
\hat y(n)=\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

Second, it computes the a priori error:

$$
e(n)=y(n)-\hat y(n)=y(n)-\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

Third, it updates the coefficient vector:

$$
\mathbf{c}(n)=\mathbf{c}(n-1)+\Delta\mathbf{c}(n),
$$

where $\Delta\mathbf{c}(n)$ is determined by the adaptive algorithm.

The distinction between $\mathbf{c}(n-1)$ and $\mathbf{c}(n)$ is important. The old coefficients produce the current output; the current error then changes the coefficients for future samples.

## 1.3 Linear Combiner versus FIR Adaptive Filter

The same vector formula can represent two different physical structures.

> ![Figure 1.3](./CourseADSP2026/Fig/Chapter_7/fig_1_3_textbook_fig_10_9_p509.png)
>
> *Figure 1.3 (Textbook Fig. 10.9, p. 509): Difference between a multiple-input linear combiner and a single-input FIR adaptive filter.*

For a multiple-input linear combiner,

$$
\mathbf{x}(n)=
\begin{bmatrix}
 x_1(n) & x_2(n) & \cdots & x_M(n)
\end{bmatrix}^T.
$$

For a single-input FIR filter,

$$
\mathbf{x}(n)=
\begin{bmatrix}
 x(n) & x(n-1) & \cdots & x(n-M+1)
\end{bmatrix}^T.
$$

The output equation is the same in both cases:

$$
\hat y(n)=\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

This is why adaptive-filter theory is often developed using vector notation. The same algorithm can apply to sensor arrays, echo cancelers, predictors, and equalizers.

## 1.4 Optimum Filters versus Adaptive Filters

The optimum filter assumes that the statistics are known and solves

$$
\mathbf{R}(n)\mathbf{c}_o(n)=\mathbf{d}(n).
$$

The adaptive filter does not know $\mathbf{R}(n)$ and $\mathbf{d}(n)$ exactly. It uses sample-by-sample updates instead.

> ![Figure 1.4](./CourseADSP2026/Fig/Chapter_7/fig_1_4_textbook_fig_10_10_p510.png)
>
> *Figure 1.4 (Textbook Fig. 10.10, p. 510): Optimum filtering computes coefficients from statistics; adaptive filtering updates coefficients directly from incoming data.*

This distinction gives two useful interpretations.

First, in a stationary environment, an adaptive filter tries to approach a fixed Wiener solution.

Second, in a nonstationary environment, an adaptive filter tries to track a moving Wiener solution.

## 1.5 A Priori and A Posteriori Errors

The timing of the update matters.

> ![Figure 1.5](./CourseADSP2026/Fig/Chapter_7/fig_1_5_textbook_fig_10_11_p512.png)
>
> *Figure 1.5 (Textbook Fig. 10.11, p. 512): Timing diagrams for a priori and a posteriori adaptive algorithms.*

The **a priori error** uses the coefficient vector before it has been updated by sample $n$:

$$
e(n)=y(n)-\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

The **a posteriori error** uses the updated coefficient vector:

$$
\epsilon(n)=y(n)-\mathbf{c}^H(n)\mathbf{x}(n).
$$

In many LMS derivations, the a priori error is used to update the coefficients. In some RLS and projection algorithms, the a posteriori error has a special algebraic role.

The practical lesson is simple: when implementing an adaptive filter, one must be precise about whether the error is computed before or after the update.

## 1.6 Stationary and Nonstationary Modes of Operation

> ![Figure 1.6](./CourseADSP2026/Fig/Chapter_7/fig_1_6_textbook_fig_10_12_p515.png)
>
> *Figure 1.6 (Textbook Fig. 10.12, p. 515): Modes of operation in stationary and nonstationary signal operating environments.*

In a stationary SOE, the optimum coefficient vector $\mathbf{c}_o$ is fixed. The adaptive filter first has a transient phase, then fluctuates around $\mathbf{c}_o$ in steady state.

In a nonstationary SOE, the optimum coefficient vector $\mathbf{c}_o(n)$ changes. The adaptive filter has a tracking task. It must stay close to the moving optimum without injecting too much adaptation noise.

This difference is the reason that one cannot choose a step size by only asking for stability. The step size also determines steady-state fluctuations and tracking ability.

---

# §2 Steepest Descent: Deterministic Gradient Adaptation

> 📖 Textbook §10.3 (Method of Steepest Descent)

## 2.1 The MSE Surface

For a linear combiner or FIR adaptive filter, the MSE cost is

$$
P(\mathbf{c})=E\{|y(n)-\mathbf{c}^H\mathbf{x}(n)|^2\}.
$$

Expanding the square gives

$$
P(\mathbf{c})
= P_y - \mathbf{c}^H\mathbf{d} - \mathbf{d}^H\mathbf{c}
+ \mathbf{c}^H\mathbf{R}\mathbf{c},
$$

where

$$
P_y=E\{|y(n)|^2\},\quad
\mathbf{R}=E\{\mathbf{x}(n)\mathbf{x}^H(n)\},\quad
\mathbf{d}=E\{\mathbf{x}(n)y^\ast(n)\}.
$$

If $\mathbf{R}$ is positive definite, this is a quadratic bowl. Its minimum is obtained by the Wiener solution

$$
\boxed{\mathbf{c}_o=\mathbf{R}^{-1}\mathbf{d}.}
$$

The gradient with respect to the coefficient vector has the form

$$
\nabla P(\mathbf{c})=-2[\mathbf{d}-\mathbf{R}\mathbf{c}].
$$

The vector $\mathbf{d}-\mathbf{R}\mathbf{c}$ therefore points toward the minimum in the negative-gradient direction.

## 2.2 Gradient Search Intuition

> ![Figure 2.1](./CourseADSP2026/Fig/Chapter_7/fig_2_1_textbook_fig_10_13_p518.png)
>
> *Figure 2.1 (Textbook Fig. 10.13, p. 518): Gradient search of the MSE surface. Each update moves the coefficient vector downhill toward the minimum.*

The steepest-descent algorithm updates coefficients as

$$
\boxed{\mathbf{c}_{k+1}=\mathbf{c}_k+2\mu[\mathbf{d}-\mathbf{R}\mathbf{c}_k].}
$$

Here $k$ is an iteration index, not necessarily the same as the sample index $n$. The algorithm assumes that $\mathbf{R}$ and $\mathbf{d}$ are known. Therefore, steepest descent is not yet a practical adaptive filter for unknown statistics; it is the conceptual bridge to LMS.

## 2.3 Coefficient Error Dynamics

Define the coefficient error vector

$$
\tilde{\mathbf{c}}_k=\mathbf{c}_k-\mathbf{c}_o.
$$

Using $\mathbf{d}=\mathbf{R}\mathbf{c}_o$, the steepest-descent update becomes

$$
\tilde{\mathbf{c}}_{k+1}=(\mathbf{I}-2\mu\mathbf{R})\tilde{\mathbf{c}}_k.
$$

Diagonalize the correlation matrix as

$$
\mathbf{R}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^H,
$$

where $\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_M)$. In the principal-component coordinates,

$$
\tilde c'_{k+1,i}=(1-2\mu\lambda_i)\tilde c'_{k,i}.
$$

Thus each eigen-direction decays with its own factor $1-2\mu\lambda_i$.

## 2.4 Stability Condition

For convergence in every eigen-direction, we need

$$
|1-2\mu\lambda_i|<1
\quad \text{for all } i.
$$

This gives

$$
\boxed{0<\mu<\frac{1}{\lambda_{\max}}.}
$$

If $\mu$ is too small, convergence is slow. If $\mu$ is too large, the trajectory oscillates or diverges.

## 2.5 Eigenvalue Spread and Convergence Rate

The eigenvalue spread is

$$
\chi(\mathbf{R})=\frac{\lambda_{\max}}{\lambda_{\min}}.
$$

A small eigenvalue spread means the MSE bowl is close to circular. All directions have similar curvature, so gradient descent converges smoothly.

A large eigenvalue spread means the MSE bowl is elongated. The algorithm must choose a step size small enough for the steep direction, but then progress is slow along the flat direction.

> ![Figure 2.2](./CourseADSP2026/Fig/Chapter_7/fig_2_2_textbook_fig_10_14_p521.png)
>
> *Figure 2.2 (Textbook Fig. 10.14, p. 521): Steepest-descent behavior with small eigenvalue spread. The trajectory approaches the optimum relatively directly.*

> ![Figure 2.3](./CourseADSP2026/Fig/Chapter_7/fig_2_3_textbook_fig_10_15_p522.png)
>
> *Figure 2.3 (Textbook Fig. 10.15, p. 522): Steepest-descent behavior with large eigenvalue spread. The elongated error surface causes slower convergence.*

> ![Figure 2.4](./CourseADSP2026/Fig/Chapter_7/fig_2_4_textbook_fig_10_16_p523.png)
>
> *Figure 2.4 (Textbook Fig. 10.16, p. 523): Effect of step size for large eigenvalue spread. A larger step may accelerate convergence but can produce oscillatory behavior.*

The practical lesson is fundamental:

> **Adaptive-filter convergence depends not only on input power, but also on the eigenvalue distribution of the input correlation matrix.**

Colored inputs usually have large eigenvalue spread. Therefore LMS converges slowly for highly correlated inputs unless additional techniques are used.

---

# §3 Least-Mean-Square Adaptive Filters

> 📖 Textbook §10.4 (Least-Mean-Square Adaptive Filters)

## 3.1 From Steepest Descent to LMS

Steepest descent requires the exact gradient

$$
\nabla P(\mathbf{c})=-2[\mathbf{d}-\mathbf{R}\mathbf{c}],
$$

which requires $\mathbf{R}$ and $\mathbf{d}$. LMS replaces the exact gradient by an instantaneous estimate from the current sample.

For the instantaneous squared error

$$
|e(n)|^2=|y(n)-\mathbf{c}^H(n-1)\mathbf{x}(n)|^2,
$$

the instantaneous negative-gradient direction is proportional to

$$
\mathbf{x}(n)e^\ast(n).
$$

This gives the LMS update

$$
\boxed{\mathbf{c}(n)=\mathbf{c}(n-1)+2\mu\mathbf{x}(n)e^\ast(n).}
$$

Some books absorb the factor 2 into the step size and write

$$
\mathbf{c}(n)=\mathbf{c}(n-1)+\mu\mathbf{x}(n)e^\ast(n).
$$

Both conventions describe the same algorithm. In this lecture, when discussing textbook formulas, we use the $2\mu$ convention.

## 3.2 Geometric Interpretation of LMS

> ![Figure 3.1](./CourseADSP2026/Fig/Chapter_7/fig_3_1_textbook_fig_10_17_p525.png)
>
> *Figure 3.1 (Textbook Fig. 10.17, p. 525): Geometric approach to the LMS algorithm. The update removes a component of the coefficient error in the direction of the current input vector.*

The LMS update can be interpreted as a projection-like correction. At time $n$, the input vector $\mathbf{x}(n)$ gives only one direction in the coefficient space. The algorithm can reduce the part of the coefficient error that is visible along that direction. It cannot correct components that are orthogonal to $\mathbf{x}(n)$ using this sample alone.

This explains why LMS needs many samples. Each new input vector gives another direction of information.

## 3.3 Complete LMS Algorithm for an FIR Adaptive Filter

For an $M$-tap FIR adaptive filter, define

$$
\mathbf{x}(n)=
\begin{bmatrix}
 x(n) & x(n-1) & \cdots & x(n-M+1)
\end{bmatrix}^T.
$$

At each time instant:

1. **Filtering**

$$
\hat y(n)=\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

2. **Error computation**

$$
e(n)=y(n)-\hat y(n).
$$

3. **Coefficient update**

$$
\boxed{\mathbf{c}(n)=\mathbf{c}(n-1)+2\mu\mathbf{x}(n)e^\ast(n).}
$$

> ![Figure 3.2](./CourseADSP2026/Fig/Chapter_7/fig_3_2_textbook_fig_10_18_p526.png)
>
> *Figure 3.2 (Textbook Fig. 10.18, p. 526): FIR adaptive filter realization using the LMS algorithm.*

The algorithm is attractive because it is simple. It requires no matrix inversion and no explicit correlation estimation.

For complex-valued data, the conjugate on $e^\ast(n)$ is essential. For real-valued data, the conjugate has no effect.

## 3.4 LMS in a Stationary Signal Operating Environment

For analysis, assume a stationary environment in which

$$
y(n)=\mathbf{c}_o^H\mathbf{x}(n)+e_o(n),
$$

where $e_o(n)$ is the optimum error, orthogonal to the input vector in the Wiener sense.

> ![Figure 3.3](./CourseADSP2026/Fig/Chapter_7/fig_3_3_textbook_fig_10_19_p527.png)
>
> *Figure 3.3 (Textbook Fig. 10.19, p. 527): LMS algorithm in a stationary signal operating environment.*

The LMS error can be decomposed as

$$
e(n)=e_o(n)+[\mathbf{c}_o-\mathbf{c}(n-1)]^H\mathbf{x}(n).
$$

The first term is irreducible error. The second term is caused by coefficient mismatch. Adaptation tries to reduce the second term.

## 3.5 Mean Convergence

Under common independence assumptions used for LMS analysis, the expected coefficient error approximately follows

$$
E\{\tilde{\mathbf{c}}(n)\}
=(\mathbf{I}-2\mu\mathbf{R})E\{\tilde{\mathbf{c}}(n-1)\}.
$$

Therefore, the same eigenvalue argument gives the mean-convergence condition

$$
\boxed{0<\mu<\frac{1}{\lambda_{\max}}}
\quad \text{under the textbook } 2\mu \text{ convention.}
$$

If the LMS update is written without the factor 2, the corresponding bound is commonly written as

$$
0<\mu<\frac{2}{\lambda_{\max}}.
$$

A convenient conservative rule is based on the trace:

$$
\lambda_{\max}\le \operatorname{tr}(\mathbf{R})=E\{\|\mathbf{x}(n)\|^2\}.
$$

Thus a simple practical bound is

$$
0<\mu<\frac{1}{\operatorname{tr}(\mathbf{R})}
\quad \text{or} \quad
0<\mu<\frac{2}{\operatorname{tr}(\mathbf{R})}
$$

depending on the step-size convention.

## 3.6 Mean-Square Behavior, EMSE, and Misadjustment

Mean convergence only describes the average coefficient vector. It does not describe the steady-state random fluctuations around the optimum.

The MSE can be decomposed as

$$
P(n)=P_o+P_{\text{ex}}(n),
$$

where

- $P_o$ is the Wiener MMSE;
- $P_{\text{ex}}(n)$ is the excess MSE caused by imperfect coefficients.

The steady-state excess MSE is called EMSE:

$$
\text{EMSE}=\lim_{n\to\infty}P_{\text{ex}}(n).
$$

The misadjustment is the normalized EMSE:

$$
\mathcal{M}=\frac{\text{EMSE}}{P_o}.
$$

For small step sizes, a useful approximation is that misadjustment grows approximately linearly with the step size and with total input power. Therefore:

- larger $\mu$ gives faster convergence but larger steady-state EMSE;
- smaller $\mu$ gives lower EMSE but slower convergence.

This is the central LMS design tradeoff.

## 3.7 LMS Learning Curves and Eigenvalue Spread

> ![Figure 3.4](./CourseADSP2026/Fig/Chapter_7/fig_3_4_textbook_fig_10_20_p537.png)
>
> *Figure 3.4 (Textbook Fig. 10.20, p. 537): LMS performance with small eigenvalue spread. The averaged trajectory is smooth, but individual learning curves are noisy because the gradient is random.*

> ![Figure 3.5](./CourseADSP2026/Fig/Chapter_7/fig_3_5_textbook_fig_10_21_p538.png)
>
> *Figure 3.5 (Textbook Fig. 10.21, p. 538): LMS performance with large eigenvalue spread. Convergence is slower even with the same step-size setting.*

These figures show a key difference between steepest descent and LMS.

Steepest descent follows a deterministic path on the MSE surface. LMS follows a random path because each instantaneous gradient is noisy. Averaging many LMS trajectories recovers the deterministic trend, but each individual realization fluctuates.

The figures also show why highly correlated inputs are difficult: large eigenvalue spread slows convergence.

## 3.8 LMS Step-Size Selection

A practical step-size choice must balance four constraints.

| Design Goal | Effect of Larger $\mu$ | Effect of Smaller $\mu$ |
|------------|-------------------------|--------------------------|
| Stability | More risk of instability | Safer |
| Convergence speed | Faster | Slower |
| Steady-state EMSE | Larger | Smaller |
| Tracking ability | Better tracking of fast changes | Poorer tracking of fast changes |

A good engineering approach is:

1. Estimate the average input power $E\{\|\mathbf{x}(n)\|^2\}$.
2. Choose a conservative initial $\mu$ well below the stability bound.
3. Increase $\mu$ if convergence is too slow.
4. Decrease $\mu$ if steady-state fluctuations or residual error are too large.
5. For strongly time-varying input power, use NLMS or a variable step-size method.

---

# §4 LMS Variants and Practical Extensions

> 📖 Textbook §10.4.4–§10.4.5; related practical discussion in §10.2 and §10.7

## 4.1 Normalized LMS (NLMS)

The LMS update uses a fixed step size. If the input vector norm changes substantially, a fixed $\mu$ can be too small when the input power is low and too large when the input power is high.

NLMS scales the update by the input-vector energy:

$$
\boxed{
\mathbf{c}(n)=\mathbf{c}(n-1)+\frac{\tilde\mu}{\|\mathbf{x}(n)\|^2}\mathbf{x}(n)e^\ast(n).
}
$$

To avoid division by a very small number, the practical form is

$$
\mathbf{c}(n)=\mathbf{c}(n-1)+\frac{\tilde\mu}{\epsilon+\|\mathbf{x}(n)\|^2}\mathbf{x}(n)e^\ast(n),
$$

where $\epsilon>0$ is a small regularization constant.

NLMS is especially useful when the input power varies over time, as in speech, audio, or communication channels with fading.

## 4.2 LMS-Newton Idea

The slow convergence of LMS for colored inputs is caused by eigenvalue spread. A Newton-type method would precondition the gradient by $\mathbf{R}^{-1}$:

$$
\mathbf{c}(n)=\mathbf{c}(n-1)+2\mu\mathbf{R}^{-1}\mathbf{x}(n)e^\ast(n).
$$

If $\mathbf{R}^{-1}$ were known, this would make the error surface effectively spherical and reduce dependence on eigenvalue spread.

The difficulty is that estimating and inverting $\mathbf{R}$ can be expensive. RLS can be viewed as a practical recursive way of using inverse-correlation information.

## 4.3 Transform-Domain LMS

Another way to reduce eigenvalue spread is to transform the input into a domain where components are less correlated.

> ![Figure 4.1](./CourseADSP2026/Fig/Chapter_7/fig_4_1_textbook_fig_10_30_p547.png)
>
> *Figure 4.1 (Textbook Fig. 10.30, p. 547): Transform-domain LMS adaptive filter structure. A decorrelation transform improves the conditioning seen by LMS.*

The general idea is:

$$
\mathbf{u}(n)=\mathbf{T}\mathbf{x}(n),
$$

where $\mathbf{T}$ may be a DFT, DCT, or another approximate decorrelating transform. LMS is then applied in the transformed domain.

If the transformed components are scaled by estimated powers, the algorithm can use different effective step sizes for different modes.

This is useful when:

- the input is strongly correlated;
- fast convergence is needed;
- full RLS is too expensive.

## 4.4 Block LMS

> ![Figure 4.2](./CourseADSP2026/Fig/Chapter_7/fig_4_2_textbook_fig_10_31_p547.png)
>
> *Figure 4.2 (Textbook Fig. 10.31, p. 547): Block adaptive filter structure. Several samples are processed together before the coefficient update is performed.*

In block LMS, the coefficient update is based on a block of samples rather than a single sample. A simplified block update has the form

$$
\mathbf{c}_{b+1}=\mathbf{c}_b+2\mu\sum_{n\in \text{block } b}\mathbf{x}(n)e^\ast(n).
$$

Block processing has two advantages.

First, it can average gradient noise over multiple samples. Second, it can exploit FFT-based convolution for long filters.

The disadvantage is latency: coefficients are updated only once per block.

## 4.5 Affine Projection Algorithm

NLMS uses the current input vector to update the filter. The affine projection algorithm uses the most recent $K$ input vectors.

Let

$$
\mathbf{X}(n)=
\begin{bmatrix}
\mathbf{x}(n) & \mathbf{x}(n-1) & \cdots & \mathbf{x}(n-K+1)
\end{bmatrix}.
$$

The update has the conceptual form

$$
\mathbf{c}(n)=\mathbf{c}(n-1)+\mathbf{X}(n)
[\mathbf{X}^H(n)\mathbf{X}(n)+\epsilon\mathbf{I}]^{-1}\mathbf{e}^\ast(n),
$$

where $\mathbf{e}(n)$ is a vector of recent errors.

When $K=1$, affine projection reduces to NLMS. For larger $K$, it often converges faster for correlated inputs, but its computational cost is higher.

## 4.6 Leaky LMS

If the input correlation matrix is singular or nearly singular, some coefficient directions may be weakly controlled by data. Coefficients can drift in those directions.

Leaky LMS adds a penalty on coefficient norm:

$$
J(n)=|e(n)|^2+\alpha\|\mathbf{c}(n)\|^2.
$$

A typical update is

$$
\boxed{\mathbf{c}(n)=(1-\alpha\mu)\mathbf{c}(n-1)+\mu\mathbf{x}(n)e^\ast(n).}
$$

The leakage term pulls the coefficients toward zero. It improves robustness, but it introduces bias because the steady-state solution is no longer exactly the Wiener solution.

## 4.7 Reduced-Complexity and Variable-Step LMS Methods

Several LMS variants trade accuracy, complexity, and convergence speed.

| Variant | Main Idea | Advantage | Cost / Limitation |
|--------|-----------|-----------|-------------------|
| Sign-error LMS | Replace $e(n)$ by $\operatorname{sgn}(e(n))$ | Lower multiplication cost | Slower and less accurate |
| Sign-data LMS | Replace input samples by signs | Simple hardware implementation | Lower precision |
| Sign-sign LMS | Use signs of both error and data | Very low complexity | Coarse adaptation |
| Block LMS | Update once per block | FFT acceleration for long filters | Latency |
| Variable-step LMS | Adjust $\mu(n)$ over time | Fast convergence and low EMSE | Requires step-control rule |
| Leaky LMS | Add coefficient shrinkage | Robust to singular correlation | Biased solution |

The guiding principle is always the same: modify the update to fit the signal environment and implementation constraints.

## 4.8 Gradient Adaptive Lattice Filters

A lattice structure can orthogonalize prediction-error signals order by order. This reduces coupling between parameters and can accelerate convergence when the input is colored.

The advantage is better numerical behavior and potentially faster convergence. The disadvantage is increased implementation complexity and computational cost compared with direct-form LMS.

Lattice structures are especially natural in linear prediction, speech modeling, and adaptive spectral estimation.

---

# §5 LMS Applications: Echo Cancelation, Noise Cancelation, and Equalization

> 📖 Textbook §10.1; §10.4.4

## 5.1 Adaptive Noise Cancelation Revisited

In adaptive noise cancelation, the primary input is

$$
d(n)=s(n)+v_1(n),
$$

and the reference input is $v_2(n)$, which is correlated with $v_1(n)$ but uncorrelated with $s(n)$.

The adaptive filter output $\hat v_1(n)$ estimates the noise in the primary channel. The output error is

$$
e(n)=d(n)-\hat v_1(n).
$$

When the filter minimizes $E\{|e(n)|^2\}$, it suppresses the part of the primary-channel noise predictable from the reference input. The desired signal remains because it is not correlated with the reference noise.

This is why the reference sensor must measure noise that is correlated with the interference but should not contain the desired signal.

## 5.2 Echo Cancelation in Full-Duplex Data Transmission

> ![Figure 3.6](./CourseADSP2026/Fig/Chapter_7/fig_3_6_textbook_fig_10_22_p539.png)
>
> *Figure 5.1 (Textbook Fig. 10.22, p. 539): Full-duplex transmission system using echo cancelers in modems.*

The echo canceler uses the known transmitted signal as the input to an adaptive FIR filter. The desired response is the received signal containing echo. The adaptive filter learns the echo path and subtracts an echo replica.

A simplified echo model is

$$
y(n)=\mathbf{c}_o^T\mathbf{x}(n),
$$

where $\mathbf{c}_o$ represents the echo-path impulse response. The adaptive filter tries to identify $\mathbf{c}_o$.

> ![Figure 3.7](./CourseADSP2026/Fig/Chapter_7/fig_3_7_textbook_fig_10_23_p539.png)
>
> *Figure 5.2 (Textbook Fig. 10.23, p. 539): Simulation setup for adaptive echo cancelation.*

This is a system-identification configuration. The input $x(n)$ is known; the unknown system is the echo path; the adaptive filter tries to match it.

## 5.3 Convergence-Residual Tradeoff in Echo Cancelation

> ![Figure 3.8](./CourseADSP2026/Fig/Chapter_7/fig_3_8_textbook_fig_10_24_p541.png)
>
> *Figure 5.3 (Textbook Fig. 10.24, p. 541): LMS echo-cancelation performance. Larger step size gives faster convergence but higher residual echo power.*

The figure shows the same LMS tradeoff in a practical setting.

A large step size reduces the echo quickly. However, after convergence, the residual echo power remains higher because the coefficients fluctuate more.

A small step size converges more slowly. However, the final residual echo can be lower.

For echo cancelation, the best step size depends on how quickly the echo path changes. A slowly changing echo path favors small $\mu$. A rapidly changing echo path requires larger $\mu$ or a more advanced algorithm.

## 5.4 Adaptive Channel Equalization

A communication channel spreads symbols in time. The received sample contains not only the desired symbol but also neighboring symbols. This is intersymbol interference (ISI).

An adaptive equalizer uses a training sequence or decision-directed feedback to adjust a filter that compensates for the channel.

> ![Figure 3.9](./CourseADSP2026/Fig/Chapter_7/fig_3_9_textbook_fig_10_25_p542.png)
>
> *Figure 5.4 (Textbook Fig. 10.25, p. 542): Adaptive equalizer model in a data transmission system.*

> ![Figure 3.10](./CourseADSP2026/Fig/Chapter_7/fig_3_10_textbook_fig_10_26_p543.png)
>
> *Figure 5.5 (Textbook Fig. 10.26, p. 543): System used for experimental investigation of LMS adaptive equalization.*

The equalizer input is the distorted received sequence. The desired response is often a delayed version of the transmitted training sequence:

$$
y_d(n)=y(n-D).
$$

The delay $D$ is selected so that the equalizer can approximate a realizable inverse of the channel.

## 5.5 Eigenvalue Spread in Equalization

> ![Figure 3.11](./CourseADSP2026/Fig/Chapter_7/fig_3_11_textbook_fig_10_27_p544.png)
>
> *Figure 5.6 (Textbook Fig. 10.27, p. 544): LMS adaptive equalizer performance. Channel characteristics affect eigenvalue spread and therefore convergence behavior.*

A more distorted channel often produces a more correlated equalizer input. This increases eigenvalue spread and slows LMS convergence.

This explains why communication equalizers often benefit from normalized LMS, affine projection, transform-domain methods, or RLS.

## 5.6 Step-Size Effect in Equalization

> ![Figure 3.12](./CourseADSP2026/Fig/Chapter_7/fig_3_12_textbook_fig_10_28_p544.png)
>
> *Figure 5.7 (Textbook Fig. 10.28, p. 544): MSE learning curves of LMS equalization for different step sizes.*

The same pattern appears again.

- Small $\mu$: slow learning, low fluctuation.
- Medium $\mu$: useful compromise.
- Large $\mu$: fast initial learning but possibly high steady-state error or instability.

## 5.7 Time-Domain Realizations of Equalization

> ![Figure 3.13](./CourseADSP2026/Fig/Chapter_7/fig_3_13_textbook_fig_10_29_p545.png)
>
> *Figure 5.8 (Textbook Fig. 10.29, p. 545): Transmitted, received, and equalized sample sequences for an FIR LMS equalizer.*

The equalized sequence should look closer to the transmitted sequence than the raw received sequence. This visual comparison is useful because equalization is not just an abstract MSE problem. It directly affects symbol decisions.

---

# §6 Recursive Least-Squares Adaptive Filters

> 📖 Textbook §10.5 (Recursive Least-Squares Adaptive Filters); §10.7 (Fast RLS Algorithms)

## 6.1 Motivation: Why RLS?

LMS is simple, but its convergence can be slow when the input is correlated. RLS is more complex, but it often converges much faster and is less sensitive to eigenvalue spread.

The key difference is the criterion.

| Algorithm | Criterion | Statistics Used | Typical Behavior |
|----------|-----------|-----------------|------------------|
| LMS | Stochastic gradient approximation to MSE | Instantaneous gradient | Simple, robust, slower for colored inputs |
| RLS | Deterministic weighted least squares | All past data with forgetting | Fast convergence, higher complexity |

RLS minimizes the exponentially weighted least-squares cost

$$
J_n(\mathbf{c})=
\sum_{i=0}^{n}\lambda^{n-i}
|y(i)-\mathbf{c}^H\mathbf{x}(i)|^2.
$$

Here $0<\lambda\le 1$ is the forgetting factor.

If $\lambda=1$, all past samples are weighted equally. If $\lambda<1$, older samples are discounted, allowing tracking of time variation.

> ![Figure 5.1](./CourseADSP2026/Fig/Chapter_7/fig_5_1_textbook_fig_10_32_p549.png)
>
> *Figure 6.1 (Textbook Fig. 10.32, p. 549): Exponential weighting of observations. Older data receive smaller weights when $\lambda<1$.*

## 6.2 Weighted Normal Equations

Define

$$
\hat{\mathbf{R}}(n)=
\sum_{i=0}^{n}\lambda^{n-i}\mathbf{x}(i)\mathbf{x}^H(i),
$$

and

$$
\hat{\mathbf{d}}(n)=
\sum_{i=0}^{n}\lambda^{n-i}\mathbf{x}(i)y^\ast(i).
$$

The least-squares solution satisfies

$$
\boxed{\hat{\mathbf{R}}(n)\mathbf{c}(n)=\hat{\mathbf{d}}(n).}
$$

The matrices can be updated recursively:

$$
\hat{\mathbf{R}}(n)=\lambda\hat{\mathbf{R}}(n-1)+\mathbf{x}(n)\mathbf{x}^H(n),
$$

$$
\hat{\mathbf{d}}(n)=\lambda\hat{\mathbf{d}}(n-1)+\mathbf{x}(n)y^\ast(n).
$$

A direct solution would require solving a linear system at every time step. RLS avoids this by updating the inverse correlation matrix.

## 6.3 Matrix Inversion Lemma and RLS Gain

Let

$$
\mathbf{P}(n)=\hat{\mathbf{R}}^{-1}(n).
$$

Using the matrix inversion lemma, the inverse can be updated without a full matrix inversion:

$$
\mathbf{g}(n)=
\frac{\mathbf{P}(n-1)\mathbf{x}(n)}
{\lambda+\mathbf{x}^H(n)\mathbf{P}(n-1)\mathbf{x}(n)},
$$

$$
\mathbf{P}(n)=\lambda^{-1}
\left[\mathbf{P}(n-1)-\mathbf{g}(n)\mathbf{x}^H(n)\mathbf{P}(n-1)\right].
$$

The a priori error is

$$
e(n)=y(n)-\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

The coefficient update is

$$
\boxed{\mathbf{c}(n)=\mathbf{c}(n-1)+\mathbf{g}(n)e^\ast(n).}
$$

> ![Figure 5.2](./CourseADSP2026/Fig/Chapter_7/fig_5_2_textbook_fig_10_33_p552.png)
>
> *Figure 6.2 (Textbook Fig. 10.33, p. 552): Basic elements of the a priori LS adaptive filter. The gain vector controls how much the current error changes the coefficient vector.*

## 6.4 Conventional RLS Algorithm

A practical conventional RLS algorithm can be written as follows.

### Initialization

Choose

$$
\mathbf{c}(-1)=\mathbf{0},
\quad
\mathbf{P}(-1)=\delta^{-1}\mathbf{I},
$$

where $\delta>0$ is small when little prior information is available.

### Recursion for each sample $n$

1. Compute the gain denominator:

$$
\alpha(n)=\lambda+\mathbf{x}^H(n)\mathbf{P}(n-1)\mathbf{x}(n).
$$

2. Compute the gain vector:

$$
\mathbf{g}(n)=\frac{\mathbf{P}(n-1)\mathbf{x}(n)}{\alpha(n)}.
$$

3. Compute the a priori error:

$$
e(n)=y(n)-\mathbf{c}^H(n-1)\mathbf{x}(n).
$$

4. Update the coefficients:

$$
\mathbf{c}(n)=\mathbf{c}(n-1)+\mathbf{g}(n)e^\ast(n).
$$

5. Update the inverse correlation matrix:

$$
\mathbf{P}(n)=\lambda^{-1}
\left[\mathbf{P}(n-1)-\mathbf{g}(n)\mathbf{x}^H(n)\mathbf{P}(n-1)\right].
$$

## 6.5 Choosing the Forgetting Factor

The forgetting factor determines the memory of RLS.

| $\lambda$ | Behavior | Use Case |
|----------|----------|----------|
| $\lambda=1$ | Uses all past data equally | Stationary environment |
| $\lambda$ close to 1 | Long memory, low noise | Slowly varying environment |
| Smaller $\lambda$ | Short memory, faster tracking | Rapidly varying environment |

A common practical range is roughly $0.95\le \lambda <1$ for many slowly varying systems, but the correct value depends on the application.

The effective memory length is often interpreted as being on the order of

$$
\frac{1}{1-\lambda}.
$$

Thus $\lambda=0.99$ remembers about 100 samples, while $\lambda=0.999$ remembers about 1000 samples in a rough engineering sense.

## 6.6 RLS Performance in Equalization

> ![Figure 5.3](./CourseADSP2026/Fig/Chapter_7/fig_5_3_textbook_fig_10_34_p559.png)
>
> *Figure 6.3 (Textbook Fig. 10.34, p. 559): RLS adaptive equalizer performance. RLS converges rapidly compared with LMS in similar equalization tasks.*

> ![Figure 5.4](./CourseADSP2026/Fig/Chapter_7/fig_5_4_textbook_fig_10_35_p559.png)
>
> *Figure 6.4 (Textbook Fig. 10.35, p. 559): RLS MSE learning curves for adaptive equalization.*

Compared with LMS, RLS usually converges in a number of samples proportional to the filter order, rather than being strongly limited by eigenvalue spread. This is the main practical advantage of RLS.

However, RLS has higher computational complexity, typically $O(M^2)$ per sample for conventional RLS, compared with $O(M)$ for LMS.

## 6.7 Sliding-Window RLS

Exponential forgetting is not the only way to localize the data. Sliding-window RLS uses only the most recent $L$ samples:

$$
J_n(\mathbf{c})=
\sum_{i=n-L+1}^{n}|y(i)-\mathbf{c}^H\mathbf{x}(i)|^2.
$$

This gives equal weight to the recent window and zero weight to older samples.

The advantage is a clear finite memory. The disadvantage is computational complexity: the algorithm must add the newest sample and remove the oldest sample, which makes the recursion more involved.

## 6.8 Square-Root and QR-RLS Algorithms

Conventional RLS updates $\mathbf{P}(n)$ directly. In finite-precision arithmetic, this can lead to numerical problems such as loss of symmetry or loss of positive definiteness.

Square-root algorithms update a factor of the correlation matrix or inverse correlation matrix, such as a Cholesky factor. QR-RLS uses orthogonal transformations to maintain numerical stability.

The main idea is not to form unstable inverses directly. Instead, update a factorization.

## 6.9 Fast RLS and Lattice-Ladder RLS

For long FIR filters, conventional RLS can be too expensive. Fast RLS algorithms exploit the shift structure of FIR input vectors.

Important families include:

- fast transversal filters;
- FAEST algorithms;
- RLS lattice-ladder filters;
- QR-based RLS lattice algorithms.

> ![Figure 5.5](./CourseADSP2026/Fig/Chapter_7/fig_5_5_textbook_fig_10_40_p590.png)
>
> *Figure 6.5 (Textbook Fig. 10.40, p. 590): Classification of RLS algorithms for array processing and FIR filtering.*

The algorithm hierarchy can be understood by asking two questions.

First, is the algorithm designed for a general linear combiner or for the special shift structure of an FIR filter?

Second, does the implementation prioritize algebraic simplicity, computational speed, numerical robustness, or parallel hardware structure?

## 6.10 LMS versus RLS: The Main Comparison

| Feature | LMS | RLS |
|---------|-----|-----|
| Cost per sample | $O(M)$ | Usually $O(M^2)$ for conventional RLS |
| Memory | Very small | Stores $\mathbf{P}(n)$ or matrix factors |
| Convergence speed | Can be slow for colored inputs | Usually fast |
| Sensitivity to eigenvalue spread | High | Low |
| Numerical issues | Mild | More serious unless stabilized |
| Tracking control | Step size $\mu$ | Forgetting factor $\lambda$ |
| Best use case | Low-cost, robust real-time adaptation | Fast convergence, difficult correlation structure |

A simple rule is:

> Use LMS when simplicity and robustness are more important than fastest convergence. Use RLS when fast convergence is important and the computational cost is acceptable.

---

# §7 Tracking, Algorithm Selection, and Figure Checklist

> 📖 Textbook §10.8 (Tracking Performance of Adaptive Algorithms)

## 7.1 Why Tracking Is Different from Convergence

Convergence analysis assumes a fixed optimum coefficient vector $\mathbf{c}_o$. Tracking analysis assumes a moving optimum $\mathbf{c}_o(n)$.

A useful model is

$$
\mathbf{c}_o(n)=\rho\mathbf{c}_o(n-1)+\boldsymbol{\psi}(n),
$$

where $\boldsymbol{\psi}(n)$ represents random parameter drift. When $\rho=1$, this is a random-walk model.

Tracking has a different tradeoff from stationary convergence. An algorithm must use recent data strongly enough to follow changes, but not so strongly that coefficient noise becomes excessive.

## 7.2 Local Statistics: Exponential and Sliding Windows

> ![Figure 6.1](./CourseADSP2026/Fig/Chapter_7/fig_6_1_textbook_fig_10_41_p591.png)
>
> *Figure 7.1 (Textbook Fig. 10.41, p. 591): Exponentially growing and fixed-length sliding windows. Tracking requires local rather than global statistics.*

For a nonstationary SOE, global statistics are not meaningful because old data may describe an outdated system.

There are three common approaches.

| Approach | Mechanism | Example |
|----------|-----------|---------|
| Constant step-size stochastic gradient | Never fully stop adapting | LMS |
| Exponential forgetting | Downweight old data | RLS with $\lambda<1$ |
| Sliding window | Use only recent samples | Sliding-window RLS |

## 7.3 Tracking Model

> ![Figure 6.2](./CourseADSP2026/Fig/Chapter_7/fig_6_2_textbook_fig_10_42_p595.png)
>
> *Figure 7.2 (Textbook Fig. 10.42, p. 595): Setup and model used for analysis of adaptive tracking.*

A common tracking performance metric is the mean-square deviation

$$
D(n)=E\{\|\mathbf{c}(n)-\mathbf{c}_o(n)\|^2\}.
$$

Another is misadjustment, which measures excess output error relative to the irreducible error floor.

The degree of nonstationarity compares the power introduced by the changing optimum filter to the irreducible error power. When the optimum changes quickly, the algorithm needs a shorter memory or larger step size.

## 7.4 Matched, Slow, and Fast Adaptation

The textbook illustrates three qualitative regimes.

| Regime | Description | Practical Symptom |
|--------|-------------|-------------------|
| Slow adaptation | Step size too small or memory too long | Filter lags behind changing optimum |
| Matched adaptation | Adaptation roughly matches environment speed | Good tracking with moderate fluctuations |
| Fast adaptation | Step size too large or memory too short | Filter follows quickly but coefficient noise is high |

> ![Figure 6.3](./CourseADSP2026/Fig/Chapter_7/fig_6_3_textbook_fig_10_43_p600.png)
>
> *Figure 7.3 (Textbook Fig. 10.43, p. 600): Matched LMS adaptation for slowly time-varying parameters.*

> ![Figure 6.4](./CourseADSP2026/Fig/Chapter_7/fig_6_4_textbook_fig_10_44_p600.png)
>
> *Figure 7.4 (Textbook Fig. 10.44, p. 600): LMS learning curves under matched adaptation.*

The matched LMS case shows a reasonable compromise. The coefficients follow the changing optimum without excessive fluctuations.

> ![Figure 6.5](./CourseADSP2026/Fig/Chapter_7/fig_6_5_textbook_fig_10_49_p604.png)
>
> *Figure 7.5 (Textbook Fig. 10.49, p. 604): Matched RLS adaptation for slowly time-varying parameters.*

> ![Figure 6.6](./CourseADSP2026/Fig/Chapter_7/fig_6_6_textbook_fig_10_50_p605.png)
>
> *Figure 7.6 (Textbook Fig. 10.50, p. 605): RLS learning curves under matched adaptation.*

RLS uses the forgetting factor $\lambda$ instead of the LMS step size $\mu$ as the main tracking-control parameter. A smaller $\lambda$ gives faster tracking but larger steady-state fluctuations.

## 7.5 Practical Algorithm Selection

The following decision table is useful for applications.

| Situation | Recommended Starting Point | Reason |
|----------|----------------------------|--------|
| Low-cost hardware, moderate accuracy | LMS | Minimal arithmetic and memory |
| Input power varies strongly | NLMS | Normalizes update by input energy |
| Input is highly correlated | Transform-domain LMS, affine projection, or RLS | Reduces eigenvalue-spread problem |
| Fast initial convergence is critical | RLS | Much faster than LMS in many colored-input settings |
| Very long echo path | Block LMS or frequency-domain adaptive filtering | FFT acceleration |
| Numerical robustness is critical | QR-RLS or square-root RLS | Maintains stable matrix factors |
| Rapidly changing system | Larger LMS step size or smaller RLS forgetting factor | Shorter effective memory |
| Slowly changing system | Smaller LMS step size or $\lambda$ close to 1 | Lower steady-state EMSE |

## 7.6 Common Implementation Pitfalls

### Pitfall 1: Step Size Chosen Without Input Power

A step size that works for one input amplitude can fail for another. Always relate $\mu$ to input power or use NLMS.

### Pitfall 2: Forgetting Factor Too Close to One in a Nonstationary System

If $\lambda$ is too close to one, RLS uses data that are too old. The filter may converge well in a stationary experiment but fail to track real changes.

### Pitfall 3: Insufficient Regularization in RLS Initialization

The initialization

$$
\mathbf{P}(-1)=\delta^{-1}\mathbf{I}
$$

must be chosen carefully. If $\delta$ is too small, the initial gain can be very large. If it is too large, adaptation starts too conservatively.

### Pitfall 4: Using a Desired Response That Is Not Available in Deployment

Supervised adaptation requires $y(n)$. In communication systems, $y(n)$ may be available during a training period but not during normal data transmission. Decision-directed adaptation can be used later, but decision errors can destabilize adaptation.

### Pitfall 5: Treating a Noisy Learning Curve as Failure

LMS uses random instantaneous gradients. Individual learning curves are noisy by nature. Performance should often be assessed by averaging or by examining steady-state error statistics.

## 7.7 Chapter Summary

The main concepts of the chapter are as follows.

1. Adaptive filters update coefficients from data, making them useful when statistics are unknown or time varying.
2. The Wiener filter remains the ideal reference solution in a stationary environment.
3. Steepest descent explains the role of the MSE surface, eigenvalue spread, and step-size stability.
4. LMS replaces the exact gradient with an instantaneous gradient estimate.
5. LMS is simple and robust but can be slow for colored inputs.
6. NLMS, transform-domain LMS, block LMS, affine projection, leakage, and variable-step methods address practical limitations of basic LMS.
7. RLS minimizes a recursively updated weighted least-squares criterion and usually converges much faster than LMS.
8. RLS costs more and requires careful numerical implementation.
9. Tracking nonstationary systems requires local statistics: LMS uses constant adaptation, while RLS uses forgetting or sliding windows.
10. Every adaptive algorithm involves a tradeoff among convergence speed, steady-state error, tracking ability, computational cost, and numerical robustness.

## 7.8 Method Comparison at a Glance

| Method | Update Information | Main Tuning Parameter | Complexity | Strength | Weakness |
|--------|--------------------|-----------------------|------------|----------|----------|
| Steepest descent | Exact $\mathbf{R}$ and $\mathbf{d}$ | $\mu$ | Matrix-vector product | Clear theory | Requires known statistics |
| LMS | Current $\mathbf{x}(n)$ and $e(n)$ | $\mu$ | $O(M)$ | Simple, robust | Slow for colored inputs |
| NLMS | Current $\mathbf{x}(n)$, $e(n)$, and input norm | $\tilde\mu$ | $O(M)$ | Handles input-power variation | Still affected by correlation |
| Transform-domain LMS | Transformed input | Mode-dependent gains | More than LMS | Faster for colored inputs | Needs transform/power estimates |
| Affine projection | Recent $K$ input vectors | Projection order $K$ | Higher than LMS | Faster for correlated inputs | Matrix solve per update |
| RLS | All past data with forgetting | $\lambda$, $\delta$ | $O(M^2)$ | Fast convergence | More complex, numerical care |
| QR-RLS | Factorized LS problem | $\lambda$, $\delta$ | High | Numerically stable | More implementation effort |

## 7.9 Figure Checklist

All figures used in this lecture were cropped from the uploaded textbook PDF and saved in `./CourseADSP2026/Fig/Chapter_7/`.

| Lecture Figure | Textbook Figure | File |
|---------------|-----------------|------|
| Figure 0.1 | Fig. 10.3, p. 501 | `fig_0_1_textbook_fig_10_3_p501.png` |
| Figure 0.2 | Fig. 10.5, p. 505 | `fig_0_2_textbook_fig_10_5_p505.png` |
| Figure 0.3 | Fig. 10.6, p. 506 | `fig_0_3_textbook_fig_10_6_p506.png` |
| Figure 1.1 | Fig. 10.7, p. 507 | `fig_1_1_textbook_fig_10_7_p507.png` |
| Figure 1.2 | Fig. 10.8, p. 508 | `fig_1_2_textbook_fig_10_8_p508.png` |
| Figure 1.3 | Fig. 10.9, p. 509 | `fig_1_3_textbook_fig_10_9_p509.png` |
| Figure 1.4 | Fig. 10.10, p. 510 | `fig_1_4_textbook_fig_10_10_p510.png` |
| Figure 1.5 | Fig. 10.11, p. 512 | `fig_1_5_textbook_fig_10_11_p512.png` |
| Figure 1.6 | Fig. 10.12, p. 515 | `fig_1_6_textbook_fig_10_12_p515.png` |
| Figure 2.1 | Fig. 10.13, p. 518 | `fig_2_1_textbook_fig_10_13_p518.png` |
| Figure 2.2 | Fig. 10.14, p. 521 | `fig_2_2_textbook_fig_10_14_p521.png` |
| Figure 2.3 | Fig. 10.15, p. 522 | `fig_2_3_textbook_fig_10_15_p522.png` |
| Figure 2.4 | Fig. 10.16, p. 523 | `fig_2_4_textbook_fig_10_16_p523.png` |
| Figure 3.1 | Fig. 10.17, p. 525 | `fig_3_1_textbook_fig_10_17_p525.png` |
| Figure 3.2 | Fig. 10.18, p. 526 | `fig_3_2_textbook_fig_10_18_p526.png` |
| Figure 3.3 | Fig. 10.19, p. 527 | `fig_3_3_textbook_fig_10_19_p527.png` |
| Figure 3.4 | Fig. 10.20, p. 537 | `fig_3_4_textbook_fig_10_20_p537.png` |
| Figure 3.5 | Fig. 10.21, p. 538 | `fig_3_5_textbook_fig_10_21_p538.png` |
| Figure 5.1 | Fig. 10.22, p. 539 | `fig_3_6_textbook_fig_10_22_p539.png` |
| Figure 5.2 | Fig. 10.23, p. 539 | `fig_3_7_textbook_fig_10_23_p539.png` |
| Figure 5.3 | Fig. 10.24, p. 541 | `fig_3_8_textbook_fig_10_24_p541.png` |
| Figure 5.4 | Fig. 10.25, p. 542 | `fig_3_9_textbook_fig_10_25_p542.png` |
| Figure 5.5 | Fig. 10.26, p. 543 | `fig_3_10_textbook_fig_10_26_p543.png` |
| Figure 5.6 | Fig. 10.27, p. 544 | `fig_3_11_textbook_fig_10_27_p544.png` |
| Figure 5.7 | Fig. 10.28, p. 544 | `fig_3_12_textbook_fig_10_28_p544.png` |
| Figure 5.8 | Fig. 10.29, p. 545 | `fig_3_13_textbook_fig_10_29_p545.png` |
| Figure 4.1 | Fig. 10.30, p. 547 | `fig_4_1_textbook_fig_10_30_p547.png` |
| Figure 4.2 | Fig. 10.31, p. 547 | `fig_4_2_textbook_fig_10_31_p547.png` |
| Figure 6.1 | Fig. 10.32, p. 549 | `fig_5_1_textbook_fig_10_32_p549.png` |
| Figure 6.2 | Fig. 10.33, p. 552 | `fig_5_2_textbook_fig_10_33_p552.png` |
| Figure 6.3 | Fig. 10.34, p. 559 | `fig_5_3_textbook_fig_10_34_p559.png` |
| Figure 6.4 | Fig. 10.35, p. 559 | `fig_5_4_textbook_fig_10_35_p559.png` |
| Figure 6.5 | Fig. 10.40, p. 590 | `fig_5_5_textbook_fig_10_40_p590.png` |
| Figure 7.1 | Fig. 10.41, p. 591 | `fig_6_1_textbook_fig_10_41_p591.png` |
| Figure 7.2 | Fig. 10.42, p. 595 | `fig_6_2_textbook_fig_10_42_p595.png` |
| Figure 7.3 | Fig. 10.43, p. 600 | `fig_6_3_textbook_fig_10_43_p600.png` |
| Figure 7.4 | Fig. 10.44, p. 600 | `fig_6_4_textbook_fig_10_44_p600.png` |
| Figure 7.5 | Fig. 10.49, p. 604 | `fig_6_5_textbook_fig_10_49_p604.png` |
| Figure 7.6 | Fig. 10.50, p. 605 | `fig_6_6_textbook_fig_10_50_p605.png` |

---

## 7.10 Suggested Teaching Flow

For a classroom lecture, the chapter can be taught in four conceptual passes.

### Pass 1: Motivation and Architecture

Start from Wiener filtering and ask why it is insufficient in changing environments. Then introduce Figures 1.1–1.6 to establish the architecture, supervised error, a priori/a posteriori timing, and stationary/nonstationary modes.

### Pass 2: LMS from the MSE Surface

Use steepest descent to explain the MSE surface, eigenvalue spread, and stability. Then replace exact gradients by instantaneous gradients to derive LMS. Emphasize that LMS is noisy but simple.

### Pass 3: Practical LMS Engineering

Discuss step-size selection, EMSE, misadjustment, NLMS, transform-domain LMS, block LMS, and application examples. Echo cancelation and equalization are the best applications for intuition.

### Pass 4: RLS and Tracking

Introduce RLS as weighted least squares with recursive inverse-correlation updates. Compare LMS and RLS. Finish with tracking: local statistics, forgetting factors, and the matched/slow/fast adaptation tradeoff.
