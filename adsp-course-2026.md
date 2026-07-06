---
layout: course
---

<style>
  .lecture-toggle {
    margin: 1rem 0;
    padding: 0 1rem 1rem;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
  }

  .lecture-toggle:not([open]) {
    padding-bottom: 0;
  }

  .lecture-toggle > summary {
    display: block;
    margin: 0 -1rem;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    background: #e8f1ff;
    color: #0f172a;
    cursor: pointer;
    font-weight: 700;
    list-style: none;
  }

  .lecture-title {
    display: block;
  }

  .lecture-toggle > summary::-webkit-details-marker {
    display: none;
  }

  .lecture-title::before {
    content: "+";
    display: inline-block;
    width: 1.2em;
    margin-right: 0.35em;
    font-weight: 700;
  }

  .lecture-toggle[open] > summary {
    margin-bottom: 1rem;
    border-bottom: 1px solid #cbd5e1;
    border-radius: 8px 8px 0 0;
  }

  .lecture-toggle[open] .lecture-title::before {
    content: "-";
  }

  .lecture-intro {
    display: block;
    margin: 0.5rem 0 0 1.55rem;
    padding: 0.55rem 0.8rem;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    background: #f8fbff;
    color: #1e293b;
    font-weight: 400;
    line-height: 1.5;
  }

  @media (prefers-color-scheme: dark) {
    .lecture-toggle {
      border-color: #475569;
      background: #0f172a;
    }

    .lecture-toggle > summary {
      background: #1e40af;
      color: #f8fafc;
    }

    .lecture-intro {
      border-left-color: #60a5fa;
      background: #1e293b;
      color: #e2e8f0;
    }
  }
</style>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 1: Discrete-Time Signal Processing Undergraduate DSP Review</span><span class="lecture-intro">This lecture reviews the core tools of undergraduate digital signal processing, including discrete-time signals, transforms, systems, filtering, and stability. It prepares the notation and intuition needed for the statistical and adaptive methods used later in the course.</span></summary>

{% include_relative CourseADSP2026/lecture_1.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 2: Analysis of Discrete Random Signals</span><span class="lecture-intro">This lecture introduces random signals from a signal-processing perspective, focusing on stationarity, correlation, covariance, power spectra, and spectral factorization. These ideas provide the probabilistic foundation for prediction, estimation, and optimum filtering.</span></summary>

{% include_relative CourseADSP2026/lecture_2.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 3: Linear Prediction and Lattice Filters</span><span class="lecture-intro">This lecture studies how future or missing signal samples can be estimated from past observations using linear prediction. It also develops lattice filter structures and related algorithms that are central to speech modeling, spectral analysis, and adaptive filtering.</span></summary>

{% include_relative CourseADSP2026/lecture_3.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 4: Linear Signal Modeling and Parametric Spectral Estimation</span><span class="lecture-intro">This lecture connects random signal behavior with compact parametric models such as AR, MA, and ARMA processes. It emphasizes how model assumptions lead to practical spectral estimation methods and how residual analysis helps judge model quality.</span></summary>

{% include_relative CourseADSP2026/lecture_4.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 5: Power Spectrum Estimation</span><span class="lecture-intro">This lecture focuses on estimating power spectra from finite data records. It compares nonparametric and model-based approaches, highlighting the trade-offs among resolution, variance, bias, data length, and prior modeling assumptions.</span></summary>

{% include_relative CourseADSP2026/lecture_5.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 6: Optimum Linear Filtering, Wiener Filters and Kalman Filters</span><span class="lecture-intro">This lecture develops optimum linear filtering as a framework for estimation, denoising, prediction, and tracking. It covers Wiener filtering and Kalman filtering, showing how second-order statistics and state-space models lead to principled signal estimates.</span></summary>

{% include_relative CourseADSP2026/lecture_6.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary><span class="lecture-title">Lecture 7: Adaptive Filters, LMS, RLS, and Tracking</span><span class="lecture-intro">This lecture explains how filters can learn and update from streaming data when signal statistics are unknown or changing. It introduces adaptive filtering algorithms such as LMS and RLS, with attention to convergence, stability, tracking, and practical implementation.</span></summary>

{% include_relative CourseADSP2026/lecture_7.md %}

</details>
