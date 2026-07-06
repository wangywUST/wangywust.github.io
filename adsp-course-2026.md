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

  .lecture-toggle > summary::-webkit-details-marker {
    display: none;
  }

  .lecture-toggle > summary::before {
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

  .lecture-toggle[open] > summary::before {
    content: "-";
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
  }
</style>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 1: Discrete-Time Signal Processing Undergraduate DSP Review</summary>

{% include_relative CourseADSP2026/lecture_1.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 2: Analysis of Discrete Random Signals</summary>

{% include_relative CourseADSP2026/lecture_2.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 3: Linear Prediction and Lattice Filters</summary>

{% include_relative CourseADSP2026/lecture_3.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 4: Linear Signal Modeling and Parametric Spectral Estimation</summary>

{% include_relative CourseADSP2026/lecture_4.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 5: Power Spectrum Estimation</summary>

{% include_relative CourseADSP2026/lecture_5.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 6: Optimum Linear Filtering, Wiener Filters and Kalman Filters</summary>

{% include_relative CourseADSP2026/lecture_6.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>Lecture 7: Adaptive Filters, LMS, RLS, and Tracking</summary>

{% include_relative CourseADSP2026/lecture_7.md %}

</details>
