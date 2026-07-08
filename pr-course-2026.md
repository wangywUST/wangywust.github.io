---
layout: course
---

<style>
  .lecture-page {
    margin: 1rem 0;
    padding: 1rem;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
  }

  .lecture-header {
    margin: 0 -1rem;
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #cbd5e1;
    border-radius: 8px 8px 0 0;
    background: #e8f1ff;
    color: #0f172a;
  }

  .lecture-title {
    display: block;
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.3;
  }

  .lecture-description {
    display: block;
    margin: 0.6rem 0 0;
    padding: 0.55rem 0.8rem;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    background: #f8fbff;
    color: #1e293b;
    font-weight: 400;
    line-height: 1.5;
  }

  @media (prefers-color-scheme: dark) {
    .lecture-page {
      border-color: #475569;
      background: #0f172a;
    }

    .lecture-header {
      border-bottom-color: #475569;
      background: #1e40af;
      color: #f8fafc;
    }

    .lecture-description {
      border-left-color: #60a5fa;
      background: #1e293b;
      color: #e2e8f0;
    }
  }
</style>

<section class="lecture-page" markdown="1">
<header class="lecture-header">
<span class="lecture-title">Lecture 1: Pattern Recognition and Machine Learning Introduction</span>
<span class="lecture-description">This lecture introduces the core viewpoint of PRML Chapter 1: probability as a language for reasoning under uncertainty, decision theory as a framework for making optimal choices, and information theory as a way to measure uncertainty and model comparison.</span>
</header>

{% include_relative CoursePR2026/lecture_1.md %}

</section>
