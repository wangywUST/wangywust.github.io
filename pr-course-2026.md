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

  .lecture-toggle:nth-of-type(1) > summary {
    background: #e8f1ff;
  }

  .lecture-toggle:nth-of-type(2) > summary {
    background: #e8f8f1;
  }

  .lecture-toggle:nth-of-type(3) > summary {
    background: #fff4df;
  }

  .lecture-toggle:nth-of-type(4) > summary {
    background: #f1ecff;
  }

  .lecture-toggle:nth-of-type(5) > summary {
    background: #ffeef2;
  }

  .lecture-toggle:nth-of-type(6) > summary {
    background: #e9f7fb;
  }

  .lecture-toggle:nth-of-type(7) > summary {
    background: #f2f6e8;
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

    .lecture-toggle:nth-of-type(1) > summary {
      background: #1e40af;
    }

    .lecture-toggle:nth-of-type(2) > summary {
      background: #166534;
    }

    .lecture-toggle:nth-of-type(3) > summary {
      background: #92400e;
    }

    .lecture-toggle:nth-of-type(4) > summary {
      background: #5b21b6;
    }

    .lecture-toggle:nth-of-type(5) > summary {
      background: #9f1239;
    }

    .lecture-toggle:nth-of-type(6) > summary {
      background: #155e75;
    }

    .lecture-toggle:nth-of-type(7) > summary {
      background: #3f6212;
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

{% include_relative CoursePR2026/lecture_1.md %}

</details>

<script>
  document.addEventListener("DOMContentLoaded", function () {
    const lectureToggles = Array.from(document.querySelectorAll(".lecture-toggle"));

    function showOnly(openLecture) {
      lectureToggles.forEach(function (lecture) {
        lecture.hidden = Boolean(openLecture && lecture !== openLecture);
      });
    }

    lectureToggles.forEach(function (lecture) {
      lecture.addEventListener("toggle", function () {
        if (lecture.open) {
          showOnly(lecture);
          return;
        }

        const openLecture = lectureToggles.find(function (item) {
          return item.open;
        });
        showOnly(openLecture);
      });
    });
  });
</script>
