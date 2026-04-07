# Lecture 9: How to Make an Academic Conference Poster

A practical guide for researchers presenting at conferences.

---

## 1. Understand the Purpose

A poster is a **visual conversation starter**, not a paper on a wall. Your goal is to:

- Communicate your core contribution in 30 seconds
- Give attendees enough detail to ask good questions
- Invite dialogue — you are there to explain, not just display

---

## 2. Know the Constraints Before You Design

Before opening any design tool, confirm with the conference:

| Constraint | Typical Value | Why It Matters |
|---|---|---|
| Poster size | A0 (841 × 1189 mm) or 36" × 48" | Determines font sizes and layout |
| Orientation | Portrait or Landscape | Landscape ≠ default |
| File format | PDF / PNG | Print resolution ≥ 300 DPI |
| Mounting method | Pins or velcro | Affects margins |

---

## 3. Content Structure

A proven layout:


<div style="text-align: center;">
  <img src="./CourseNLP2026/poster_example.jpg" width="100%">
  <p style="margin-top: 10px;">An example from https://github.com/zhoubolei/bolei_awesome_posters</p>
</div>

### Section-by-section guide

**Title bar**
- Title: ≤ 12 words, readable from 3 meters
- Include your institution logo, QR code to paper/project page
- Font size: title ≥ 72pt, authors ≥ 36pt

**Motivation / Problem**
- One paragraph or 3 bullets max
- State the gap your work addresses
- Add a teaser figure if possible

**Method**
- Use a pipeline/architecture diagram, not prose
- Label every component clearly
- A reader should grasp the approach in 60 seconds

**Results**
- Lead with your best result (bar charts, tables, or qualitative examples)
- Bold or highlight the number you want people to remember
- Avoid tables with >6 columns — use figures instead

**Conclusion & Future Work**
- 3–5 bullet points only
- State the takeaway message in one sentence

**References & QR Code**
- Keep references to ≤ 5 key citations
- Include a QR code linking to: paper PDF, project page, or GitHub

---

## 4. Visual Design Principles

### Layout
- Use a **3-column grid** for portrait; **2-row grid** for landscape
- Leave at least 10% of total area as whitespace — crowded posters repel readers
- Align everything to a grid; misaligned boxes look unprofessional

### Typography
- Body text: **≥ 24pt** (readable at arm's length)
- Section headers: **≥ 36pt**
- Title: **≥ 72pt**
- Use at most **2 fonts**: one sans-serif for body, optionally one for display/title
- Good free choices: Inter, Source Sans 3, Lato, Noto Sans

### Color
- Choose a **primary color** (from your institution palette or the conference theme)
- Use it for section headers and key callouts only
- Keep background white or very light gray
- Ensure sufficient contrast (WCAG AA: 4.5:1 ratio for body text)
- Avoid red/green combinations (color blindness)

### Figures
- Every figure must have a **caption** (1–2 sentences)
- Export figures at **≥ 300 DPI**; vector (SVG/PDF) is better
- Prefer simple, clean plots over complex multi-panel arrangements
- Use consistent color coding across all figures

---

## 5. Recommended Tools

| Tool | Best For | Cost |
|---|---|---|
| **PowerPoint / Keynote** | Beginners, quick turnaround | Free (institutional) |
| **Adobe Illustrator** | Pixel-perfect control | Paid |
| **Canva** | Fast, template-based design | Free tier available |
| **Inkscape** | Vector editing, open source | Free |
| **LaTeX + beamerposter** | Programmatic control, academic styling | Free |
| **Figma** | Collaborative design | Free tier available |

> **Tip for LaTeX users**: Use the `beamerposter` package with a custom theme. Version-control your poster source alongside your paper.

---

## 6. Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| Too much text | Replace paragraphs with bullets and diagrams |
| Font too small | Never go below 24pt for body |
| No clear hierarchy | Use size + color to signal importance |
| Figures exported at 72 DPI | Always export at 300 DPI minimum |
| No QR code | Add one linking to paper or project page |
| Ignoring whitespace | Leave breathing room between sections |
| Equations everywhere | Move math to the paper; show intuition on poster |
| Printing last-minute | Send to print **3+ days** before the conference |

---

## 7. Printing Checklist

Before sending to the printer:

- [ ] Confirm poster dimensions match conference requirements
- [ ] Export as PDF with fonts embedded
- [ ] Resolution ≥ 300 DPI for all raster images
- [ ] Bleed margins set if required by printer
- [ ] Color profile: CMYK for professional printing, RGB for in-house
- [ ] Proofread title, author names, and affiliations
- [ ] QR codes are scannable at printed size
- [ ] Bring a backup copy on USB / cloud storage

---

## 8. During the Session

- Prepare a **60-second verbal pitch** — practice it
- Stand **beside** (not in front of) your poster
- Bring printed handouts or business cards with your QR code
- Engage passersby with a question: *"Are you working on anything related to X?"*
- Take photos of interested attendees' questions — they are future paper ideas

---

## Quick Reference: Font Size Guide

| Element | Minimum Size |
|---|---|
| Main title | 72 pt |
| Author names | 36 pt |
| Section headers | 36 pt |
| Body text | 24 pt |
| Figure captions | 20 pt |
| References | 18 pt |

---

*Good luck at your conference. A great poster is a great conversation.*