---
layout: course
---

# CSE 188: Natural Language Processing

This is a new course at UC Merced starting in Spring 2026. In this course, students will learn the fundamentals of modeling, theory, ethics, and systems aspects of state-of-the-art natural language processing technologies, as well as gain hands-on experience working with them.

## The Goal of This Course

To offer useful, fundamental, and detailed NLP knowledge to students.

## Staff Members

Instructuor: Yiwei Wang (yiweiwang2@ucmerced.edu)

TA: Freeman Cheng (freemancheng@ucmerced.edu), Hang Wu (hangwu@ucmerced.edu)

## Course Modality

Either onsite or virtual attendance is acceptable. Regardless of whether you attend lectures in person, you are encouraged to join the course using the [Zoom link](https://ucmerced.zoom.us/my/ucmnlp). This is necessary for participating in the in-course question answering.

Office Hour: 2:00 PM - 3:00 PM, Every Tuesday at SE2 214.

## Coursework

1. In-Course Question Answering
2. Final Projects

### In-Course Question Answering

In each class, students will be asked several questions. Each student may answer each question only once. The student who correctly answers a question first and explains the answer clearly to the class will be granted 1 credit. Final scores will be calculated based on accumulated credits throughout the semester. This score does not contribute to the final grade. At the end of the course, students who achieve high scores in question answering will be highlighted for recognition.

The timestamp for in-course question answering is determined by the message time in the Zoom chat. The main content of your answer should be conveyed in your Zoom chat message.

### Final Projects

Every student is required to complete a final project related to **Large Language Models (LLMs)** and present it during the final classes using a poster. 

#### 1. Submission Requirements
* **Deadline:** The project report must be submitted via the [Final Project Google Form](https://forms.gle/5PkGoAaYzGFURGGH7) before **May 1st, 2026**.
* **File Naming:** Please name your file as `Last Name, First Name.pdf`.
* **Format:** Submissions must follow the [2025 ACL long paper template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj). 
* **Examples:** You can find high-quality examples at the [ACL 2024 Project Reports](https://aclanthology.org/events/acl-2024/#2024acl-long) archive.

#### 2. Grading Criteria
The final project grade will be determined by:
* **50%** Instructor’s rating
* **50%** TAs’ rating

> **Note:** Only the final project report will be graded and counted toward your final course grade.

---

#### 3. Poster Presentation (Optional)
Students are highly encouraged to showcase their work via a poster (in the .pdf format) on this [Google Sheet](https://docs.google.com/spreadsheets/d/16QGQerK6jN9rloWvCNW3-q85itust8PNno8VMJIAhV8/edit?usp=sharing). You can find the nice examples of conference posters at [Dr. Bolei Zhou's Github Repo](https://github.com/zhoubolei/bolei_awesome_posters/tree/main).

* **How to Participate:** You may add your poster link to the last row of the sheet at any time by filling in the columns before "Question 1." **Please do not overwrite existing rows; instead, add a new row at the bottom for each poster update.**
* **Peer Engagement:** Students are encouraged to review their peers' posters and post questions in the final columns (e.g., Question 1).
* **Interaction:** Poster owners are expected to answer these questions directly within the Google Sheet.
* **In-Class Discussion:** Starting April 13, 2026, **Yiwei Wang** will review the posters and their corresponding questions during each lecture. Both existing and new questions regarding project reports and posters are welcome for discussion with the class.

#### 4. Future Benefits
While the poster is optional and does not affect your grade, active and high-quality participation will be highly valued by Yiwei Wang and may serve as a basis for future **letters of recommendation**.

#### 5. Final Project Evaluation Criteria

To help you succeed in your final projects, the evaluation will be centered around two primary dimensions used by major NLP conferences (like ACL and EMNLP): **Soundness** and **Excitement**.

---

##### 1. Soundness (Technical Integrity & Rigor)
*Soundness evaluates whether your findings are trustworthy and scientifically valid.*

* **Experimental Rigor:** * Did you use appropriate baselines? (e.g., comparing your method against standard Zero-shot or Few-shot prompts).
    * Is the data size sufficient to support your claims?
* **Ablation Studies:** * If you proposed a multi-step method, did you test the system with specific components removed to prove they actually add value?
* **Error Analysis:** * A high-quality report doesn't just show numbers; it investigates *why* the model failed. Providing a qualitative analysis of incorrect samples is essential.
* **Metric Choice:** * Are the evaluation metrics (e.g., Accuracy, F1, ROUGE, or Human Eval) logically aligned with the task goals?

##### 2. Excitement (Innovation & Impact)
*Excitement evaluates the novelty and the "interestingness" of your research.*

* **Originality:** * Does the project go beyond a simple "plug-and-play" exercise? We value original insights into LLM behavior, new prompting strategies, or unique applications.
* **Clarity of Communication:** * Is the report written clearly using the ACL LaTeX template? 
    * Are the figures and tables intuitive and professional?
* **Task Difficulty:** * Tackling a complex or under-explored problem (e.g., reasoning in specialized domains or efficiency optimization) generates more "excitement" than repeating well-known benchmarks.


> **Instructor's Note:** Active engagement in the poster session—demonstrating either **soundness** in methodology or **excitement** for the topic—is a factor I consider when writing future **letters of recommendation**, although it will not influence the grading for this course.

#### Useful Links

[2025 ACL long paper template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)

[Project Report Examples](https://aclanthology.org/events/acl-2024/#2024acl-long)

## Having Questions?

Please feel free to add your question, name, your lab session id, and your post date to the following spreadsheet: [Sheet of Students' Questions](
https://docs.google.com/spreadsheets/d/1x5WKGrvoD1TY_S0WXEkHZoE1vvwjZejS8p5ALCo_J7s/edit?usp=sharing)

If your question is not answered within one week after your post the question, please email the corresponding TA and cc me to get support. Hang Wu is responsible for lab session 02L, 03L, 04L, and Freeman is for 05L.

## Lab Sessions?

Students are encouraged to use the lab session time to work on their final projects. The project-related questions are encouraged to ask the correponding TAs or the instructor for research discussions.

## Course Syllabus

Below is a comprehensive syllabus table for the course, organized by lecture topics, key concepts, and learning objectives.

| Week | Lecture | Topic | Key Concepts | Learning Objectives |
|------|---------|-------|--------------|---------------------|
| 1 | Lecture 1 | Overview of NLP | • What is language?<br>• Language models fundamentals<br>• Large language models introduction<br>• Historical evolution (SLM → NLM → PLM → LLM) | • Understand the definition and properties of language<br>• Learn probability distribution over token sequences<br>• Grasp the evolution from statistical to large language models<br>• Understand scaling laws and emergent abilities |
| 2 | Lecture 2 | Tokenization | • Word-based tokenization<br>• Character-based tokenization<br>• Subword tokenization<br>• Byte-Pair Encoding (BPE) | • Understand different tokenization approaches<br>• Master BPE training algorithm<br>• Implement BPE tokenization inference<br>• Practice tokenization on sample text |
| 3 | Lecture 3 | Transformer Architecture | • Encoder-Decoder structure<br>• Encoder-only vs Decoder-only<br>• Attention mechanism<br>• Multi-head attention<br>• Position-wise feed-forward networks | • Understand transformer architecture variants<br>• Master scaled dot-product attention computation<br>• Calculate multi-head attention with numerical examples<br>• Understand positional encoding |
| 4 | Lecture 4 | Model Analysis | • Parameter counting<br>• Memory usage estimation<br>• FLOPs computation<br>• Intermediate activations<br>• Training time estimation | • Calculate total parameters in transformer models<br>• Estimate memory requirements for training/inference<br>• Compute computational cost (FLOPs)<br>• Analyze activation recomputation trade-offs |
| 5 | Lecture 5 | Decoder-only Transformers | • Decoder-only vs vanilla transformer<br>• Autoregressive generation<br>• Architectural differences<br>• Modern LLM design | • Compare encoder-decoder and decoder-only architectures<br>• Understand masked self-attention<br>• Learn why decoder-only became mainstream<br>• Identify application scenarios for each architecture |
| 6 | Lecture 6 | Efficient Text Generation | • KV-Cache mechanism<br>• Prefill vs decoding phase<br>• Memory optimization<br>• Inference efficiency | • Understand KV-Cache concept and implementation<br>• Calculate KV-Cache memory requirements<br>• Compare inference with/without KV-Cache<br>• Optimize generation speed |
| 7 | Lecture 7 | Decoding Algorithms | • Greedy search<br>• Beam search<br>• Sampling methods (top-k, top-p)<br>• Temperature scaling | • Master various decoding strategies<br>• Understand trade-offs between methods<br>• Implement beam search algorithm<br>• Apply temperature for controlling randomness |
| 8 | Lecture 8 | Advanced LLM Capabilities (Part 1) | • Chain-of-Thought (CoT) prompting<br>• Step-by-step reasoning<br>• Self-consistency<br>• Zero-shot vs few-shot CoT | • Apply CoT prompting techniques<br>• Improve reasoning on complex tasks<br>• Understand GSM8K and ARC benchmarks<br>• Implement self-consistency for better accuracy |
| 9 | Lecture 9 | Advanced LLM Capabilities (Part 2) | • LLM agents and autonomy<br>• Tool use (Toolformer)<br>• ReAct framework<br>• Autonomous task completion | • Design LLM-based agents<br>• Integrate external tools with LLMs<br>• Implement ReAct reasoning+acting loop<br>• Understand agent architectures (AutoGPT, HuggingGPT) |

{% include_relative CourseNLP2026/lecture_1.md %}

{% include_relative CourseNLP2026/lecture_2.md %}

{% include_relative CourseNLP2026/lecture_3.md %}

{% include_relative CourseNLP2026/lecture_4.md %}

{% include_relative CourseNLP2026/lecture_5.md %}

{% include_relative CourseNLP2026/lecture_6.md %}

{% include_relative CourseNLP2026/lecture_7.md %}

{% include_relative CourseNLP2026/lecture_8.md %}

{% include_relative CourseNLP2026/lecture_9.md %}