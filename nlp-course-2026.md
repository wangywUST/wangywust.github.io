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

## Coursework

1. In-Course Question Answering
2. Final Projects

### In-Course Question Answering

In each class, students will be asked several questions. Each student may answer each question only once. The student who correctly answers a question first and explains the answer clearly to the class will be granted 1 credit. Final scores will be calculated based on accumulated credits throughout the semester. This score does not contribute to the final grade. At the end of the course, students who achieve high scores in question answering will be highlighted for recognition.

The timestamp for in-course question answering is determined by the message time in the Zoom chat. The main content of your answer should be conveyed in your Zoom chat message.

### Final Projects

Every student should complete a final project related to LLMs and present it in the final classes. A project report should be completed before the presentation and submitted to [Final Project Google Form](TBD) The file should be named as "Your Name.pdf" The format should follow the [2025 ACL long paper template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj). Good examples are at [Project Report Examples](https://aclanthology.org/events/acl-2024/#2024acl-long)

The final project scores will be calculated through 50% Instructor's rating + 50% TAs' rating.

### Useful Links

<!-- [Final Project Google Folder](https://drive.google.com/drive/folders/1foBTff-e2GpbRGu97B125TFwSLKJX54l?usp=sharing) -->

[2025 ACL long paper template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)

[Project Report Examples](https://aclanthology.org/events/acl-2024/#2024acl-long)

## Having Questions?

Please feel free to add your question, name, your lab session id, and your post date to the following spreadsheet: [Sheet of Students' Questions](
https://docs.google.com/spreadsheets/d/1x5WKGrvoD1TY_S0WXEkHZoE1vvwjZejS8p5ALCo_J7s/edit?usp=sharing)

## Lab Sessions?

This course does not have in-person lab sessions. Students are encouraged to use the lab session time to work remotely on their final projects.

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
| 10 | Lecture 10 | Vision-Language Models | • Multimodal LLMs<br>• Vision encoders (CLIP, ViT)<br>• Image captioning and VQA<br>• GPT-4V, MM1, Flamingo | • Understand vision-language model architectures<br>• Process images with text in unified models<br>• Apply VLMs to visual question answering<br>• Evaluate on multimodal benchmarks |

{% include_relative CourseNLP2026/lecture_1.md %}

{% include_relative CourseNLP2026/lecture_2.md %}

{% include_relative CourseNLP2026/lecture_3.md %}

{% include_relative CourseNLP2026/lecture_4.md %}

{% include_relative CourseNLP2026/lecture_5.md %}

{% include_relative CourseNLP2026/lecture_6.md %}

{% include_relative CourseNLP2026/lecture_7.md %}

{% include_relative CourseNLP2026/lecture_8.md %}