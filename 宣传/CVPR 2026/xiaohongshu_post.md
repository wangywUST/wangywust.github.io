# 小红书文案：CVPR 2026 工作介绍

## 标题备选

下周 CVPR 2026 见：我们带来 1 个 Tutorial + 4 篇工作

CVPR 2026 参会预告：多模态推理、Video LLM、Diffusion、视觉表征与声音生成

去 Denver 参加 CVPR 2026：这次我们关注从感知到模拟的多模态智能

## 正文

下周我们会在 Denver 参加 CVPR 2026。这次带来的内容很丰富：1 个 tutorial + 4 篇 paper，主题从多模态推理、Video LLM、diffusion generalization，到 object-centric representation 和 creative sound generation。

如果你也会在 CVPR，欢迎来听、来聊、来交换想法。

Tutorial

From Perception to Simulation: The Emergence of World Models in Multi-modal Reasoning

时间：June 3rd, 2026, 2 PM GMT-7

地点：Room 301/302, Colorado Convention Center

这个 tutorial 关注一个很核心的问题：多模态系统如何从“看见世界”走向“理解、预测并模拟世界”。我们会围绕 world models、multi-modal reasoning、perception to simulation 展开，讨论它们在视觉、语言、视频和智能体中的新进展。

Paper 1

EchoFoley: Event-Centric Hierarchical Control for Video Grounded Creative Sound Generation

关键词：Audio-visual generation, Creative Foley, Event control

这项工作关注视频配音/声音生成里的细粒度可控性。EchoFoley 不只是给视频“配一段声音”，而是希望能控制每个声音事件什么时候发生、是什么声音、以什么方式发生，让视频里的声音更服务于叙事和创作。

Paper 2

Finding Distributed Object-Centric Properties in Self-Supervised Transformers

关键词：Self-supervised ViTs, Object discovery, Representation analysis

这项工作重新审视 self-supervised Vision Transformers 中的 object-centric 信息。我们发现物体相关信息并不只集中在最后一层或某个 [CLS] attention map，而是分布在不同层、不同 attention components 之中。基于这个观察，Object-DINO 可以在不额外训练的情况下提取更稳定的物体中心表征。

Paper 3

PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs

关键词：Video LLMs, Temporal encoding, Training-free stabilizer

Video LLM 很容易受到帧采样和时间位置编码扰动的影响。PAS 是一个 training-free、plug-and-play 的时间编码稳定器，通过平滑 temporal kernel 来降低 Video LLM 对小幅时间偏移的敏感性，在几乎不增加计算开销的情况下提升视频理解的鲁棒性。

Paper 4

Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance

关键词：Diffusion models, Generalization, Segmented guidance

这项工作从 weak-to-strong 的视角理解 diffusion guidance，并提出 SGG，把不同 guidance 方法的优势结合起来，改善 diffusion model 在复杂生成任务中的泛化能力。它既可以用于 inference-time guidance，也可以迁移到 training objective 中。

这次我们的关键词大概可以概括为：

多模态推理

世界模型

Video LLM 时间建模

自监督视觉表征

可控声音生成

Diffusion 泛化

下周 CVPR 2026 Denver 见。对这些方向感兴趣的朋友，欢迎现场交流。

#CVPR2026 #计算机视觉 #多模态学习 #VideoLLM #DiffusionModels #WorldModels #视觉表征 #生成式AI #学术会议 #AI研究

## 配图建议

首图：使用 promo.html 导出的总览海报。

后续图：依次放 tutorial 宣传图、EchoFoley、Object-DINO、PAS、SGG 的论文缩略图。

如果只能发一张图：优先用总览海报。
