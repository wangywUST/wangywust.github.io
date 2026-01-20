<!-- # Calculating Parameters in the Vanilla Transformer Model

Looking at Table 3 in the "Attention Is All You Need" paper, I'll explain how to calculate the number of trainable parameters for the Transformer architecture. This calculation is important for understanding model complexity and computational requirements.

## Basic Transformer Architecture

The Transformer model consists of an encoder and decoder, each with:
- Multi-head attention mechanisms
- Feed-forward neural networks
- Layer normalization components
- Embedding layers and positional encodings

## Parameter Calculation Formula

The total parameters in a Transformer model can be calculated by summing:
1. Embedding parameters
2. Encoder parameters
3. Decoder parameters
4. Output layer parameters

## Parameter Breakdown for Base Model

Let's use the base configuration from the paper:
- N = 6 (number of encoder/decoder layers)
- dmodel = 512 (embedding dimension)
- dff = 2048 (feed-forward layer dimension)
- h = 8 (number of attention heads)
- dk = dv = 64 (key and value dimensions per head)

### 1. Embedding Layer Parameters
- Input embedding: Vocabulary_size × dmodel
- Positional encoding: Not counted (fixed sinusoidal, not learned)
- For vocabulary size of approximately 37,000 tokens:
  - 37,000 × 512 = 18,944,000 parameters

### 2. Encoder Layer Parameters (per layer)
- Multi-head attention:
  - Query, key, value projections: 3 × (dmodel × dmodel) = 3 × 512 × 512
  - Output projection: dmodel × dmodel = 512 × 512
- Feed-forward network:
  - First layer: dmodel × dff = 512 × 2048
  - Second layer: dff × dmodel = 2048 × 512
- Layer normalization (2 sets):
  - Scale and bias: 2 × 2 × dmodel = 2 × 2 × 512

Total per encoder layer:
```
(3 × 512 × 512) + (512 × 512) + (512 × 2048) + (2048 × 512) + (2 × 2 × 512)
= 786,432 + 262,144 + 1,048,576 + 1,048,576 + 2,048
= 3,147,776 parameters
```

For 6 encoder layers: 6 × 3,147,776 = 18,886,656 parameters

### 3. Decoder Layer Parameters (per layer)
- Similar to encoder but with an additional multi-head attention layer:
  - Self-attention: Same as encoder attention
  - Encoder-decoder attention: Same as encoder attention
  - Feed-forward and normalization: Same as encoder

Total per decoder layer: Encoder layer parameters + additional attention layer
```
3,147,776 + 786,432 + 262,144 + 2,048 (for additional normalization)
= 4,198,400 parameters
```

For 6 decoder layers: 6 × 4,198,400 = 25,190,400 parameters

### 4. Output Linear Layer
- Linear projection to vocabulary: dmodel × Vocabulary_size
- 512 × 37,000 = 18,944,000 parameters

### 5. Parameter Sharing
- The paper mentions weight sharing between embedding layers and output projection
- This reduces the total by approximately 18,944,000 parameters

## Total Parameter Count
```
18,944,000 (embeddings) + 18,886,656 (encoder) + 25,190,400 (decoder) + 0 (output, due to sharing)
≈ 63,021,056 parameters
```

This closely matches the approximately 65 million parameters reported for the base model in Table 3.

## Variations

The paper explores various configurations by changing:
- Number of layers (N)
- Model dimension (dmodel)
- Feed-forward dimension (dff)
- Number of attention heads (h)
- Key/value dimensions (dk, dv)

For example, reducing N from 6 to 2 reduces the parameter count to approximately 36 million, while increasing dmodel to 1024 and using 16 heads increases it to about 213 million for the "big" model.

The parameter calculation principles remain the same across all variations; you just substitute the new dimensions into the formulas above. -->

# Lecture 4: Analysis of Transformer Models: Parameter Count, Computation, Activations

In-Class Question 1: Given layer number $N$ as 6, model dimension $d_{model}$ as 512, feed-forward dimension $d_{ff}$ = 2048, number of attention heads $h$ = 8, what is the total number of learnable parameters in a vanilla Transformer model?

In-Class Question 2: Given layer number $N$ as 6, model dimension $d_{model}$ as 1024, feed-forward dimension $d_{ff}$ = 4096, number of attention heads $h$ = 16, what is the total number of learnable parameters in a vanilla Transformer model?

Reference Tutorial: [Parameter size of vanilla transformer](https://colab.research.google.com/drive/1jdhD6yexq2PBZYg9zg5FG-YRGyL8ZJ0L?usp=sharing)
Reference Tutorial: [Analysis of Transformer Models](https://zhuanlan.zhihu.com/p/624740065)

---

## 1. Introduction

Welcome to this expanded class on analyzing the memory and computational efficiency of training large language models (LLMs). With the rise of models like OpenAI’s ChatGPT, researchers and engineers have become increasingly interested in the mechanics behind Large Language Models. The “large” aspect of these models refers both to the **number of model parameters** and the **scale of training data**. For example, GPT-3 has 175 billion parameters and was trained on 570 GB of data. Consequently, training such models presents two key challenges: **memory efficiency** and **computational efficiency**.

Most large models in industry today utilize the **transformer architecture**. Their structures can be broadly divided into encoder-decoder (exemplified by T5) and decoder-only. The decoder-only structure can be split into *Causal LM* (represented by the GPT series) and *Prefix LM* (represented by GLM). Causal language models like GPT have achieved significant success, so many mainstream LLMs employ the Causal LM paradigm. In this class, we will focus on the decoder-only transformer framework, analyzing its parameter count, computational requirements, and intermediate activations to better understand the memory and computational efficiency of training and inference.

To make the analysis clearer, let us define the following notation:

- $l$: Number of transformer layers  
- $h$: Hidden dimension  
- $a$: Number of attention heads  
- $V$: Vocabulary size  
- $b$: Training batch size  
- $s$: Sequence length  

---

## 2. Model Parameter Count

A transformer model commonly consists of $l$ identical layers, each containing a **self-attention block** and an **MLP block**. The decoder-only structure also includes an **embedding layer** and a **final output layer** (often weight-tied with the embedding).

### 2.1 Parameter Breakdown per Layer

1. **Self-Attention Block**  
   The trainable parameters here include:
   - Projection matrices for queries, keys, and values: $W_Q, W_K, W_V \in \mathbb{R}^{h \times h}$  
   - Output projection matrix: $W_O \in \mathbb{R}^{h \times h}$  
   - Their corresponding bias vectors (each in $\mathbb{R}^{h}$)

   Hence, the parameter count in self-attention is:
   $$
   3(h \times h) + (h \times h) + \text{(4 biases)} = 4h^2 + 4h.
   $$
   However, in multi-head attention, we often split $h$ into $a$ heads, each of dimension $h/a$. Internally, $W_Q, W_K, W_V$ can be viewed as $[h, a\times (h/a)] = [h, h]$, so the total dimension is still $h\times h$. This is why the simpler $h^2$ counting still holds.

2. **MLP Block**  
   Usually, the MLP block has two linear layers:
   - First layer: $W_1 \in \mathbb{R}^{h \times (4h)}$ and bias in $\mathbb{R}^{4h}$
   - Second layer: $W_2 \in \mathbb{R}^{(4h) \times h}$ and bias in $\mathbb{R}^{h}$

   Therefore, the MLP block has:
   $$
   h \times (4h) + (4h) \;+\; (4h)\times h + h \;=\; 8h^2 + 5h
   $$
   parameters in total.

3. **Layer Normalization**  
   Both the self-attention and MLP blocks have a layer normalization containing a scaling parameter $\gamma$ and a shifting parameter $\beta$ in $\mathbb{R}^{h}$. So two layer norms contribute $4h$ parameters:
   $$
   2 \times (h + h) = 4h.
   $$

Summing these, each transformer layer has:
$$
(4h^2 + 4h) + (8h^2 + 5h) + 4h = 12h^2 + 13h
$$
trainable parameters.

4. **Embedding Layer**  
   There is a **word embedding** matrix in $\mathbb{R}^{V \times h}$, which contributes $Vh$ parameters. In many LLM implementations (such as GPT variants), this same matrix is shared with the final output projection for logits (output embedding). Hence the total parameters for input and output embeddings are typically counted as $Vh$ rather than $2Vh$.

If the position encoding is trainable, it might add a few more parameters, but often relative position encodings (e.g., RoPE, ALiBi) contain no trainable parameters. We will ignore any small parameter additions from positional encodings.

Thus, an $l$-layer transformer model has a total trainable parameter count of:
$$
l \times (12h^2 + 13h) + Vh.
$$

When $h$ is large, we can approximate $13h$ by a smaller term compared to $12h^2$, so the parameter count is roughly:
$$
12\,l\,h^2.
$$

### 2.2 Estimating LLaMA Parameter Counts

Below is a table comparing the approximate $12\,l\,h^2$ calculation for various LLaMA models to their actual parameter counts:

| Actual Parameter Count | Hidden Dimension h | Layer Count l | 12lh^2           |
|------------------------|--------------------|---------------|------------------|
| 6.7B                   | 4096               | 32            | 6,442,450,944    |
| 13.0B                  | 5120               | 40            | 12,582,912,000   |
| 32.5B                  | 6656               | 60            | 31,897,681,920   |
| 65.2B                  | 8192               | 80            | 64,424,509,440   |

We see that the approximation $12\,l\,h^2$ is quite close to actual parameter counts.

---

## 2.3 Memory Usage Analysis During Training

The main memory consumers during **training** are:

1. **Model Parameters**  
2. **Intermediate Activations** (from the forward pass)  
3. **Gradients**  
4. **Optimizer States** (e.g., AdamW’s first and second moments)

We first analyze parameters, gradients, and optimizer states. The topic of **intermediate activations** will be discussed later in detail.

Large models often use the AdamW optimizer with mixed precision (float16 for forward/backward passes and float32 for optimizer updates). Let the total number of trainable parameters be $\Phi$. During a single training iteration:

- There is one gradient element per parameter ($\Phi$ elements total).
- AdamW maintains two optimizer states (first-order and second-order moments), so that is $2\Phi$ elements in total.

A float16 element is 2 bytes, and a float32 element is 4 bytes. In mixed precision training:
- Model parameters (for the forward and backward pass) are stored in float16.
- Gradients are computed in float16.
- For parameter updates, the optimizer internally uses float32 copies of parameters and gradients, as well as float32 for its two moment states.

Hence, each trainable parameter uses (approximately) the following:

- **Forward/backward parameter**: float16 $\to$ 2 bytes  
- **Gradient**: float16 $\to$ 2 bytes  
- **Optimizer parameter copy**: float32 $\to$ 4 bytes  
- **Optimizer gradient copy**: float32 $\to$ 4 bytes  
- **First-order moment**: float32 $\to$ 4 bytes  
- **Second-order moment**: float32 $\to$ 4 bytes  

Summing:
$$
2 + 2 + 4 + 4 + 4 + 4 = 20\ \text{bytes per parameter}.
$$

Therefore, training a large model with $\Phi$ parameters under mixed precision with AdamW requires approximately:
$$
20\,\Phi \quad \text{bytes}
$$
to store parameters, gradients, and optimizer states.

#### Practical Note on Distributed Training

In practice, **distributed training** techniques like **ZeRO (Zero Redundancy Optimizer)** can partition optimizer states across multiple GPUs, reducing per-GPU memory usage. However, the *total* memory across the entire cluster remains on the same order as the above calculation (though effectively shared among GPUs).

---

## 2.4 Memory Usage Analysis During Inference

During **inference**, there are no gradients or optimizer states, nor do we need to store all intermediate activations for backpropagation. Thus, the main memory usage is from the model parameters themselves. If float16 is used for inference, this is roughly:
$$
2\,\Phi \quad \text{bytes}.
$$

When using a **key-value (KV) cache** for faster autoregressive inference, some additional memory is used (analyzed later). There is also small overhead for the input data and temporary buffers, but this is typically negligible compared to parameter storage and KV cache.

---

## 3. Computational Requirements (FLOPs) Estimation

**FLOPs** (floating point operations) measure computational cost. For two matrices $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{m \times l}$, computing $AB$ takes roughly $2nml$ FLOPs (one multiplication and one addition per element pair).

In one training iteration with input shape $[b, s]$, let’s break down the self-attention and MLP costs in a single transformer layer.

### 3.1 Self-Attention Block

A simplified representation of the self-attention operations is:

$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V
$$

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q K^\mathsf{T}}{\sqrt{h}}\right) \cdot V,
$$

$$
x_{\text{out}} = \text{Attention}(Q,K,V)\,W_O + x.
$$

Let $x\in \mathbb{R}^{b\times s\times h}$. The major FLOP contributors are:

1. **Computing $Q, K, V$**  
   Each matrix multiplication has shape $[b, s, h]\times[h, h]\to[b, s, h]$.  
   - Cost: $3 \times 2 \,b\,s\,h^2 = 6\,b\,s\,h^2$ (the factor 2 arises from multiply + add).

2. **$Q K^\mathsf{T}$**  
   - $Q, K \in \mathbb{R}^{b \times s \times h}$, often reinterpreted as $[b, a, s, \frac{h}{a}]$.  
   - The multiplication result has shape $[b, a, s, s]$.  
   - Cost: $2\,b\,s^2\,h$.

3. **Weighted $V$**  
   - We multiply the attention matrix $[b, a, s, s]$ by $V \in [b, a, s, \frac{h}{a}]$.  
   - Cost: $2\,b\,s^2\,h$.

4. **Output linear projection**  
   - $[b, s, h]\times[h, h]\to[b, s, h]$.  
   - Cost: $2\,b\,s\,h^2$.

Hence, the self-attention block requires about:
$$
6\,b\,s\,h^2 + 2\,b\,s\,h^2 + 2\,b\,s^2\,h + 2\,b\,s^2\,h
$$
which simplifies to
$$
8\,b\,s\,h^2 + 4\,b\,s^2\,h.
$$
(We will combine final terms more precisely in the overall layer cost.)

### 3.2 MLP Block

The MLP block typically is:
$$
x_{\text{MLP}} = \mathrm{GELU}\bigl(x_{\text{out}} W_1\bigr)\,W_2 + x_{\text{out}},
$$
where $W_1 \in [h, 4h]$ and $W_2 \in [4h, h]$. The major FLOP contributors are:

1. **First linear layer**:  
   - $[b, s, h]\times [h, 4h]\to[b, s, 4h]$.  
   - Cost: $2\,b\,s\,h\,(4h) = 8\,b\,s\,h^2$.

2. **Second linear layer**:  
   - $[b, s, 4h]\times [4h, h]\to[b, s, h]$.  
   - Cost: $2\,b\,s\,(4h)\,h = 8\,b\,s\,h^2$.

Nonlinear activations like GELU also incur some cost, but often it is modest compared to large matrix multiplications.

### 3.3 Summing Over One Transformer Layer

Combining self-attention and MLP:

- Self-Attention: ~$8\,b\,s\,h^2 + 4\,b\,s^2\,h$  
- MLP: ~$16\,b\,s\,h^2$ (sum of the two 8’s)

Thus, each transformer layer requires about:
$$
(8 + 16)\,b\,s\,h^2 + 4\,b\,s^2\,h \;=\; 24\,b\,s\,h^2 + 4\,b\,s^2\,h
$$
FLOPs.  

Additionally, computing logits in the final output layer has cost:
$$
2\,b\,s\,h\,V.
$$

For an $l$-layer transformer, one forward pass with input $[b, s]$ thus has a total cost:

$$
l \times \Bigl(24\,b\,s\,h^2 + 4\,b\,s^2\,h\Bigr) \;+\; 2\,b\,s\,h\,V.
$$

In many large-scale settings, $h\gg s$, so $4\,b\,s^2\,h$ can be smaller relative to $24\,b\,s\,h^2$, and $2\,b\,s\,h\,V$ can also be relatively smaller if $V$ is not extremely large. Hence a common approximation is:

$$
\approx 24\,l\,b\,s\,h^2.
$$

### 3.4 Relationship Between Computation and Parameter Count

Recall the parameter count is roughly $12\,l\,h^2$. Comparing:

$$
\frac{24\,b\,s\,h^2\,l}{12\,l\,h^2} = 2\,b\,s.
$$

Hence, for **each token**, **each parameter** performs about 2 FLOPs in one forward pass (one multiplication + one addition). In a training iteration (forward + backward), the cost is typically **3** times the forward pass. Thus per token-parameter we have
$$
2 \times 3 = 6
$$
FLOPs in total.

However, **activation recomputation** (discussed in Section 4.3) can add another forward-like pass during backpropagation, making the factor 4 instead of 3. Then per token-parameter we get $2 \times 4 = 8$ FLOPs.

---

## 3.5 Estimating Training Costs

Consider GPT-3 (175B parameters), which has about $1.75\times 10^{11}$ parameters trained on $3\times 10^{11}$ tokens. Each parameter-token pair does about 6 FLOPs in forward+backward:

$$
6 \times 1.746\times 10^{11} \times 3\times 10^{11}
\;=\; 3.1428\times 10^{23}\,\text{FLOPs}.
$$

<div style="text-align: center;">
  <img src="./CourseNLP2026/lecture_4_figure_1.png" width="100%">
  <p style="margin-top: 10px;">Large Language Model's Costs (https://arxiv.org/pdf/2005.14165v4)</p>
</div>

### 3.6 Training Time Estimation

Given the total FLOPs and the GPU hardware specs, we can estimate **training time**. The raw GPU FLOP rate alone does not reflect real-world utilization, and typical utilization might be between 0.3 and 0.55 due to factors like data loading, communication, and logging overheads.

Also note that **activation recomputation** adds an extra forward pass, giving a factor of 4 (forward + backward + recomputation) instead of 3. Thus, per token-parameter we get $2 \times 4 = 8$ FLOPs.

Hence, training time can be roughly estimated by:
$$
\text{Training Time} \approx \frac{8 \times (\text{tokens count}) \times (\text{model parameter count})}
{\text{GPU count} \times \text{GPU peak performance (FLOPs)} \times \text{GPU utilization}}.
$$

#### Example: GPT-3 (175B)

Using 1024 A100 (40GB) GPUs to train GPT-3 on 300B tokens:
- Peak performance per A100 (40GB) is about 312 TFLOPS.
- Assume GPU utilization at 0.45.
- Parameter count $\approx 175\text{B}$.
- Training tokens = 300B.

Estimated training time:

$$
\text{Time} \approx
\frac{8 \times 300\times 10^9 \times 175\times 10^9}
{1024 \times 312\times 10^{12} \times 0.45}
\;\approx\; 34\,\text{days}.
$$

This is consistent with reported real-world results in [7].

#### Example: LLaMA-65B

Using 2048 A100 (80GB) GPUs to train LLaMA-65B on 1.4T tokens:
- Peak performance per A100 (80GB) is about 624 TFLOPS.
- Assume GPU utilization at 0.3.
- Parameter count $\approx 65\text{B}$.
- Training tokens = 1.4T.

Estimated training time:

$$
\text{Time} \approx
\frac{8 \times 1.4\times 10^{12} \times 65\times 10^9}
{2048 \times 624\times 10^{12} \times 0.3}
\;\approx\; 21\,\text{days}.
$$

This also aligns with [4].

In-Class Question 1: What is the training time of using 4096 H100 GPUs to train LLaMA-70B on 300B tokens?

In-Class Question 2: What is the training time of using 1024 H100 GPUs to train LLaMA-70B on 1.4T tokens?

---

## 4. Intermediate Activation Analysis

During training, **intermediate activations** (values generated in the forward pass that are needed for the backward pass) can consume a large portion of memory. These include layer inputs, dropout masks, etc., but exclude model parameters and optimizer states. Although there are small buffers for means and variances in layer normalization, their total size is generally negligible compared to the main tensor dimensions.

Typically, float16 or bfloat16 is used to store activations. We assume 2 bytes per element for these. Dropout masks often use 1 byte per element (or sometimes bit-packing is used in advanced implementations).

Let us analyze the main contributors for each layer.

### 4.1 Self-Attention Block

Using:
$$
Q = x\,W_Q,\quad K = x\,W_K,\quad V = x\,W_V,
$$
$$
\text{Attention}(Q,K,V)= \text{softmax}\Bigl(\frac{QK^\mathsf{T}}{\sqrt{h}}\Bigr)\cdot V,
$$
$$
x_{\text{out}} = \text{Attention}(Q,K,V)\,W_O + x,
$$
we consider:

1. **Input $x$**  
   - Shape $[b, s, h]$, stored as float16 $\to 2\,b\,s\,h$ bytes.

2. **Q and K**  
   - Each is $[b, s, h]$ in float16, so $2\,b\,s\,h$ bytes each. Together: $4\,b\,s\,h$ bytes.

3. **$QK^\mathsf{T}$** (softmax input)  
   - Shape is $[b, a, s, s]$. Since $a \times \frac{h}{a}=h$, memory cost is $2\,b\,a\,s^2$ bytes.

4. **Dropout mask for the attention matrix**  
   - Typically uses 1 byte per element, shape $[b, a, s, s]\to b\,a\,s^2$ bytes.

5. **Softmax output (scores) and $V$**  
   - Score has $2\,b\,a\,s^2$ bytes, $V$ has $2\,b\,s\,h$ bytes.

6. **Output projection input**  
   - $[b, s, h]$ in float16 $\to 2\,b\,s\,h$ bytes.
   - Another dropout mask for the output: $[b, s, h]$ at 1 byte each $\to b\,s\,h$ bytes.

Summing these (grouping terms carefully), the self-attention block activations total around:
$$
11\,b\,s\,h + 5\,b\,s^2\,a \quad \text{(bytes, counting float16 and dropout masks)}.
$$

### 4.2 MLP Block

For the MLP:
$$
x = \mathrm{GELU}(x_{\text{out}}\,W_1)\,W_2 + x_{\text{out}},
$$
the main stored activations are:

1. **Input to first linear layer**: $[b,s,h]$ at float16 $\to 2\,b\,s\,h$ bytes.  
2. **Hidden activation** ($[b,s,4h]$) before or after GELU: $2\times 4\,b\,s\,h = 8\,b\,s\,h$ bytes. (One copy typically for the linear output and one for the activation function input/output; actual usage can vary by implementation.)  
3. **Output of second linear layer**: $[b,s,h]$ in float16 $\to 2\,b\,s\,h$ bytes.  
4. **Dropout mask**: $[b,s,h]$ at 1 byte per element $\to b\,s\,h$ bytes.

Hence, the MLP block’s stored activations sum to about:
$$
19\,b\,s\,h \quad \text{bytes}.
$$

### 4.3 Layer Normalization

Each layer has two layer norms (one for self-attention, one for MLP), each storing its input in float16. That is:
$$
2\times (2\,b\,s\,h) = 4\,b\,s\,h \quad \text{bytes}.
$$

Thus, **per layer**, the activation memory is roughly:
$$
(11\,b\,s\,h + 5\,b\,s^2\,a) + 19\,b\,s\,h + 4\,b\,s\,h
\;=\; 34\,b\,s\,h + 5\,b\,s^2\,a.
$$

An $l$-layer transformer has approximately:
$$
l \times \bigl(34\,b\,s\,h + 5\,b\,s^2\,a\bigr)
$$
bytes of intermediate activation memory.

### 4.4 Comparison with Parameter Memory

Unlike model parameter memory, which is essentially **constant** with respect to $b$ and $s$, **activation memory grows** with $b$ and $s$. Reducing batch size $b$ or sequence length $s$ is a common way to mitigate OOM (Out Of Memory) issues. For example:

- GPT-3 (175B parameters, $96$ layers, $h = 12288, a=96$) at sequence length $s=2048$:
  - Model parameters: $175\text{B} \times 2\text{ bytes}\approx 350\text{ GB}$ in float16.
  - Intermediate activations:  
    - If $b=1$, about $275$ GB (close to $0.79\times$ parameter memory).  
    - If $b=64$, about $17.6$ TB ($\approx 50\times$ parameter memory).  
    - If $b=128$, about $35.3$ TB ($\approx 100\times$ parameter memory).

Thus, activation memory can easily exceed parameter memory, especially at large batch sizes.

### 4.5 Activation Recomputation

To reduce peak activation memory, **activation recomputation** (or **checkpointing**) is often used. The idea is:
1. In the forward pass, we **do not** store all intermediate activations.  
2. In the backward pass, we **recompute** them from stored checkpoints (e.g., re-run part of the forward pass) before proceeding with gradient computations.

This trades extra computation for less memory usage and can cut activation memory from $O(l)$ to something smaller like $O(\sqrt{l})$, depending on the strategy. In practice, a common approach is to only store the activations at certain checkpoints (e.g., after each transformer block) and recompute the missing parts in the backward pass.

---

## 5. Conclusion

In this class, we explored how to estimate and analyze key aspects of training for large language models:

1. **Parameter Count**  
   - For a transformer-based LLM, each layer has approximately $12h^2 + 13h$ parameters, plus $Vh$ for the embeddings, leading to a total of  
     $$
     l(12h^2+13h)+Vh.
     $$
   - When $h$ is large, we often approximate it as $12\,l\,h^2$.

2. **Memory Usage**  
   - During **training**, parameters, gradients, and optimizer states typically use about $20\,\Phi$ bytes under mixed precision with AdamW (where $\Phi$ is the total parameter count).  
   - **Intermediate activations** can exceed parameter storage, especially with large batch size $b$ and long sequence length $s$. Techniques like **activation recomputation** help reduce this memory footprint.  
   - During **inference**, only parameters (2 bytes each in float16) and the **KV cache** are major memory consumers.

3. **FLOP Estimation**  
   - Roughly **2 FLOPs** per token-parameter during a forward pass (one multiplication + one addition).  
   - Training (forward + backward) yields about **6 FLOPs** per token-parameter if no recomputation is used, or **8 FLOPs** per token-parameter if activation recomputation is used.

By dissecting these components, we gain a clearer picture of **why** training large language models requires extensive memory and computation, and **how** various strategies (e.g., activation recomputation, KV cache) are applied to optimize hardware resources. Such understanding is crucial for practitioners to make informed decisions about scaling laws, distributed training setups, and memory-saving techniques.

---

## 6. References

1. Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 2020, 21(1): 5485-5551.  
2. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Advances in neural information processing systems*, 2017, 30.  
3. Brown T, Mann B, Ryder N, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 2020, 33: 1877-1901.  
4. Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.  
5. Sheng Y, Zheng L, Yuan B, et al. High-throughput generative inference of large language models with a single gpu. *arXiv preprint arXiv:2303.06865*, 2023.  
6. Korthikanti V, Casper J, Lym S, et al. Reducing activation recomputation in large transformer models. *arXiv preprint arXiv:2205.05198*, 2022.  
7. Narayanan D, Shoeybi M, Casper J, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2021: 1-15.  
8. Smith S, Patwary M, Norick B, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. *arXiv preprint arXiv:2201.11990*, 2022.

---