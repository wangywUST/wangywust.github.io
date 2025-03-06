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

# Lecture 4: Analysis of Transformer Models: Parameter Count, Computation, Activations, and KV Cache

In-Class Question 1: Given layer number $N$ as 6, model dimension $d_{model}$ as 512, feed-forward dimension $d_{ff}$ = 2048, number of attention heads $h$ = 8, what is the total number of learnable parameters in a vanilla Transformer model?

In-Class Question 2: Given layer number $N$ as 6, model dimension $d_{model}$ as 1024, feed-forward dimension $d_{ff}$ = 4096, number of attention heads $h$ = 16, what is the total number of learnable parameters in a vanilla Transformer model?

Reference Tutorial: [Parameter size of vanilla transformer](https://colab.research.google.com/drive/1jdhD6yexq2PBZYg9zg5FG-YRGyL8ZJ0L?usp=sharing)
Reference Tutorial: [Analysis of Transformer Models](https://zhuanlan.zhihu.com/p/624740065)

## 1. Introduction

Welcome to this tutorial on analyzing the memory and computational efficiency of training large language models (LLMs). With the rise of models like OpenAI's ChatGPT, researchers and engineers have become increasingly interested in the mechanics behind Large Language Models. The "large" aspect of these models refers both to the number of model parameters and the scale of training data. For example, GPT-3 has 175 billion parameters and was trained on 570 GB of data. Consequently, training such models presents two key challenges: **memory efficiency** and **computational efficiency**.

Most large models in industry today utilize the transformer architecture. Their structures can be broadly divided into encoder-decoder (exemplified by T5) and decoder-only. The decoder-only structure can be split into *Causal LM* (represented by the GPT series) and *Prefix LM* (represented by GLM). Causal language models like GPT have achieved significant success, so many mainstream LLMs employ the Causal LM paradigm. In this tutorial, we will focus on the decoder-only transformer framework, analyzing its parameter count, computational requirements, intermediate activations, and KV cache usage to better understand the memory and computational efficiency of training and inference.

To make the analysis clearer, let us define the following notation:

- $l$: Number of transformer layers  
- $h$: Hidden dimension  
- $a$: Number of attention heads  
- $V$: Vocabulary size  
- $b$: Training batch size  
- $s$: Sequence length  

---

## 2. Model Parameter Count

A transformer model commonly consists of $l$ identical layers, each containing a **self-attention block** and an **MLP block**. Let’s break down their parameters:

1. **Self-Attention Block**  
   The trainable parameters here include:
   - $W_Q, W_K, W_V, W_O \in [h, h]$
   - The corresponding biases, each in $[h]$

   Hence, the total parameter count in self-attention is $4h^2 + 4h$.

2. **MLP Block**  
   Usually, the MLP block has two linear layers:
   - First layer: $W_1 \in [h, 4h]$ and bias in $[4h]$
   - Second layer: $W_2 \in [4h, h]$ and bias in $[h]$

   Therefore, the MLP block has $8h^2 + 5h$ parameters.

3. **Layer Normalization**  
   Both the self-attention and MLP blocks have a layer normalization containing scaling parameter $\gamma$ and shifting parameter $\beta$ in $[h]$. So two layer norms contribute $4h$ parameters.

Putting this together, each transformer layer has:
$$
12h^2 + 13h
$$
trainable parameters.

In addition, there is a word embedding matrix, typically in $[V, h]$, which contributes $Vh$ parameters. Often, the final output layer is weight-shared with this embedding matrix.

If the position encoding is trainable, it might add a few more parameters, but often relative position encodings (e.g., RoPE, ALiBi) contain no trainable parameters. We will ignore any small parameter additions from these positional encodings.

Thus, an $l$-layer transformer model has a total trainable parameter count of:
$$
l(12h^2 + 13h) + Vh.
$$
When $h$ is large, the linear terms can be ignored, and the model’s parameter count is approximately:
$$
12\,l\,h^2.
$$

### Estimating LLaMA Parameter Counts

Below is a table comparing the approximate $12lh^2$ calculation for various LLaMA models to their actual parameter counts:

| Actual Parameter Count | Hidden Dimension h | Layer Count l | 12lh^2        |
|------------------------|--------------------|---------------|--------------------|
| 6.7B                   | 4096               | 32            | 6,442,450,944      |
| 13.0B                  | 5120               | 40            | 12,582,912,000     |
| 32.5B                  | 6656               | 60            | 31,897,681,920     |
| 65.2B                  | 8192               | 80            | 64,424,509,440     |

### 2.1 Memory Usage Analysis During Training

The main memory consumers during training are:

1. **Model Parameters**  
2. **Intermediate Activations** (from forward pass)  
3. **Gradients**  
4. **Optimizer States**

We will first analyze parameters, gradients, and optimizer states. The topic of intermediate activations will be discussed in detail later.

Large models often use the AdamW optimizer with mixed precision (float16 for forward/backward passes and float32 for optimizer updates). Let the total number of trainable parameters be $\Phi$. During a single training iteration:

- There is one gradient element per parameter ($\Phi$ elements total).
- AdamW maintains two optimizer states (first-order and second-order moments), so $2\Phi$ elements in total.

A float16 element is 2 bytes, and a float32 element is 4 bytes. In mixed precision training:

- Model parameters are in float16 for forward/backward.
- Gradients are computed in float16.
- For parameter updates, the optimizer uses float32 parameters, float32 gradients, and float32 optimizer states.

Hence, each trainable parameter uses:
$$
(2 + 4) + (2 + 4) + (4 + 4) = 20 \text{ bytes}.
$$
Therefore, training a large model with $\Phi$ parameters under mixed precision with AdamW requires approximately:
$$
20\,\Phi \text{ bytes}
$$
to store parameters, gradients, and optimizer states.

### 2.2 Memory Usage Analysis During Inference

During **inference**, there are no gradients or optimizer states, nor is there any need to store full intermediate activations for backpropagation. Thus, the main memory usage is from the model parameters themselves. If float16 is used for inference, this is roughly:
$$
2\,\Phi \text{ bytes}.
$$
When using a key-value (KV) cache for faster autoregressive inference, some additional memory is used (analyzed later). There is also small memory overhead for input data and temporary buffers, but this is typically negligible compared to parameter storage and KV cache.

---

## 3. Computational Requirements (FLOPs) Estimation

**FLOPs** (floating point operations) measure computational cost. For two matrices $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{m \times l}$, computing $AB$ takes roughly $2nml$ FLOPs (one multiplication and one addition per pair).

In one training iteration with input shape $[b, s]$, let’s break down the self-attention and MLP costs in a single transformer layer.

### 3.1 Self-Attention Block

Self-attention operations can be summarized as:

$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V
$$

$$
x_{\text{out}} = \text{softmax}\bigl(\frac{QK^T}{\sqrt{h}}\bigr) \cdot V \cdot W_O + x
$$

1. Computing $Q, K, V$:  
   Each matrix multiplication has shape $[b, s, h]\times[h, h]\to [b, s, h]$.  
   Cost: $3\times 2bsh^2 = 6bsh^2$.
2. $QK^T$:  
   Inputs $[b, a, s, \frac{h}{a}]\times [b, a, \frac{h}{a}, s]\to [b, a, s, s]$.  
   Cost: $2bs^2h$.
3. Weighted $V$:  
   Inputs $[b, a, s, s]\times [b, a, s, \frac{h}{a}]\to [b, a, s, \frac{h}{a}]$.  
   Cost: $2bs^2h$.
4. Output linear projection:  
   $[b, s, h]\times [h, h]\to [b, s, h]$.  
   Cost: $2bsh^2$.

### 3.2 MLP Block

The MLP block:

$$
x = f_{\text{gelu}}(x_{\text{out}}W_1)W_2 + x_{\text{out}}
$$

1. First linear layer:  
   $[b, s, h]\times [h, 4h]\to [b, s, 4h]$.  
   Cost: $8bsh^2$.
2. Second linear layer:  
   $[b, s, 4h]\times [4h, h]\to [b, s, h]$.  
   Cost: $8bsh^2$.

Hence, each transformer layer requires about:
$$
24bsh^2 + 4bs^2h
$$
FLOPs. Additionally, computing logits in the final output layer costs:
$$
2bshV.
$$

For an $l$-layer transformer, one training iteration with input $[b, s]$ thus has a total cost:

$$
l \times (24bsh^2 + 4bs^2h) \;+\; 2bshV.
$$

### Relationship Between Computation and Parameter Count

When $h$ is large and $h \gg s$, the above can be approximated by $24\,b\,s\,h^2\,l$. Earlier, we noted the parameter count is roughly $12\,l\,h^2$. Comparing these:

$$
\frac{24\,b\,s\,h^2\,l}{12\,l\,h^2} = 2\,b\,s.
$$

Hence, for **each token**, **each parameter** performs about 2 FLOPs in one forward pass (one multiplication and one addition). In one training iteration, the backward pass doubles that cost, so forward + backward is a factor of 3. Therefore, per token per parameter, we have
$$
2 \times 3 = 6
$$
FLOPs in total.

### 3.3 Estimating Training Costs

Consider GPT3-175B, which has $174{,}600M$ parameters trained on $300B$ tokens. Each parameter-token pair does 6 FLOPs in a forward+backward cycle:

$$
6 \times 174{,}600 \times 10^6 \times 300 \times 10^9 
\;=\; 3.1428 \times 10^{23}\,\text{FLOPs}.
$$

<div style="text-align: center;">
  <img src="./Course/lecture_4_figure_1.png" width="100%">
  <p style="margin-top: 10px;">Large Language Model's Costs</p>
</div>

### 3.4 Training Time Estimation

Given the total FLOPs and the GPU hardware, we can estimate training time. The raw GPU FLOP rate alone does not reflect real-world utilization (due to overheads like data loading, communication, logging), so typical GPU utilization might be 0.3 to 0.55.

Also note that **activation recomputation** (discussed later) adds an extra forward pass to save memory, resulting in a factor of 4 (forward + backward + recomputation) instead of 3. Thus, per token per parameter, we get $2\times 4 = 8$ FLOPs.

Given the training tokens, hardware environment configuration, the computation time for training a transformer model is:

$$
\text{Training Time} \approx \frac{8 \times \text{tokens count} \times \text{model parameter count}}{\text{GPU count} \times \text{GPU peak performance in flops} \times \text{GPU utilization}}
$$

#### Example: GPT3-175B

Using 1024 A100 (40GB) GPUs to train GPT3-175B on 300B tokens:

- Peak performance per A100 (40GB) is around 312 TFLOPS.
- Assume GPU utilization at 0.45.
- Parameter count $\approx 175$B.
- Training tokens = 300B.

Estimated training time:

$$
\text{Time} \approx \frac{8 \times 300\times 10^9 \times 175\times 10^9}{1024 \times 312 \times 10^{12} \times 0.45} \approx 34\,\text{days}.
$$

This matches reported real-world results in [7].

#### Example: LLaMA-65B

Using 2048 A100 (80GB) GPUs to train LLaMA-65B on 1.4T tokens:

- Peak performance per A100 (80GB) is about 624 TFLOPS.
- Assume GPU utilization at 0.3.
- Parameter count $\approx 65$B.
- Training tokens = 1.4T.

Estimated training time:

$$
\text{Time} \approx \frac{8 \times 1.4 \times 10^{12} \times 65\times 10^9}{2048 \times 624\times 10^{12} \times 0.3} \approx 21\,\text{days}.
$$

This also aligns with [4].

---

## 4. Intermediate Activation Analysis

During training, **intermediate activations** (values generated in the forward pass and required during the backward pass) can also consume a large portion of memory. These include layer inputs, dropout masks, etc., but exclude model parameters and optimizer states. Although there are small buffers for means and variances in layer normalization, they are negligible compared to the main tensor dimensions.

Typically, float16 or bfloat16 is used to store activations. We will assume 2 bytes per element for these. Dropout masks often use 1 byte per element.

Let’s analyze this for each layer (self-attention + MLP + associated layer norms).

### 4.1 Self-Attention Block

Given:
$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V,\quad
x_{\text{out}} = \text{softmax}\bigl(\frac{QK^T}{\sqrt{h}}\bigr)\cdot V \cdot W_O + x,
$$
we consider:

1. **Input $x$**: $[b, s, h]$, stored as float16 $\to 2bsh$ bytes.  
2. **Q and K**: each $[b, s, h]$, so $4bsh$ bytes in total (since both are stored in float16).  
3. **$QK^T$** (softmax input): shape $[b, a, s, s]$, so $2bs^2a$ bytes.  
4. **Dropout mask for softmax**: $bs^2a$ bytes (1 byte per element).  
5. **Score and V**: Score has $2bs^2a$ bytes, $V$ has $2bsh$ bytes.  
6. **Output projection and dropout**: input to linear is $2bsh$, dropout mask is $bsh$ bytes.

Summing up, self-attention activations total about $11bsh + 5bs^2a$ bytes per layer.

### 4.2 MLP Block

For the MLP:
$$
x = f_{\text{gelu}}(x_{\text{out}}W_1)\,W_2 + x_{\text{out}},
$$
we have:

1. Input to first linear layer: $2bsh$ bytes.  
2. GELU activation input: $8bsh$ bytes.  
3. Second linear layer input: $8bsh$ bytes.  
4. Dropout mask: $bsh$ bytes.

Total is $19bsh$ bytes in the MLP block.

### 4.3 Layer Normalization

Each layer has two layer norms (one for self-attention, one for MLP), each storing its input in $2bsh$. Together, that is $4bsh$ bytes.

Hence, **per layer**, the activation memory is:
$$
(11bsh + 5bs^2a) + 19bsh + 4bsh = 34bsh + 5bs^2a.
$$
An $l$-layer transformer then has approximately:
$$
(34\,b\,s\,h + 5\,b\,s^2\,a)\times l
$$
bytes of intermediate activation memory (ignoring small contributions from the embedding or final projection).

### Comparison with Parameter Memory

Unlike model parameter memory, which remains constant regardless of batch size $b$ or sequence length $s$, **activation memory grows with $b$ and $s$**. Reducing batch size is a common way to mitigate Out Of Memory (OOM) issues, effectively decreasing the needed activation memory rather than parameter memory.

For GPT3-175B (96 layers, $h=12288, a=96$) using float16:

- Model parameters: 175B, stored at 2 bytes each $\to 350$ GB.
- Intermediate activations ($s=2048$):

  - If $b=1$, about $275$ GB (close to 0.79$\times$ parameter memory).
  - If $b=64$, about 17.6 TB (50$\times$ parameter memory).
  - If $b=128$, about 35.3 TB (100$\times$ parameter memory).

Evidently, activation memory can greatly exceed parameter memory at large batch sizes. To reduce activation memory, **activation recomputation** is frequently employed. It replays certain forward operations during backward pass (a time-space trade-off), reducing memory from $O(n)$ to $O(\sqrt{n})$ at the cost of additional forward compute.

---

## 5. KV Cache

For faster generative inference, transformers often use a **KV cache**, which stores keys and values from past tokens so that each new token only attends to this pre-stored context rather than recomputing key and value from scratch.

A typical inference process has two phases:

1. **Prefill Phase**: The prompt sequence is fed into the model, generating the key and value cache for each layer.  
2. **Decoding Phase**: Tokens are generated one by one, each time updating and using the cached keys and values.

Let the layer $i$ have weights $W_Q^i, W_K^i, W_V^i, W_O^i \in \mathbb{R}^{h\times h}$ and MLP weights $W_1^i, W_2^i$. Denote:

- $x_K^i, x_V^i, x_Q^i, x_{\text{out}}^i \in \mathbb{R}^{b\times s\times h}$ during prefill phase.
- $t^i \in \mathbb{R}^{b \times 1 \times h}$ represents the newly generated token embedding at layer $i$ during decoding.

### 5.1 Memory Usage of the KV Cache

Suppose the input sequence length is $s$, the output length is $n$, and we store KV cache in float16. The shape of $K$ or $V$ for each layer is $[b, (s+n), h]$. Because each entry is 2 bytes, and there are two caches ($K$ and $V$), the total memory for layer $i$ is:
$$
2 \times b(s+n)h \times 2 = 4\,b(s+n)h.
$$
For $l$ layers, the total KV cache memory is:
$$
4\,l\,b\,h\,(s+n).
$$

#### GPT3 Example

Consider GPT3 again with 350 GB of parameter memory. Suppose $b=64$, $s=512$, and $n=32$. Then:

- Model parameters: 350 GB
- KV cache: $4\,l\,b\,h\,(s+n)\approx 164$ GB

Here, the KV cache is roughly half the size of the model parameters for these settings.

---

## 6. Conclusion

In this tutorial, we explored how to estimate and analyze:

1. **Parameter Count**: For a transformer-based LLM, each layer has $12h^2+13h$ parameters, plus $Vh$ for embeddings, leading to a total of $l(12h^2+13h)+Vh$.
2. **Memory Usage**:  
   - During training, parameters, gradients, and optimizer states typically use $20\Phi$ bytes under mixed precision with AdamW (where $\Phi$ is the total parameter count).  
   - Intermediate activations can exceed parameter storage, especially with large $b$ and $s$. Techniques like activation recomputation help mitigate this.  
   - During inference, only parameters (2 bytes each in float16) and the KV cache are major memory consumers.
3. **FLOP Estimation**: Approximately $2$ FLOPs per token-parameter in forward pass. Factoring in backward pass (and possibly activation recomputation) influences this multiple.
4. **KV Cache**: A popular mechanism for fast generative inference, storing keys and values for each token in float16 to reduce repeated computation.

By dissecting these components, we gain a clearer picture of **why** training large language models demands extensive memory and computation, and **how** various strategies (e.g., activation recomputation, KV cache) are applied to optimize usage of hardware resources.

---

## 7. References

1. Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 2020, 21(1): 5485-5551.  
2. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Advances in neural information processing systems*, 2017, 30.  
3. Brown T, Mann B, Ryder N, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 2020, 33: 1877-1901.  
4. Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.  
5. Sheng Y, Zheng L, Yuan B, et al. High-throughput generative inference of large language models with a single gpu. *arXiv preprint arXiv:2303.06865*, 2023.  
6. Korthikanti V, Casper J, Lym S, et al. Reducing activation recomputation in large transformer models. *arXiv preprint arXiv:2205.05198*, 2022.  
7. Narayanan D, Shoeybi M, Casper J, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2021: 1-15.  
8. Smith S, Patwary M, Norick B, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. *arXiv preprint arXiv:2201.11990*, 2022.

---