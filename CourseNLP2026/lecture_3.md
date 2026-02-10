# Lecture 3: Transformer Architecture

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [General Transformer Architecture](#general-transformer-architecture)
  - [Encoder-Decoder Transformers](#encoder-decoder-transformers)
  - [Encoder-Only Transformers](#encoder-only-transformers)
  - [Decoder-Only Transformers](#decoder-only-transformers)
- [Attention Mechanism](#attention-mechanism)
  - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
    - [Example 1: Detailed Numerical Computation](#example-1-detailed-numerical-computation)
    - [Example 2: Another Small-Dimension Example](#example-2-another-small-dimension-example)
  - [Multi-Head Attention](#multi-head-attention)
    - [Example 1: Two-Head Attention Computation (Conceptual Illustration)](#example-1-two-head-attention-computation-conceptual-illustration)
    - [Example 2: Two-Head Attention with Full Numerical Details](#example-2-two-head-attention-with-full-numerical-details)
    - [Example 3: Three-Head Attention with Another Set of Numbers (Short Demonstration)](#example-3-three-head-attention-with-another-set-of-numbers-short-demonstration)
- [Position-Wise Feed-Forward Networks](#position-wise-feed-forward-networks)
  - [Example: Numerical Computation of the Feed-Forward Network](#example-numerical-computation-of-the-feed-forward-network)
- [Training and Optimization](#training-and-optimization)
  - [Optimizer and Learning Rate Scheduling](#optimizer-and-learning-rate-scheduling)
    - [Example: Learning Rate Calculation](#example-learning-rate-calculation)
- [Conclusion](#conclusion)

---

## Introduction
The Transformer model is a powerful deep learning architecture that has achieved groundbreaking results in various fields—most notably in Natural Language Processing (NLP), computer vision, and speech recognition—since it was introduced in *Attention Is All You Need* (Vaswani et al., 2017). Its core component is the self-attention mechanism, which efficiently handles long-range dependencies in sequences while allowing for extensive parallelization. Many subsequent models, such as BERT, GPT, Vision Transformer (ViT), and multimodal Transformers, are built upon this foundational structure.

## Background
Before the Transformer, sequential modeling primarily relied on Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs). These networks often struggled with capturing long-distance dependencies, parallelization, and computational efficiency. In contrast, the self-attention mechanism of Transformers captures global dependencies across input and output sequences simultaneously and offers excellent parallelization capabilities.

## General Transformer Architecture
Modern Transformer architectures typically fall into one of three categories: encoder-decoder, encoder-only, or decoder-only, depending on the application scenario.

### Encoder-Decoder Transformers
An encoder-decoder Transformer first encodes the input sequence into a contextual representation, then the decoder uses this encoded information to generate the target sequence. Typical applications include machine translation and text summarization. Models like T5 and MarianMT are representative of this structure.

### Encoder-Only Transformers
Encoder-only models focus on learning bidirectional contextual representations of input sequences for classification, retrieval, and language understanding tasks. BERT and its variants (RoBERTa, ALBERT, etc.) belong to this category.

### Decoder-Only Transformers
Decoder-only models generate outputs in an autoregressive manner, making them well-suited for text generation, dialogue systems, code generation, and more. GPT series, LLaMA, and PaLM are examples of this type.

---

## Attention Mechanism
The core of the Transformer lies in its attention mechanism, which allows the model to focus on the most relevant parts of the input sequence given a query. Below, we detail the Scaled Dot-Product Attention and the Multi-Head Attention mechanisms.

### What is Attention?

The attention mechanism describes a recent new group of layers in neural networks that has attracted a lot of interest in the past few years, especially in sequence tasks. There are a lot of different possible definitions of "attention" in the literature, but the one we will use here is the following: _the attention mechanism describes a weighted average of (sequence) elements with the weights dynamically computed based on an input query and elements' keys_. So what does this exactly mean? The goal is to take an average over the features of multiple elements. However, instead of weighting each element equally, we want to weight them depending on their actual values. In other words, we want to dynamically decide on which inputs we want to "attend" more than others. In particular, an attention mechanism has usually four parts we need to specify:

* **Query**: The query is a feature vector that describes what we are looking for in the sequence, i.e. what would we maybe want to pay attention to.
* **Keys**: For each input element, we have a key which is again a feature vector. This feature vector roughly describes what the element is "offering", or when it might be important. The keys should be designed such that we can identify the elements we want to pay attention to based on the query.
* **Values**: For each input element, we also have a value vector. This feature vector is the one we want to average over.
* **Score function**: To rate which elements we want to pay attention to, we need to specify a score function $f_{attn}$. The score function takes the query and a key as input, and output the score/attention weight of the query-key pair. It is usually implemented by simple similarity metrics like a dot product, or a small MLP.


The weights of the average are calculated by a softmax over all score function outputs. Hence, we assign those value vectors a higher weight whose corresponding key is most similar to the query. If we try to describe it with pseudo-math, we can write: 

$$
\alpha_i = \frac{\exp\left(f_{attn}\left(\text{key}_i, \text{query}\right)\right)}{\sum_j \exp\left(f_{attn}\left(\text{key}_j, \text{query}\right)\right)}, \hspace{5mm} \text{out} = \sum_i \alpha_i \cdot \text{value}_i
$$

Visually, we can show the attention over a sequence of words as follows:

<div style="text-align: center;">
  <img src="./CourseNLP2026/attention_example.svg" width="50%">
  <p style="margin-top: 10px;">Attention Example</p>
</div>

For every word, we have one key and one value vector. The query is compared to all keys with a score function (in this case the dot product) to determine the weights. The softmax is not visualized for simplicity. Finally, the value vectors of all words are averaged using the attention weights.

Most attention mechanisms differ in terms of what queries they use, how the key and value vectors are defined, and what score function is used. The attention applied inside the Transformer architecture is called **self-attention**. In self-attention, each sequence element provides a key, value, and query. For each element, we perform an attention layer where based on its query, we check the similarity of the all sequence elements' keys, and returned a different, averaged value vector for each element. We will now go into a bit more detail by first looking at the specific implementation of the attention mechanism which is in the Transformer case the scaled dot product attention.

### Scaled Dot-Product Attention
Given a query matrix $Q$, key matrix $K$, and value matrix $V$, the attention formula is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d_k}} \Bigr)V
$$

where $d_k$ is the dimensionality of the key vectors (often the same as the query dimensionality). 
Every row of $Q$ corresponds a token's embedding.

#### Example 1: Detailed Numerical Computation
Suppose we have the following matrices (small dimensions chosen for illustrative purposes):

$$
Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}, \quad
K = \begin{bmatrix}
1 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
V = \begin{bmatrix}
0 & 2 \\
1 & 1 \\
2 & 0
\end{bmatrix}
$$

1. **Compute $QK^T$**  
   According to the example setup:

   $$
   QK^T = \begin{bmatrix}
   1 & 0 & 1 \\
   1 & 1 & 0 \\
   2 & 1 & 1
   \end{bmatrix}
   $$

2. **Scale by $\sqrt{d_k}$**  
   Here, $d_k = 2$. Thus, $\sqrt{2} \approx 1.41$. So,

   $$
   \frac{QK^T}{\sqrt{2}} \approx
   \begin{bmatrix}
   0.71 & 0    & 0.71 \\
   0.71 & 0.71 & 0    \\
   1.41 & 0.71 & 0.71
   \end{bmatrix}
   $$

3. **Apply softmax row-wise**  
   The softmax of a vector $x$ is given by
   $$
   \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}.
   $$
   Let's calculate this row by row:

   - Row 1: $[0.71, 0, 0.71]$  
     * Calculate exponentials:
       * $e^{0.71} \approx 2.034$ (for the 1st and 3rd elements)
       * $e^{0} = 1$ (for the 2nd element)
     * Sum of exponentials: $2.034 + 1 + 2.034 \approx 5.068$
     * Softmax values:
       * $\frac{2.034}{5.068} \approx 0.401$
       * $\frac{1}{5.068} \approx 0.197$
       * $\frac{2.034}{5.068} \approx 0.401$
     * Final result: $[0.401, 0.197, 0.401]$ ≈ $[0.40, 0.20, 0.40]$

   - Row 2: $[0.71, 0.71, 0]$  
     * Calculate exponentials:
       * $e^{0.71} \approx 2.034$ (for the 1st and 2nd elements)
       * $e^{0} = 1$ (for the 3rd element)
     * Sum of exponentials: $2.034 + 2.034 + 1 \approx 5.068$
     * Softmax values:
       * $\frac{2.034}{5.068} \approx 0.401$
       * $\frac{2.034}{5.068} \approx 0.401$
       * $\frac{1}{5.068} \approx 0.197$
     * Final result: $[0.401, 0.401, 0.197]$ ≈ $[0.40, 0.40, 0.20]$

   - Row 3: $[1.41, 0.71, 0.71]$  
     * Calculate exponentials:
       * $e^{1.41} \approx 4.096$
       * $e^{0.71} \approx 2.034$ (for the 2nd and 3rd elements)
     * Sum of exponentials: $4.096 + 2.034 + 2.034 \approx 8.164$
     * Softmax values:
       * $\frac{4.096}{8.164} \approx 0.501$
       * $\frac{2.034}{8.164} \approx 0.249$
       * $\frac{2.034}{8.164} \approx 0.249$
     * Final result: $[0.501, 0.249, 0.249]$ ≈ $[0.50, 0.25, 0.25]$

   The final softmax matrix $\alpha$ is:
   $$
   \alpha = \begin{bmatrix}
   0.40 & 0.20 & 0.40 \\
   0.40 & 0.40 & 0.20 \\
   0.50 & 0.25 & 0.25
   \end{bmatrix}
   $$

   Key observations about the softmax results:
   1. All output values are between 0 and 1
   2. Each row sums to 1
   3. Equal input values (Row 1) result in equal output probabilities
   4. Larger input values receive larger output probabilities (middle values in Rows 2 and 3)

   (slight rounding applied).

4. **Multiply by \(V\)**  

   $$
   \text{Attention}(Q, K, V) = \alpha V.
   $$
   - Row 1 weights \([0.40, 0.20, 0.40]\) on \(V\):

     $$
     0.40 \times [0,2] + 0.20 \times [1,1] + 0.40 \times [2,0]
     = [0 + 0.20 + 0.80,\; 0.80 + 0.20 + 0]
     = [1.00,\; 1.00].
     $$

   - Row 2 weights \([0.40, 0.40, 0.20]\):

     $$
     0.40 \times [0,2] + 0.40 \times [1,1] + 0.20 \times [2,0]
     = [0,\;0.80] + [0.40,\;0.40] + [0.40,\;0]
     = [0.80,\;1.20].
     $$

   - Row 3 weights \([0.50, 0.25, 0.25]\):

     $$
     0.50 \times [0,2] + 0.25 \times [1,1] + 0.25 \times [2,0]
     = [0,\;1.0] + [0.25,\;0.25] + [0.50,\;0]
     = [0.75,\;1.25].
     $$

   **Final Output**:

   $$
   \begin{bmatrix}
   1.00 & 1.00 \\
   0.80 & 1.20 \\
   0.75 & 1.25
   \end{bmatrix}
   $$

   (rounded values).

---

#### Example 2: Another Small-Dimension Example
Let us consider an even smaller example:

$$
Q = \begin{bmatrix}
1 & 1
\end{bmatrix}, \quad
K = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
V = \begin{bmatrix}
2 & 3 \\
4 & 1
\end{bmatrix}.
$$

Here, $Q$ is $1 \times 2$, $K$ is $2 \times 2$, and $V$ is $2 \times 2$.

1. **Compute $QK^T$**  
   Since $K$ is a square matrix, $K^T = K$:

   $$
   QK^T = QK =
   \begin{bmatrix}
   1 & 1
   \end{bmatrix}
   \begin{bmatrix}
   1 & 0 \\
   0 & 1
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 & 1
   \end{bmatrix}.
   $$

2. **Scale by $\sqrt{d_k}$**  
   $d_k = 2$. Thus, $\frac{1}{\sqrt{2}} \approx \frac{1}{1.41} \approx 0.71$. So

   $$
   \frac{[1,\;1]}{1.41} \approx [0.71,\;0.71].
   $$

3. **Softmax**  
   $[0.71, 0.71]$ has equal values, so the softmax is $[0.5, 0.5]$.

4. **Multiply by $V$**  

   $$
   [0.5,\;0.5]
   \begin{bmatrix}
   2 & 3 \\
   4 & 1
   \end{bmatrix}
   =
   0.5 \times [2,3] + 0.5 \times [4,1]
   =
   [1,1.5] + [2,0.5]
   =
   [3,2].
   $$

**Final Output**: $[3,\;2]$.

#### Example 3: Larger Q and K with V as a Column Vector
Let us consider an example where $Q$ and $K$ have a larger dimension, but $V$ has only one column:

$$
Q = \begin{bmatrix}
1 & 1 & 1 & 1
\end{bmatrix}, \quad
K = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
V = \begin{bmatrix}
2 \\
4 \\
6 \\
8
\end{bmatrix}.
$$

In-Course Question: Attention computation result of the above Q, K, V.

<!-- 1. **Compute $QK^T$**  
   Since $K$ is a square matrix and $K^T = K$:

   $$
   QK^T = QK =
   \begin{bmatrix}
   1 & 1 & 1 & 1
   \end{bmatrix}
   \begin{bmatrix}
   1 & 0 & 0 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 1
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 & 1 & 1 & 1
   \end{bmatrix}.
   $$

2. **Scale by $\sqrt{d_k}$**  
   Here, $d_k = 4$. Thus, $\frac{1}{\sqrt{4}} = \frac{1}{2} = 0.5$. So,

   $$
   \frac{[1,\;1,\;1,\;1]}{2} = [0.5,\;0.5,\;0.5,\;0.5].
   $$

3. **Softmax**  
   Since all values are equal, the softmax yields equal weights:

   $$
   \text{softmax}([0.5,\;0.5,\;0.5,\;0.5]) = [0.25,\;0.25,\;0.25,\;0.25].
   $$

4. **Multiply by $V$**  

   $$
   [0.25,\;0.25,\;0.25,\;0.25]
   \begin{bmatrix}
   2 \\
   4 \\
   6 \\
   8
   \end{bmatrix}
   = 0.25 \times 2 + 0.25 \times 4 + 0.25 \times 6 + 0.25 \times 8 = 0.5 + 1 + 1.5 + 2 = 5.
   $$

**Final Output**: $5$. -->


---

### Multi-Head Attention
Multi-head attention projects $Q, K, V$ into multiple subspaces and performs several parallel scaled dot-product attentions (referred to as "heads"). These are concatenated, then transformed via a final linear projection:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O,
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V).
$$

Below are multiple examples illustrating how multi-head attention calculations are performed, with increasingly detailed numeric demonstrations.

#### Example 1: Two-Head Attention Computation (Conceptual Illustration)
Let us assume we have a 2-head setup ($h = 2$), each head operating on half the dimension of $Q, K, V$. For instance, if the original dimension is 4, each head dimension could be 2.

- **Step 1**: Linear transformations and splitting  

  $$
  Q W^Q \rightarrow [Q_1,\ Q_2], \quad
  K W^K \rightarrow [K_1,\ K_2], \quad
  V W^V \rightarrow [V_1,\ V_2].
  $$

  Here, $[Q_1,\ Q_2]$ means we split the transformed $Q$ along its last dimension into two sub-matrices (head 1 and head 2).

- **Step 2**: Compute scaled dot-product attention for each head  

  $$
  \text{head}_1 = \text{Attention}(Q_1, K_1, V_1), \quad
  \text{head}_2 = \text{Attention}(Q_2, K_2, V_2).
  $$

  Suppose after computation:

  $$
  \text{head}_1 = \begin{bmatrix}
  h_{11} & h_{12} \\
  h_{21} & h_{22} \\
  h_{31} & h_{32}
  \end{bmatrix}, \quad
  \text{head}_2 = \begin{bmatrix}
  g_{11} & g_{12} \\
  g_{21} & g_{22} \\
  g_{31} & g_{32}
  \end{bmatrix}.
  $$

- **Step 3**: Concatenate and apply final linear transform  
  Concatenating the heads yields a $3 \times 4$ matrix (if each head is $3 \times 2$):

  $$
  \text{Concat}(\text{head}_1, \text{head}_2) =
  \begin{bmatrix}
  h_{11} & h_{12} & g_{11} & g_{12} \\
  h_{21} & h_{22} & g_{21} & g_{22} \\
  h_{31} & h_{32} & g_{31} & g_{32}
  \end{bmatrix}.
  $$

  We then multiply by $W^O$ (e.g., a $4 \times 4$ matrix) to get the final multi-head attention output.

> *Note*: Actual numeric computation requires specifying all projection matrices $W_i^Q, W_i^K, W_i^V, W^O$ and the input $Q, K, V$. Below, we provide more concrete numeric examples.

---

#### Example 2: Two-Head Attention with Full Numerical Details
In this example, we will provide explicit numbers for a 2-head setup. We will assume each of $Q, K, V$ has shape $(3,4)$: there are 3 “tokens” (or time steps), each with a hidden size of 4. We split that hidden size into 2 heads, each with size 2.

**Step 0: Define inputs and parameters**  
Let

$$
Q = \begin{bmatrix}
1 & 2 & 1 & 0\\
0 & 1 & 1 & 1\\
1 & 0 & 2 & 1
\end{bmatrix},\quad
K = \begin{bmatrix}
1 & 1 & 0 & 2\\
2 & 1 & 1 & 0\\
0 & 1 & 1 & 1
\end{bmatrix},\quad
V = \begin{bmatrix}
1 & 1 & 0 & 0\\
0 & 2 & 1 & 1\\
1 & 1 & 2 & 2
\end{bmatrix}.
$$

We also define the projection matrices for the two heads. For simplicity, we assume each projection matrix has shape $(4,2)$ (since we project dimension 4 down to dimension 2), and $W^O$ will have shape $(4,4)$ to map the concatenated result $(3,4)$ back to $(3,4)$.

Let’s define:

$$
W^Q_1 = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 0\\
0 & 1
\end{bmatrix}, \quad
W^K_1 = \begin{bmatrix}
1 & 0\\
0 & 1\\
0 & 1\\
1 & 0
\end{bmatrix}, \quad
W^V_1 = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 0\\
0 & 1
\end{bmatrix},
$$

$$
W^Q_2 = \begin{bmatrix}
0 & 1\\
1 & 0\\
1 & 1\\
0 & 0
\end{bmatrix}, \quad
W^K_2 = \begin{bmatrix}
0 & 1\\
1 & 0\\
1 & 0\\
1 & 1
\end{bmatrix}, \quad
W^V_2 = \begin{bmatrix}
0 & 1\\
1 & 1\\
0 & 1\\
1 & 0
\end{bmatrix}.
$$

And let:

$$
W^O = \begin{bmatrix}
1 & 0 & 0 & 1\\
0 & 1 & 1 & 0\\
1 & 0 & 1 & 0\\
0 & 1 & 0 & 1
\end{bmatrix}.
$$

We will go step by step.

---

**Step 1: Compute $Q_1, K_1, V_1$ for Head 1**  

$$
Q_1 = Q \times W^Q_1,\quad
K_1 = K \times W^K_1,\quad
V_1 = V \times W^V_1.
$$

- $Q_1 = Q W^Q_1$.  
  Each row of $Q$ is multiplied by $W^Q_1$:

  $$
  Q = \begin{bmatrix}
  1 & 2 & 1 & 0\\
  0 & 1 & 1 & 1\\
  1 & 0 & 2 & 1
  \end{bmatrix},
  \quad
  W^Q_1 = \begin{bmatrix}
  1 & 0\\
  0 & 1\\
  1 & 0\\
  0 & 1
  \end{bmatrix}.
  $$

  - Row 1 of $Q$: $[1,2,1,0]$

    $$
    [1,2,1,0]
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1*1 + 2*0 + 1*1 + 0*0,\; 1*0 + 2*1 + 1*0 + 0*1]
    =
    [2,\;2].
    $$

  - Row 2: $[0,1,1,1]$

    $$
    [0,1,1,1]
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1,\;2].
    $$

  - Row 3: $[1,0,2,1]$

    $$
    [1,0,2,1]
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [3,\;1].
    $$

  Thus,

  $$
  Q_1 = \begin{bmatrix}
  2 & 2\\
  1 & 2\\
  3 & 1
  \end{bmatrix}.
  $$

- $K_1 = K W^K_1$.  

  $$
  K = \begin{bmatrix}
  1 & 1 & 0 & 2\\
  2 & 1 & 1 & 0\\
  0 & 1 & 1 & 1
  \end{bmatrix},\quad
  W^K_1 = \begin{bmatrix}
  1 & 0\\
  0 & 1\\
  0 & 1\\
  1 & 0
  \end{bmatrix}.
  $$

  - Row 1: $[1,1,0,2]$

    $$
    [1,1,0,2]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [3,\;1].
    $$

  - Row 2: $[2,1,1,0]$

    $$
    [2,1,1,0]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [2,\;2].
    $$

  - Row 3: $[0,1,1,1]$

    $$
    [0,1,1,1]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [1,\;2].
    $$

  So,

  $$
  K_1 = \begin{bmatrix}
  3 & 1\\
  2 & 2\\
  1 & 2
  \end{bmatrix}.
  $$

- $V_1 = V W^V_1$.  

  $$
  V = \begin{bmatrix}
  1 & 1 & 0 & 0\\
  0 & 2 & 1 & 1\\
  1 & 1 & 2 & 2
  \end{bmatrix},\quad
  W^V_1 = \begin{bmatrix}
  1 & 0\\
  0 & 1\\
  1 & 0\\
  0 & 1
  \end{bmatrix}.
  $$

  - Row 1: $[1,1,0,0]$

    $$
    [1,1,0,0] \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1,\;1].
    $$

  - Row 2: $[0,2,1,1]$

    $$
    [0,2,1,1]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1,\;3].
    $$

  - Row 3: $[1,1,2,2]$

    $$
    [1,1,2,2]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [3,\;3].
    $$

  Therefore,

  $$
  V_1 = \begin{bmatrix}
  1 & 1\\
  1 & 3\\
  3 & 3
  \end{bmatrix}.
  $$

---

**Step 2: Compute $Q_2, K_2, V_2$ for Head 2**  

$$
Q_2 = Q \times W^Q_2,\quad
K_2 = K \times W^K_2,\quad
V_2 = V \times W^V_2.
$$

- $Q_2 = Q W^Q_2$:

  $$
  W^Q_2 = \begin{bmatrix}
  0 & 1\\
  1 & 0\\
  1 & 1\\
  0 & 0
  \end{bmatrix}.
  $$

  - Row 1 $[1,2,1,0]$:

    $$
    [1,2,1,0]
    \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 1\\
    0 & 0
    \end{bmatrix}
    =
    [3,\;2].
    $$

  - Row 2 $[0,1,1,1]$:

    $$
    [0,1,1,1]
    \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 1\\
    0 & 0
    \end{bmatrix}
    =
    [2,\;1].
    $$

  - Row 3 $[1,0,2,1]$:

    $$
    [1,0,2,1]
    \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 1\\
    0 & 0
    \end{bmatrix}
    =
    [2,\;3].
    $$

  Hence,

  $$
  Q_2 = \begin{bmatrix}
  3 & 2\\
  2 & 1\\
  2 & 3
  \end{bmatrix}.
  $$

- $K_2 = K W^K_2$:

  $$
  W^K_2 = \begin{bmatrix}
  0 & 1\\
  1 & 0\\
  1 & 0\\
  1 & 1
  \end{bmatrix}.
  $$

  - Row 1 $[1,1,0,2]$:

    $$
    [1,1,0,2] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 0\\
    1 & 1
    \end{bmatrix}
    =
    [3,\;3].
    $$

  - Row 2 $[2,1,1,0]$:

    $$
    [2,1,1,0] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 0\\
    1 & 1
    \end{bmatrix}
    =
    [2,\;2].
    $$

  - Row 3 $[0,1,1,1]$:

    $$
    [0,1,1,1] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 0\\
    1 & 1
    \end{bmatrix}
    =
    [3,\;1].
    $$

  So,

  $$
  K_2 = \begin{bmatrix}
  3 & 3\\
  2 & 2\\
  3 & 1
  \end{bmatrix}.
  $$

- $V_2 = V W^V_2$:

  $$
  W^V_2 = \begin{bmatrix}
  0 & 1\\
  1 & 1\\
  0 & 1\\
  1 & 0
  \end{bmatrix}.
  $$

  - Row 1 $[1,1,0,0]$:

    $$
    [1,1,0,0] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [1,\;2].
    $$

  - Row 2 $[0,2,1,1]$:

    $$
    [0,2,1,1] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [3,\;3].
    $$

  - Row 3 $[1,1,2,2]$:

    $$
    [1,1,2,2] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [3,\;4].
    $$

  Thus,

  $$
  V_2 = \begin{bmatrix}
  1 & 2\\
  3 & 3\\
  3 & 4
  \end{bmatrix}.
  $$

---

**Step 3: Compute each head’s Scaled Dot-Product Attention**  

We now have for head 1:

$$
Q_1 = \begin{bmatrix}2 & 2\\1 & 2\\3 & 1\end{bmatrix},\;
K_1 = \begin{bmatrix}3 & 1\\2 & 2\\1 & 2\end{bmatrix},\;
V_1 = \begin{bmatrix}1 & 1\\1 & 3\\3 & 3\end{bmatrix}.
$$

Similarly for head 2:

$$
Q_2 = \begin{bmatrix}3 & 2\\2 & 1\\2 & 3\end{bmatrix},\;
K_2 = \begin{bmatrix}3 & 3\\2 & 2\\3 & 1\end{bmatrix},\;
V_2 = \begin{bmatrix}1 & 2\\3 & 3\\3 & 4\end{bmatrix}.
$$

Assume each key vector dimension is $d_k = 2$. Hence the scale is $\frac{1}{\sqrt{2}} \approx 0.707$.

- **Head 1**:  
  1. $Q_1 K_1^T$.  

     $K_1^T$ is

     $$
     \begin{bmatrix}
     3 & 2 & 1\\
     1 & 2 & 2
     \end{bmatrix}.
     $$

     $$
     Q_1 K_1^T =
     \begin{bmatrix}
     2 & 2\\
     1 & 2\\
     3 & 1
     \end{bmatrix}
     \times
     \begin{bmatrix}
     3 & 2 & 1\\
     1 & 2 & 2
     \end{bmatrix}
     =
     \begin{bmatrix}
     8 & 8 & 6\\
     5 & 6 & 5\\
     10 & 8 & 5
     \end{bmatrix}.
     $$

  2. Scale: $\frac{Q_1 K_1^T}{\sqrt{2}}$:

     $$
     \approx
     \begin{bmatrix}
     5.66 & 5.66 & 4.24\\
     3.54 & 4.24 & 3.54\\
     7.07 & 5.66 & 3.54
     \end{bmatrix}.
     $$

  3. Apply softmax row-wise (approx results after exponentiation and normalization):

     $$
     \alpha_1 \approx
     \begin{bmatrix}
     0.45 & 0.45 & 0.11\\
     0.25 & 0.50 & 0.25\\
     0.79 & 0.19 & 0.02
     \end{bmatrix}.
     $$

  4. Multiply by $V_1$:

     $$
     \text{head}_1 = \alpha_1 \times V_1.
     $$

     Approximating:

     $$
     \text{head}_1 \approx
     \begin{bmatrix}
     1.23 & 2.13\\
     1.50 & 2.50\\
     1.04 & 1.42
     \end{bmatrix}.
     $$

- **Head 2**:  
  1. $Q_2 K_2^T$.  

     $$
     Q_2 = \begin{bmatrix}
     3 & 2\\
     2 & 1\\
     2 & 3
     \end{bmatrix},\quad
     K_2 = \begin{bmatrix}
     3 & 3\\
     2 & 2\\
     3 & 1
     \end{bmatrix}.
     $$

     Then

     $$
     K_2^T = \begin{bmatrix}
     3 & 2 & 3\\
     3 & 2 & 1
     \end{bmatrix}.
     $$

     $$ 
     Q_2 K_2^T =
     \begin{bmatrix}
     15 & 10 & 11\\
     9  & 6  & 7\\
     15 & 10 & 9
     \end{bmatrix}.
     $$

  2. Scale: multiply by $1/\sqrt{2} \approx 0.707$:

     $$
     \approx
     \begin{bmatrix}
     10.61 & 7.07 & 7.78\\
     6.36 & 4.24 & 4.95\\
     10.61 & 7.07 & 6.36
     \end{bmatrix}.
     $$

  3. Softmax row-wise (approx):

     $$
     \alpha_2 \approx
     \begin{bmatrix}
     0.92 & 0.03 & 0.05\\
     0.73 & 0.09 & 0.18\\
     0.96 & 0.03 & 0.01
     \end{bmatrix}.
     $$

  4. Multiply by $V_2$:

     $$
     V_2 = \begin{bmatrix}
     1 & 2\\
     3 & 3\\
     3 & 4
     \end{bmatrix}.
     $$

     Approximating:

     $$
     \text{head}_2 \approx
     \begin{bmatrix}
     1.16 & 2.13\\
     1.53 & 2.45\\
     1.09 & 2.06
     \end{bmatrix}.
     $$

---

**Step 4: Concatenate and apply $W^O$**  
We now concatenate $\text{head}_1$ and $\text{head}_2$ horizontally to form a $(3 \times 4)$ matrix:

$$
\text{Concat}(\text{head}_1, \text{head}_2) =
\begin{bmatrix}
1.23 & 2.13 & 1.16 & 2.13 \\
1.50 & 2.50 & 1.53 & 2.45 \\
1.04 & 1.42 & 1.09 & 2.06
\end{bmatrix}.
$$

Finally, multiply by $W^O$ $(4 \times 4)$:

$$
\text{Output} = (\text{Concat}(\text{head}_1, \text{head}_2)) \times W^O.
$$

Where

$$
W^O = \begin{bmatrix}
1 & 0 & 0 & 1\\
0 & 1 & 1 & 0\\
1 & 0 & 1 & 0\\
0 & 1 & 0 & 1
\end{bmatrix}.
$$

We can do a row-by-row multiplication to get the final multi-head attention output (details omitted for brevity).

---

#### Example 3: Three-Head Attention with Another Set of Numbers (Short Demonstration)
For completeness, suppose we wanted $h=3$ heads, each of dimension $\frac{d_{\text{model}}}{3}$. The steps are exactly the same:

1. Project $Q, K, V$ into three subspaces via $W^Q_i, W^K_i, W^V_i$.
2. Perform scaled dot-product attention for each head:  
   $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$.
3. Concatenate all heads: $\text{Concat}(\text{head}_1, \text{head}_2, \text{head}_3)$.
4. Multiply by $W^O$.

Each numeric calculation is analogous to the 2-head case—just with different shapes (e.g., each head might have dimension 4/3 if the original dimension is 4, which typically would be handled with rounding or a slightly different total dimension). The procedure remains identical in principle.

---

## Position-Wise Feed-Forward Networks
Each layer in a Transformer includes a position-wise feed-forward network (FFN) that applies a linear transformation and activation to each position independently:

$$
\text{FFN}(x) = \max(0,\; xW_1 + b_1)\, W_2 + b_2,
$$

where $\max(0, \cdot)$ is the ReLU activation function.

### Example: Numerical Computation of the Feed-Forward Network
Let

$$
x = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix},\quad
W_1 = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},\quad
b_1 = \begin{bmatrix}
0 & 1
\end{bmatrix},\quad
W_2 = \begin{bmatrix}
1 & 0 \\
2 & 1
\end{bmatrix},\quad
b_2 = \begin{bmatrix}
1 & -1
\end{bmatrix}.
$$

1. **Compute $xW_1 + b_1$**  
   - Row 1: $[1, 0]$

     $$
     [1, 0]
     \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
     = [1, 1],
     $$

     then add $[0, 1]$ to get $[1, 2]$.

   - Row 2: $[0, 1]$

     $$
     [0,1]\times
     \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}
     = [0, 1],
     $$

     plus $[0, 1]$ = $[0, 2]$.

   - Row 3: $[1,1]$

     $$
     [1,1]\times
     \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}
     = [1, 2],
     $$

     plus $[0, 1]$ = $[1, 3]$.

   So

   $$
   X_1 =
   \begin{bmatrix}
   1 & 2\\
   0 & 2\\
   1 & 3
   \end{bmatrix}.
   $$

2. **ReLU activation**  
   $\max(0, X_1)$ leaves nonnegative elements unchanged. All entries are already $\ge0$, so

   $$
   \text{ReLU}(X_1) = X_1.
   $$

3. **Multiply by $W_2$ and add $b_2$**  

   $$
   W_2 =
   \begin{bmatrix}
   1 & 0\\
   2 & 1
   \end{bmatrix},\quad
   b_2 = [1, -1].
   $$
   
   $$
   X_2 = X_1 W_2.
   $$

   - Row 1 of $X_1$: $[1,2]$

     $$
     [1,2]
     \begin{bmatrix}
     1\\2
     \end{bmatrix}
     = 1*1 +2*2=5, \quad
     [1,2]
     \begin{bmatrix}
     0\\1
     \end{bmatrix}
     = 0 +2=2.
     $$
     So $[5,2]$.

   - Row 2: $[0,2]$

     $$
     [0,2]
     \begin{bmatrix}1\\2\end{bmatrix}=4,\quad
     [0,2]
     \begin{bmatrix}0\\1\end{bmatrix}=2.
     $$

   - Row 3: $[1,3]$

     $$
     [1,3]\begin{bmatrix}1\\2\end{bmatrix}=1+6=7,\quad
     [1,3]\begin{bmatrix}0\\1\end{bmatrix}=0+3=3.
     $$

   Thus

   $$
   X_2 = \begin{bmatrix}
   5 & 2\\
   4 & 2\\
   7 & 3
   \end{bmatrix}.
   $$

   Add $b_2=[1,-1]$:

   $$
   X_2 + b_2 =
   \begin{bmatrix}
   6 & 1\\
   5 & 1\\
   8 & 2
   \end{bmatrix}.
   $$

**Final Output**:

$$
\begin{bmatrix}
6 & 1\\
5 & 1\\
8 & 2
\end{bmatrix}.
$$

---

## Training and Optimization

### Optimizer and Learning Rate Scheduling
Transformers commonly use Adam or AdamW, combined with a piecewise learning rate scheduling strategy:

$$
l_{\text{rate}} = d_{\text{model}}^{-0.5}
\cdot
\min\bigl(\text{step}_\text{num}^{-0.5},\;
\text{step}_\text{num}\times \text{warmup}_\text{steps}^{-1.5}\bigr),
$$

where:
- $d_{\text{model}}$ is the hidden dimension.
- $\text{step}_\text{num}$ is the current training step.
- $\text{warmup}_\text{steps}$ is the number of warmup steps.

---

## Conclusion
The Transformer architecture has become a foundational model in modern deep learning, showing remarkable performance in NLP, computer vision, and multimodal applications. Its ability to capture long-range dependencies, combined with high parallelizability and scalability, has inspired a diverse range of research directions and practical systems. Ongoing work continues to explore ways to improve Transformer efficiency, adapt it to new scenarios, and enhance model interpretability.

---

Paper Reading: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

Below is a **paragraph-by-paragraph** (or subsection-by-subsection) **markdown** file that first **re-states** (“recaps”) each portion of the paper *Attention Is All You Need* and then **comments** on or explains that portion in more detail. Each header corresponds to a main section or subsection from the original text. The original content has been paraphrased and condensed to be more concise, but the overall structure and meaning are preserved. 

> **Note**: The original paper, “Attention Is All You Need,” was published by Ashish Vaswani et al. This markdown document is for educational purposes, offering an English re-statement of each section followed by commentary.

---

# Paper Reading: Attention Is All You Need

## Authors and Affiliations

**Original (Condensed)**
> *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.*  
> *Affiliations: Google Brain, Google Research, University of Toronto.*  

**Recap**  
A group of researchers from Google Brain, Google Research, and the University of Toronto propose a new network architecture that relies solely on attention mechanisms for sequence transduction tasks such as machine translation.

**Commentary**  
This highlights that multiple authors, each potentially focusing on different aspects—model design, optimization, and experiments—came together to create what is now often referred to as the “Transformer” architecture.

---

## Abstract

**Original (Condensed)**
> The dominant sequence transduction models use recurrent or convolutional neural networks (often with attention). This paper proposes the Transformer, which is based entirely on attention mechanisms. It does away with recurrence and convolutions entirely. Experiments on two machine translation tasks show the model is both high-performing in terms of BLEU score and more parallelizable. The paper reports a new state-of-the-art BLEU on WMT 2014 English-German (28.4) and a strong single-model result on English-French (41.8), trained much faster than previous approaches. The Transformer also generalizes well to other tasks, e.g., English constituency parsing.*

**Recap**  
The paper’s abstract introduces a novel approach called the Transformer. It uses only attention (no RNNs or CNNs) for tasks like machine translation and shows exceptional speed and accuracy results.

**Commentary**  
This is a seminal innovation in deep learning for language processing. Removing recurrence (like LSTM layers) and convolutions makes training highly parallelizable, dramatically reducing training time. At the same time, it achieves superior or comparable performance on well-known benchmarks. The abstract also hints that the Transformer concept could generalize to other sequential or structured tasks.

---

## 1 Introduction

**Original (Condensed)**
> Recurrent neural networks (RNNs), particularly LSTM or GRU models, have set the standard in sequence modeling and transduction tasks. However, they process input sequentially, limiting parallelization. Attention mechanisms have improved performance in tasks like translation, but they have traditionally been used on top of recurrent networks. This paper proposes a model that relies entirely on attention—called the Transformer—removing the need for recurrence or convolutional architectures. The result is a model that learns global dependencies and can be trained more efficiently.*

**Recap**  
The introduction situates the proposed Transformer within the history of neural sequence modeling: first purely recurrent approaches, then RNN+attention, and finally a pure-attention approach. The authors observe that while recurrent models handle sequences effectively, they rely on step-by-step processing. This strongly limits parallel computation. The Transformer’s innovation is to dispense with recurrences altogether.

**Commentary**  
The introduction highlights a major bottleneck in typical RNN-based models: the inability to parallelize across time steps in a straightforward way. Traditional attention over RNN outputs is still useful, but the authors propose a more radical approach, removing recurrences and using attention everywhere. This sets the stage for a highly parallelizable model that can scale better to longer sequences, given sufficient memory and computational resources.

In-Course Question 1: What is the number of dimensionality of the transformer's query embeddings designed in this paper.

---

## 2 Background

**Original (Condensed)**
> Efforts to reduce the sequential computation have led to alternatives like the Extended Neural GPU, ByteNet, and ConvS2S, which use convolutional networks for sequence transduction. However, even with convolution, the distance between two positions can be large in deep stacks, potentially making it harder to learn long-range dependencies. Attention mechanisms have been used for focusing on specific positions in a sequence, but typically in conjunction with RNNs. The Transformer is the first purely attention-based model for transduction.*

**Recap**  
The background section covers attempts to speed up sequence modeling, including convolution-based architectures. While they improve speed and are more parallelizable than RNNs, they still can have challenges with long-range dependencies. Attention can address such dependencies, but before this paper, it was usually combined with recurrent models.

**Commentary**  
This background motivates why researchers might try to eliminate recurrence and convolution entirely. If attention alone can handle dependency modeling, then the path length between any two positions in a sequence is effectively shorter. This suggests simpler, faster training and potentially better performance.

---

## 3 Model Architecture

The Transformer follows an **encoder-decoder** structure, but with self-attention replacing recurrences or convolutions.  

### 3.1 Encoder and Decoder Stacks

**Original (Condensed)**
> The encoder is composed of N identical layers; each layer has (1) a multi-head self-attention sub-layer, and (2) a position-wise feed-forward network. A residual connection is employed around each of these, followed by layer normalization. The decoder also has N identical layers with an additional sub-layer for attention over the encoder output. A masking scheme ensures each position in the decoder can only attend to positions before it (causal masking).*

**Recap**  
- **Encoder**: Stack of N layers. Each layer has:
  1. Self-attention
  2. Feed-forward  
  Plus skip (residual) connections and layer normalization.
- **Decoder**: Similar stack but also attends to the encoder output. Additionally, the decoder masks future positions to preserve the autoregressive property.

**Commentary**  
This design is highly modular: each layer is built around multi-head attention and a feed-forward block. The skip connections help with training stability, and layer normalization is known to speed up convergence. The causal masking in the decoder is crucial for generation tasks such as translation, ensuring that the model cannot “peek” at future tokens.

---

### 3.2 Attention

**Original (Condensed)**
> An attention function maps a query and a set of key-value pairs to an output. We use a “Scaled Dot-Product Attention,” where the dot products between query and key vectors are scaled by the square root of the dimension. A softmax yields weights for each value. We also introduce multi-head attention: queries, keys, and values are linearly projected h times, each head performing attention in parallel, then combined.*

**Recap**  
- **Scaled Dot-Product Attention**: Computes attention weights via `softmax((QK^T) / sqrt(d_k)) * V`.
- **Multi-Head Attention**: Instead of a single attention, we project Q, K, V into multiple sub-spaces (heads), do attention in parallel, then concatenate.

**Commentary**  
Dot-product attention is computationally efficient and can be parallelized easily. The scaling factor 1/√(d_k) helps mitigate large magnitude dot products when the dimensionality of keys/queries is big. Multiple heads allow the model to look at different positions/relationships simultaneously, which helps capture various types of information (e.g., syntax, semantics).

---

### 3.3 Position-wise Feed-Forward Networks

**Original (Condensed)**
> Each layer in the encoder and decoder has a feed-forward network that is applied to each position separately and identically, consisting of two linear transformations with a ReLU in between.*

**Recap**  
After multi-head attention, each token’s representation goes through a small “fully connected” or “feed-forward” sub-network. This is done independently per position.

**Commentary**  
This structure ensures that after attention-based mixing, each position is then transformed in a non-linear way. It is reminiscent of using small per-position multi-layer perceptrons to refine each embedding.

---

### 3.4 Embeddings and Softmax

**Original (Condensed)**
> Token embeddings and the final output linear transformation share the same weight matrix (with a scaling factor). The model uses learned embeddings to convert input and output tokens to vectors of dimension d_model.*

**Recap**  
The model uses standard embedding layers for tokens and ties the same weights in both the embedding and the pre-softmax projection. This helps with parameter efficiency and sometimes improves performance.

**Commentary**  
Weight tying is a known trick that can save on parameters and can help the embedding space align with the output space in generative tasks.

---

### 3.5 Positional Encoding

**Original (Condensed)**
> Because there is no recurrence or convolution, the Transformer needs positional information. The paper adds a sinusoidal positional encoding to the input embeddings, allowing the model to attend to relative positions. Learned positional embeddings perform similarly, but sinusoidal encodings might let the model generalize to sequence lengths not seen during training.*

**Recap**  
The Transformer adds sine/cosine signals of varying frequencies to the embeddings so that each position has a unique pattern. This is essential to preserve ordering information.

**Commentary**  
Without positional encodings, the self-attention mechanism would treat input tokens as an unstructured set. Positional information ensures that the model knows how tokens relate to one another in a sequence.

---

## 4 Why Self-Attention

**Original (Condensed)**
> The authors compare self-attention to recurrent and convolutional layers in terms of computation cost and how quickly signals can travel between distant positions in a sequence. Self-attention is more parallelizable and has O(1) maximum path length (all tokens can attend to all others in one step). Convolutions and recurrences require multiple steps to connect distant positions. This can help with learning long-range dependencies.*

**Recap**  
Self-attention:
- Parallelizable across sequence positions.
- Constant number of sequential operations per layer.
- Short paths between positions -> easier to learn long-range dependencies.

**Commentary**  
The authors argue that self-attention layers are efficient (especially when sequence length is not extremely large) and effective at modeling dependencies. This is a key motivation for the entire design.

---

In-class question: What is the probability assigned to the ground-truth class in the ground-truth distribution after label smoothing when training the Transformer in the default setting of this paper?

## 5 Training

### 5.1 Training Data and Batching

**Original (Condensed)**
> The authors use WMT 2014 English-German (about 4.5M sentence pairs) and English-French (36M pairs). They use subword tokenization (byte-pair encoding or word-piece) to handle large vocabularies. Training batches contain roughly 25k source and 25k target tokens.*

**Recap**  
They describe the datasets and how the text is batched using subword units. This avoids issues with out-of-vocabulary tokens.

**Commentary**  
Subword tokenization was pivotal in neural MT systems because it handles rare words well. Batching by approximate length helps the model train more efficiently and speeds up training on GPUs.

---

### 5.2 Hardware and Schedule

**Original (Condensed)**
> They trained on a single machine with 8 NVIDIA P100 GPUs. The base model was trained for 100k steps (about 12 hours), while the bigger model took around 3.5 days. Each training step for the base model took ~0.4 seconds on this setup.*

**Recap**  
Base models train surprisingly quickly—only about half a day for high-quality results. The big model uses more parameters and trains longer.

**Commentary**  
This training time is significantly shorter than earlier neural MT models, demonstrating one practical advantage of a highly parallelizable architecture.

---

### 5.3 Optimizer

**Original (Condensed)**
> The paper uses the Adam optimizer with specific hyperparameters (β1=0.9, β2=0.98, ε=1e-9). The learning rate increases linearly for the first 4k steps, then decreases proportionally to step^-0.5.*

**Recap**  
A custom learning-rate schedule is used, with a “warm-up” phase followed by a decay. This is crucial to stabilize training early on and then adapt to a more standard rate.

**Commentary**  
This “Noam” learning rate schedule (as often called) is well-known in the community. It boosts the learning rate once the model is more confident, yet prevents divergence early on.

---

### 5.4 Regularization

**Original (Condensed)**
> Three types of regularization: (1) Dropout after sub-layers and on embeddings, (2) label smoothing of 0.1, (3) early stopping / checkpoint averaging (not explicitly described here but implied). Label smoothing slightly hurts perplexity but improves translation BLEU.*

**Recap**  
- Dropout helps avoid overfitting.  
- Label smoothing makes the model less certain about each token prediction, improving generalization.

**Commentary**  
By forcing the model to distribute probability mass across different tokens, label smoothing can prevent the network from becoming overly confident in a small set of predictions, thus improving real-world performance metrics like BLEU.

---

## 6 Results

### 6.1 Machine Translation

**Original (Condensed)**
> On WMT 2014 English-German, the big Transformer achieved 28.4 BLEU, surpassing all previously reported results (including ensembles). On English-French, it got 41.8 BLEU with much less training cost compared to other models. The base model also outperforms previous single-model baselines.*

**Recap**  
Transformer sets a new SOTA on English-German and matches/exceeds on English-French with vastly reduced training time.

**Commentary**  
This was a landmark result, as both speed and quality improved. The authors highlight not just the performance, but the “cost” in terms of floating-point operations, showing how the Transformer is more efficient.

---

### 6.2 Model Variations

**Original (Condensed)**
> They explore different hyperparameters, e.g., number of attention heads, dimension of queries/keys, feed-forward layer size, and dropout. They find that more heads can help but too many heads can degrade performance. Bigger dimensions improve results at the expense of more computation.*

**Recap**  
Experiments confirm that the Transformer’s performance scales with model capacity. Properly tuned dropout is vital. Both sinusoidal and learned positional embeddings perform comparably.

**Commentary**  
This section is valuable for practitioners, as it provides insight into how to adjust model size and regularization. It also confirms that the approach is flexible.

---

### 6.3 English Constituency Parsing

**Original (Condensed)**
> They show that the Transformer can also tackle English constituency parsing, performing competitively with top models. On the WSJ dataset, it achieves strong results, and in a semi-supervised setting, it is even more impressive.*

**Recap**  
It isn’t just about machine translation: the model generalizes to other tasks with structural dependencies, illustrating self-attention’s adaptability.

**Commentary**  
Constituency parsing requires modeling hierarchical relationships in sentences. Transformer’s ability to attend to any part of the input helps capture these structures without specialized RNNs or grammar-based methods.

---

## 7 Conclusion

**Original (Condensed)**
> The Transformer architecture relies entirely on self-attention, providing improved parallelization and, experimentally, new state-of-the-art results in machine translation. The paper suggests applying this approach to other tasks and modalities, possibly restricting attention to local neighborhoods for efficiency with large sequences. The code is made available in an open-source repository.*

**Recap**  
The authors close by reiterating how self-attention replaces recurrence and convolution, giving strong speed advantages. They encourage investigating how to adapt the architecture to other domains and tasks.

**Commentary**  
This conclusion underscores the paper’s broad impact. After publication, the Transformer rapidly became the foundation of many subsequent breakthroughs, including large-scale language models. Future directions—like local attention for very long sequences—have since seen extensive research.

---

## References

*(Original references are long and primarily list papers on neural networks, attention, convolutional models, etc. Below is a very brief, high-level mention.)*

**Recap**  
The references include prior works on RNN-based machine translation, convolutional approaches, attention mechanisms, and optimization techniques.

**Commentary**  
They form a comprehensive backdrop for the evolution of neural sequence modeling, highlighting both the developments that led to the Transformer and the new directions it subsequently inspired.

---

# Overall Commentary

The paper *Attention Is All You Need* revolutionized natural language processing by introducing a purely attention-based model (the Transformer). Its core contributions can be summarized as:

1. **Eliminating Recurrence and Convolution**: Replacing them with multi-head self-attention to model dependencies in a single step.
2. **Superior Performance and Efficiency**: Achieving state-of-the-art results on crucial MT tasks faster than prior methods.
3. **Generalization**: Showing that the model concept extends beyond MT to other tasks, e.g., parsing.

This architecture laid the groundwork for many subsequent techniques, including BERT, GPT, and other large language models. The key takeaway is that attention mechanisms alone—when used in a multi-layer, multi-head framework—suffice to capture both local and global information in sequences, drastically improving efficiency and performance in a wide range of NLP tasks.

---