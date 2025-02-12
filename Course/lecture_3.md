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
  <img src="./Course/attention_example.svg" width="50%">
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

1. **Compute $QK^T$**  
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

**Final Output**: $5$.


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

# Detailed Analysis: Attention Is All You Need

## 1. Introduction
The paper opens by addressing the state of sequence transduction models circa 2017, dominated by recurrent neural networks (RNNs) particularly LSTM and gated RNN variants. The authors immediately identify the critical limitation of these architectures: their reliance on sequential computation. This sequential nature creates a fundamental constraint where computation must be performed step by step, making it impossible to parallelize processing within training examples. As sequence lengths grow, this becomes particularly problematic due to memory constraints limiting batch processing.

The introduction effectively sets up the paper's key contribution by highlighting how attention mechanisms, while proven effective in various sequence modeling tasks, had primarily been used as supplements to RNN architectures. The authors then present their novel solution: the Transformer, which completely eliminates recurrence in favor of attention mechanisms. The introduction concludes with a powerful statement about the model's practical benefits - achieving superior translation quality while being significantly more parallelizable and requiring far less training time.

## 2. Background
The background section provides crucial context by discussing previous attempts to reduce sequential computation in neural networks. The authors examine three key predecessors: the Extended Neural GPU, ByteNet, and ConvS2S. All these models used convolutional neural networks as their fundamental building blocks, computing hidden representations in parallel for all input and output positions. However, they identify a critical limitation in these approaches: the number of operations required to relate signals grows with the distance between positions - either linearly (ConvS2S) or logarithmically (ByteNet).

The section then introduces self-attention (or intra-attention), positioning it within existing literature on reading comprehension, abstractive summarization, and textual entailment. The authors note that while self-attention had shown promise, it had primarily been used alongside recurrent networks. This historical context effectively highlights the Transformer's novelty as the first model to rely entirely on self-attention for computing representations.

## 3. Model Architecture
This comprehensive section details the Transformer's novel architecture. The authors begin with the high-level encoder-decoder structure, then methodically break down each component:

The encoder stack consists of 6 identical layers, each containing two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network. They employ residual connections and layer normalization, with outputs of dimension dmodel = 512. The decoder mirrors this structure but adds a third sub-layer for attention over the encoder stack.

The attention mechanism, particularly the "Scaled Dot-Product Attention," represents the paper's core innovation. The authors provide a detailed mathematical explanation of how attention weights are computed using queries, keys, and values, including the crucial scaling factor 1/√dk. The multi-head attention allows the model to jointly attend to information from different representation subspaces, effectively creating multiple "views" of the attention mechanism.

The position-wise feed-forward networks consist of two linear transformations with a ReLU activation, applied identically to each position. The authors detail specific architectural choices like dimensionality (dmodel = 512, dff = 2048) and the sharing of embedding weights between layers.

The positional encoding section presents an elegant solution to the problem of sequence order, using sine and cosine functions of different frequencies. This allows the model to attend to relative positions through simple linear transformations of these encodings.

## 4. Why Self-Attention
This section provides a thorough comparative analysis of self-attention versus recurrent and convolutional layers. The authors evaluate three key aspects: computational complexity per layer, parallelization potential, and path length between long-range dependencies. They demonstrate that self-attention requires a constant number of sequentially executed operations, while recurrent layers require O(n) operations. The complexity analysis shows that self-attention is more efficient when sequence length is smaller than representation dimensionality, which is common in most practical applications.

The section also highlights self-attention's advantage in modeling long-range dependencies, as all positions are connected through a single step, unlike convolutional or recurrent approaches where information must flow through multiple layers or time steps.

## 5. Training
The training section provides detailed information about the model's implementation and optimization. The authors describe their use of the WMT 2014 English-German dataset (4.5 million sentence pairs) and English-French dataset (36M sentences), detailing preprocessing steps including byte-pair encoding for vocabulary creation.

The training protocol uses 8 NVIDIA P100 GPUs, with precise timing information for both base and large models. The optimizer (Adam) parameters are carefully specified, along with a custom learning rate schedule featuring a warmup period. The authors detail three types of regularization: residual dropout, attention dropout, and label smoothing.

## 6. Results
The results section presents comprehensive empirical evidence of the Transformer's effectiveness. On the WMT 2014 English-to-German translation task, the model achieves a BLEU score of 28.4, surpassing the previous state-of-the-art by over 2 BLEU points. For English-to-French translation, it reaches 41.8 BLEU, establishing new state-of-the-art performance while using significantly less computational resources than previous best models.

The authors also demonstrate the model's generalization capabilities through English constituency parsing experiments, showing competitive performance even without task-specific tuning. They provide detailed ablation studies examining the impact of various architectural choices, including the number of attention heads, key/value dimensions, and model depth.

The conclusion effectively ties together the paper's contributions, emphasizing not just the model's superior performance but also its practical advantages in training efficiency and its potential for future applications beyond text processing.