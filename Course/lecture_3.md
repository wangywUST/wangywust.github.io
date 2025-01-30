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

### Scaled Dot-Product Attention
Given a query matrix \(Q\), key matrix \(K\), and value matrix \(V\), the attention formula is:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\Bigl( \frac{QK^T}{\sqrt{d_k}} \Bigr)V
\]

where \(d_k\) is the dimensionality of the key vectors (often the same as the query dimensionality).

#### Example 1: Detailed Numerical Computation
Suppose we have the following matrices (small dimensions chosen for illustrative purposes):

\[
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
\]

1. **Compute \(QK^T\)**  
   According to the example setup:
   \[
   QK^T = \begin{bmatrix}
   1 & 1 & 1 \\
   0 & 1 & 0 \\
   1 & 2 & 1
   \end{bmatrix}
   \]

2. **Scale by \(\sqrt{d_k}\)**  
   Here, \(d_k = 2\). Thus, \(\sqrt{2} \approx 1.41\). So,
   \[
   \frac{QK^T}{\sqrt{2}} \approx
   \begin{bmatrix}
   0.71 & 0.71 & 0.71 \\
   0    & 0.71 & 0    \\
   0.71 & 1.41 & 0.71
   \end{bmatrix}
   \]

3. **Apply softmax row-wise**  
   The softmax of a vector \(\mathbf{x}\) is given by
   \[
   \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}.
   \]  
   - Row 1: \([0.71, 0.71, 0.71]\)  
     Since all elements are equal, each is \(\approx e^{0.71} \approx 2.03\). The sum is \(6.09\). Hence, each entry becomes \(\frac{2.03}{6.09} \approx 0.33\).  
   - Row 2: \([0, 0.71, 0]\)  
     \(e^{0} = 1,\; e^{0.71} \approx 2.03\). The sum is \(4.03\). So we get \([0.25, 0.50, 0.25]\).  
   - Row 3: \([0.71, 1.41, 0.71]\)  
     \(e^{0.71} \approx 2.03,\; e^{1.41} \approx 4.10\). The sum is \(8.16\). So the row becomes \([0.25, 0.50, 0.25]\).  

   The softmax matrix \(\alpha\) is:
   \[
   \alpha = \begin{bmatrix}
   0.33 & 0.33 & 0.33 \\
   0.25 & 0.50 & 0.25 \\
   0.25 & 0.50 & 0.25
   \end{bmatrix}
   \]
   (slight rounding applied).

4. **Multiply by \(V\)**  
   \[
   \text{Attention}(Q, K, V) = \alpha V.
   \]
   - Row 1 weights \([0.33, 0.33, 0.33]\) on \(V\):
     \[
     0.33 \times [0,2] + 0.33 \times [1,1] + 0.33 \times [2,0]
     = [0 + 0.33 + 0.66,\; 0.66 + 0.33 + 0]
     \approx [0.99,\; 0.99].
     \]
   - Row 2 weights \([0.25, 0.50, 0.25]\):
     \[
     0.25 \times [0,2] + 0.50 \times [1,1] + 0.25 \times [2,0]
     = [0,\;0.5] + [0.5,\;0.5] + [0.5,\;0]
     = [1.0,\;1.0].
     \]
   - Row 3 weights \([0.25, 0.50, 0.25]\):
     \[
     0.25 \times [0,2] + 0.50 \times [1,1] + 0.25 \times [2,0]
     = [0,\;0.5] + [0.5,\;0.5] + [0.5,\;0]
     = [1.0,\;1.0].
     \]

   **Final Output**:
   \[
   \begin{bmatrix}
   0.99 & 0.99 \\
   1.00 & 1.00 \\
   1.00 & 1.00
   \end{bmatrix}
   \]
   (rounded values).

---

#### Example 2: Another Small-Dimension Example
Let us consider an even smaller example:

\[
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
\]

Here, \(Q\) is \(1 \times 2\), \(K\) is \(2 \times 2\), and \(V\) is \(2 \times 2\).

1. **Compute \(QK^T\)**  
   Since \(K\) is a square matrix, \(K^T = K\):
   \[
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
   \]

2. **Scale by \(\sqrt{d_k}\)**  
   \(d_k = 2\). Thus, \(\frac{1}{\sqrt{2}} \approx \frac{1}{1.41} \approx 0.71\). So
   \[
   \frac{[1,\;1]}{1.41} \approx [0.71,\;0.71].
   \]

3. **Softmax**  
   \([0.71, 0.71]\) has equal values, so the softmax is \([0.5, 0.5]\).

4. **Multiply by \(V\)**  
   \[
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
   \]

**Final Output**: \([3,\;2]\).

---

### Multi-Head Attention
Multi-head attention projects \(Q, K, V\) into multiple subspaces and performs several parallel scaled dot-product attentions (referred to as "heads"). These are concatenated, then transformed via a final linear projection:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O,
\]
where each head is computed as:
\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V).
\]

Below are multiple examples illustrating how multi-head attention calculations are performed, with increasingly detailed numeric demonstrations.

#### Example 1: Two-Head Attention Computation (Conceptual Illustration)
Let us assume we have a 2-head setup (\(h = 2\)), each head operating on half the dimension of \(Q, K, V\). For instance, if the original dimension is 4, each head dimension could be 2.

- **Step 1**: Linear transformations and splitting  
  \[
  Q W^Q \rightarrow [Q_1,\ Q_2], \quad
  K W^K \rightarrow [K_1,\ K_2], \quad
  V W^V \rightarrow [V_1,\ V_2].
  \]
  Here, \([Q_1,\ Q_2]\) means we split the transformed \(Q\) along its last dimension into two sub-matrices (head 1 and head 2).

- **Step 2**: Compute scaled dot-product attention for each head  
  \[
  \text{head}_1 = \text{Attention}(Q_1, K_1, V_1), \quad
  \text{head}_2 = \text{Attention}(Q_2, K_2, V_2).
  \]

  Suppose after computation:
  \[
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
  \]

- **Step 3**: Concatenate and apply final linear transform  
  Concatenating the heads yields a \(3 \times 4\) matrix (if each head is \(3 \times 2\)):
  \[
  \text{Concat}(\text{head}_1, \text{head}_2) =
  \begin{bmatrix}
  h_{11} & h_{12} & g_{11} & g_{12} \\
  h_{21} & h_{22} & g_{21} & g_{22} \\
  h_{31} & h_{32} & g_{31} & g_{32}
  \end{bmatrix}.
  \]
  We then multiply by \(W^O\) (e.g., a \(4 \times 4\) matrix) to get the final multi-head attention output.

> *Note*: Actual numeric computation requires specifying all projection matrices \(W_i^Q, W_i^K, W_i^V, W^O\) and the input \(Q, K, V\). Below, we provide more concrete numeric examples.

---

#### Example 2: Two-Head Attention with Full Numerical Details
In this example, we will provide explicit numbers for a 2-head setup. We will assume each of \(Q, K, V\) has shape \((3,4)\): there are 3 “tokens” (or time steps), each with a hidden size of 4. We split that hidden size into 2 heads, each with size 2.

**Step 0: Define inputs and parameters**  
Let
\[
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
\]
We also define the projection matrices for the two heads. For simplicity, we assume each projection matrix has shape \((4,2)\) (since we project dimension 4 down to dimension 2), and \(W^O\) will have shape \((4,4)\) to map the concatenated result \((3,4)\) back to \((3,4)\).

Let’s define:
\[
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
\]
\[
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
\]

And let:
\[
W^O = \begin{bmatrix}
1 & 0 & 0 & 1\\
0 & 1 & 1 & 0\\
1 & 0 & 1 & 0\\
0 & 1 & 0 & 1
\end{bmatrix}.
\]

We will go step by step.

---

**Step 1: Compute \(Q_1, K_1, V_1\) for Head 1**  

\[
Q_1 = Q \times W^Q_1,\quad
K_1 = K \times W^K_1,\quad
V_1 = V \times W^V_1.
\]

- \(Q_1 = Q W^Q_1\).  
  Each row of \(Q\) is multiplied by \(W^Q_1\):
  \[
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
  \]
  - Row 1 of \(Q\): \([1,2,1,0]\)  
    \[
    [1,2,1,0]
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1*1 + 2*0 + 1*1 + 0*0,\;\; 1*0 + 2*1 + 1*0 + 0*1]
    =
    [1+0+1,\;\;0+2+0] = [2,\;2].
    \]
  - Row 2: \([0,1,1,1]\)
    \[
    [0,1,1,1]
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [0+0+1,\;\;0+1+1] = [1,\;2].
    \]
  - Row 3: \([1,0,2,1]\)
    \[
    [1,0,2,1]
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1+0+2,\;\;0+0+1] = [3,\;1].
    \]

  Thus,
  \[
  Q_1 = \begin{bmatrix}
  2 & 2\\
  1 & 2\\
  3 & 1
  \end{bmatrix}.
  \]

- \(K_1 = K W^K_1\).  
  \[
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
  \]
  - Row 1: \([1,1,0,2]\)
    \[
    [1,1,0,2]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [1*1 +1*0+0*0+2*1,\;\;1*0+1*1+0*1+2*0]
    =
    [1 + 2,\;\;1].
    =
    [3,\;1].
    \]
  - Row 2: \([2,1,1,0]\)
    \[
    [2,1,1,0]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [2*1 +1*0+1*0+0*1,\;\;2*0+1*1+1*1+0*0]
    =
    [2,\;2].
    \]
  - Row 3: \([0,1,1,1]\)
    \[
    [0,1,1,1]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [0+0+0+1,\;\;0+1+1+0]
    =
    [1,\;2].
    \]

  So,
  \[
  K_1 = \begin{bmatrix}
  3 & 1\\
  2 & 2\\
  1 & 2
  \end{bmatrix}.
  \]

- \(V_1 = V W^V_1\).  
  \[
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
  \]
  - Row 1: \([1,1,0,0]\)
    \[
    [1,1,0,0] \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1+0,\;0+1] = [1,\;1].
    \]
  - Row 2: \([0,2,1,1]\)
    \[
    [0,2,1,1]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [0+0+1,\;0+2+1]
    =
    [1,\;3].
    \]
  - Row 3: \([1,1,2,2]\)
    \[
    [1,1,2,2]
    \times
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    1 & 0\\
    0 & 1
    \end{bmatrix}
    =
    [1+0+2,\;0+1+2] = [3,\;3].
    \]

  Therefore,
  \[
  V_1 = \begin{bmatrix}
  1 & 1\\
  1 & 3\\
  3 & 3
  \end{bmatrix}.
  \]

---

**Step 2: Compute \(Q_2, K_2, V_2\) for Head 2**  
Similarly,

\[
Q_2 = Q \times W^Q_2,\quad
K_2 = K \times W^K_2,\quad
V_2 = V \times W^V_2.
\]

- \(Q_2 = Q W^Q_2\):  
  \[
  W^Q_2 = \begin{bmatrix}
  0 & 1\\
  1 & 0\\
  1 & 1\\
  0 & 0
  \end{bmatrix}.
  \]
  - Row 1 \([1,2,1,0]\):
    \[
    [1,2,1,0]
    \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 1\\
    0 & 0
    \end{bmatrix}
    =
    [1*0+2*1+1*1+0*0,\; 1*1+2*0+1*1+0*0]
    =
    [2+1,\;1+1] = [3,\;2].
    \]
  - Row 2 \([0,1,1,1]\):
    \[
    [0,1,1,1]
    \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 1\\
    0 & 0
    \end{bmatrix}
    =
    [0+1+1,\;0+0+1] = [2,\;1].
    \]
  - Row 3 \([1,0,2,1]\):
    \[
    [1,0,2,1]
    \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 1\\
    0 & 0
    \end{bmatrix}
    =
    [1*0+0*1+2*1+1*0,\;1*1+0*0+2*1+1*0]
    =
    [2,\;1+2] = [2,\;3].
    \]

  Hence,
  \[
  Q_2 = \begin{bmatrix}
  3 & 2\\
  2 & 1\\
  2 & 3
  \end{bmatrix}.
  \]

- \(K_2 = K W^K_2\):  
  \[
  W^K_2 = \begin{bmatrix}
  0 & 1\\
  1 & 0\\
  1 & 0\\
  1 & 1
  \end{bmatrix}.
  \]
  - Row 1 \([1,1,0,2]\):
    \[
    [1,1,0,2] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 0\\
    1 & 1
    \end{bmatrix}
    =
    [1*0+1*1+0*1+2*1,\; 1*1+1*0+0*0+2*1]
    =
    [1+2,\;1+2] = [3,\;3].
    \]
  - Row 2 \([2,1,1,0]\):
    \[
    [2,1,1,0] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 0\\
    1 & 1
    \end{bmatrix}
    =
    [2*0+1*1+1*1+0*1,\;2*1+1*0+1*0+0*1]
    =
    [1+1,\;2] = [2,\;2].
    \]
  - Row 3 \([0,1,1,1]\):
    \[
    [0,1,1,1] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    1 & 0\\
    1 & 1
    \end{bmatrix}
    =
    [0+1+1+1,\;0+0+0+1] = [3,\;1].
    \]

  So,
  \[
  K_2 = \begin{bmatrix}
  3 & 3\\
  2 & 2\\
  3 & 1
  \end{bmatrix}.
  \]

- \(V_2 = V W^V_2\):  
  \[
  W^V_2 = \begin{bmatrix}
  0 & 1\\
  1 & 1\\
  0 & 1\\
  1 & 0
  \end{bmatrix}.
  \]
  - Row 1 \([1,1,0,0]\):
    \[
    [1,1,0,0] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [1*0+1*1+0*0+0*1,\;1*1+1*1+0*1+0*0]
    =
    [1,\;2].
    \]
  - Row 2 \([0,2,1,1]\):
    \[
    [0,2,1,1] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [0+2*1+1*0+1*1,\;0+2*1+1*1+1*0]
    =
    [2+1,\;2+1]
    =
    [3,\;3].
    \]
  - Row 3 \([1,1,2,2]\):
    \[
    [1,1,2,2] \times
    \begin{bmatrix}
    0 & 1\\
    1 & 1\\
    0 & 1\\
    1 & 0
    \end{bmatrix}
    =
    [1*0+1*1+2*0+2*1,\;1*1+1*1+2*1+2*0]
    =
    [1+2,\;1+1+2]
    =
    [3,\;4].
    \]

  Thus,
  \[
  V_2 = \begin{bmatrix}
  1 & 2\\
  3 & 3\\
  3 & 4
  \end{bmatrix}.
  \]

---

**Step 3: Compute each head’s Scaled Dot-Product Attention**  

We now have for head 1:
\[
Q_1 = \begin{bmatrix}2 & 2\\1 & 2\\3 & 1\end{bmatrix},\;
K_1 = \begin{bmatrix}3 & 1\\2 & 2\\1 & 2\end{bmatrix},\;
V_1 = \begin{bmatrix}1 & 1\\1 & 3\\3 & 3\end{bmatrix}.
\]
Similarly for head 2:
\[
Q_2 = \begin{bmatrix}3 & 2\\2 & 1\\2 & 3\end{bmatrix},\;
K_2 = \begin{bmatrix}3 & 3\\2 & 2\\3 & 1\end{bmatrix},\;
V_2 = \begin{bmatrix}1 & 2\\3 & 3\\3 & 4\end{bmatrix}.
\]

Assume each key vector dimension is \(d_k = 2\). Hence the scale is \(\frac{1}{\sqrt{2}} \approx 0.707\).

- **Head 1**:  
  1. \(Q_1 K_1^T\). For brevity, we illustrate one or two row computations explicitly, then show final results.

     \(K_1^T\) is
     \(\begin{bmatrix}
     3 & 2 & 1\\
     1 & 2 & 2
     \end{bmatrix}\).

     \[
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
     (2*3 + 2*1) & (2*2 + 2*2) & (2*1 + 2*2) \\
     (1*3 + 2*1) & (1*2 + 2*2) & (1*1 + 2*2) \\
     (3*3 + 1*1) & (3*2 + 1*2) & (3*1 + 1*2)
     \end{bmatrix}.
     \]
     Concretely:
     - Row 1 = \([6+2,\;4+4,\;2+4] = [8,\;8,\;6]\)
     - Row 2 = \([3+2,\;2+4,\;1+4] = [5,\;6,\;5]\)
     - Row 3 = \([9+1,\;6+2,\;3+2] = [10,\;8,\;5]\)

     So,
     \[
     Q_1 K_1^T = \begin{bmatrix}
     8 & 8 & 6\\
     5 & 6 & 5\\
     10 & 8 & 5
     \end{bmatrix}.
     \]
  2. Scale: \(\frac{Q_1 K_1^T}{\sqrt{2}}\). Numerically multiply each element by \(\approx 0.707\).  
     \[
     \approx
     \begin{bmatrix}
     5.66 & 5.66 & 4.24\\
     3.54 & 4.24 & 3.54\\
     7.07 & 5.66 & 3.54
     \end{bmatrix}.
     \]

  3. Apply softmax row-wise. We omit the step-by-step exponent calculation for brevity, but the general procedure is:
     - For row 1: \(\text{softmax}([5.66, 5.66, 4.24])\)
     - For row 2: \(\text{softmax}([3.54, 4.24, 3.54])\)
     - For row 3: \(\text{softmax}([7.07, 5.66, 3.54])\)

     Let’s provide approximate results (after exponentiation and normalization):
     \[
     \alpha_1 \approx
     \begin{bmatrix}
     0.43 & 0.43 & 0.14\\
     0.30 & 0.41 & 0.30\\
     0.67 & 0.24 & 0.09
     \end{bmatrix}.
     \]
     (Sum of each row \(\approx 1\).)

  4. Multiply by \(V_1\):
     \[
     \text{head}_1 = \alpha_1 \times V_1.
     \]
     Where
     \[
     V_1 = \begin{bmatrix}
     1 & 1\\
     1 & 3\\
     3 & 3
     \end{bmatrix}.
     \]
     - Row 1 of \(\alpha_1\): \([0.43, 0.43, 0.14]\)
       \[
       = 0.43 \times [1,1] + 0.43 \times [1,3] + 0.14 \times [3,3].
       \]
       Approximating:
       \[
       = [0.43+0.43+0.42,\; 0.43+1.29+0.42]
       = [1.28,\;2.14].
       \]
     - Similarly for rows 2 and 3 (details omitted for brevity). Suppose final approximate:
       \[
       \text{head}_1 \approx
       \begin{bmatrix}
       1.28 & 2.14\\
       1.05 & 2.25\\
       1.78 & 2.70
       \end{bmatrix}.
       \]

- **Head 2**:  
  We repeat the same steps with \(Q_2, K_2, V_2\).  
  1. \(Q_2 K_2^T\).  
     \[
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
     \]
     Then
     \[
     K_2^T = \begin{bmatrix}
     3 & 2 & 3\\
     3 & 2 & 1
     \end{bmatrix}.
     \]
     Multiply to get \((3 \times 2) \times (2 \times 3) = 3 \times 3\):
     \[
     Q_2 K_2^T = \begin{bmatrix}
     (3*3 +2*3) & (3*2 +2*2) & (3*3+2*1)\\
     (2*3 +1*3) & (2*2 +1*2) & (2*3 +1*1)\\
     (2*3 +3*3) & (2*2 +3*2) & (2*3 +3*1)
     \end{bmatrix}
     =
     \begin{bmatrix}
     9+6 & 6+4 & 9+2\\
     6+3 & 4+2 & 6+1\\
     6+9 & 4+6 & 6+3
     \end{bmatrix}
     =
     \begin{bmatrix}
     15 & 10 & 11\\
     9  & 6  & 7\\
     15 & 10 & 9
     \end{bmatrix}.
     \]
  2. Scale: multiply by \(1/\sqrt{2} \approx 0.707\):
     \[
     \approx
     \begin{bmatrix}
     10.61 & 7.07 & 7.78\\
     6.36 & 4.24 & 4.95\\
     10.61 & 7.07 & 6.36
     \end{bmatrix}.
     \]
  3. Softmax row-wise (again, approximate):
     \[
     \alpha_2 \approx
     \begin{bmatrix}
     0.46 & 0.24 & 0.30\\
     0.43 & 0.26 & 0.31\\
     0.45 & 0.30 & 0.25
     \end{bmatrix}.
     \]
  4. Multiply by \(V_2\):
     \[
     V_2 = \begin{bmatrix}
     1 & 2\\
     3 & 3\\
     3 & 4
     \end{bmatrix}.
     \]
     For row 1 of \(\alpha_2\): \([0.46, 0.24, 0.30]\):
     \[
     = 0.46\times[1,2] + 0.24\times[3,3] + 0.30\times[3,4].
     \]
     \[
     \approx [0.46+0.72+0.90,\;0.92+0.72+1.20]
     = [2.08,\;2.84].
     \]
     Similarly compute for rows 2 and 3. Suppose we get:
     \[
     \text{head}_2 \approx
     \begin{bmatrix}
     2.08 & 2.84\\
     1.73 & 2.30\\
     2.40 & 3.10
     \end{bmatrix}.
     \]

---

**Step 4: Concatenate and apply \(W^O\)**  
We now concatenate \(\text{head}_1\) and \(\text{head}_2\) horizontally to form a \((3 \times 4)\) matrix:

\[
\text{Concat}(\text{head}_1, \text{head}_2) =
\begin{bmatrix}
1.28 & 2.14 & 2.08 & 2.84 \\
1.05 & 2.25 & 1.73 & 2.30 \\
1.78 & 2.70 & 2.40 & 3.10
\end{bmatrix}.
\]

Finally, multiply by \(W^O\) \((4 \times 4)\):

\[
\text{Output} = (\text{Concat}(\text{head}_1, \text{head}_2)) \times W^O.
\]

Where
\[
W^O = \begin{bmatrix}
1 & 0 & 0 & 1\\
0 & 1 & 1 & 0\\
1 & 0 & 1 & 0\\
0 & 1 & 0 & 1
\end{bmatrix}.
\]
We can do a row-by-row multiplication:

- First row: \([1.28,\, 2.14,\, 2.08,\, 2.84]\).
  \[
  [1.28, 2.14, 2.08, 2.84]
  \times
  W^O
  \]
  yields a 4D vector. Due to length, we won’t detail every step, but the final result will be something like:
  \[
  \approx [1.28 + 2.08,\; 2.14 + 2.08,\; 2.14 + 2.08,\; 1.28 + 2.84].
  \]
  (Following the pattern of how we multiply by each column of \(W^O\).)

Continuing likewise for rows 2 and 3 will yield the final multi-head attention output. The exact numeric results will vary depending on the rounding, but the method is straightforward.

---

#### Example 3: Three-Head Attention with Another Set of Numbers (Short Demonstration)
For completeness, suppose we wanted \(h=3\) heads, each of dimension \(\frac{d_{\text{model}}}{3}\). The steps are exactly the same:

1. Project \(Q, K, V\) into three subspaces via \(W^Q_i, W^K_i, W^V_i\).
2. Perform scaled dot-product attention for each head:  
   \(\text{head}_i = \text{Attention}(Q_i, K_i, V_i)\).
3. Concatenate all heads: \(\text{Concat}(\text{head}_1, \text{head}_2, \text{head}_3)\).
4. Multiply by \(W^O\).

Each numeric calculation is analogous to the 2-head case—just with different shapes (e.g., each head might have dimension 4/3 if the original dimension is 4, which typically would be handled with rounding or a slightly different total dimension). The procedure remains identical in principle.

---

## Position-Wise Feed-Forward Networks
Each layer in a Transformer includes a position-wise feed-forward network (FFN) that applies a linear transformation and activation to each position independently:

\[
\text{FFN}(x) = \max(0,\; xW_1 + b_1)\, W_2 + b_2,
\]
where \(\max(0, \cdot)\) is the ReLU activation function.

### Example: Numerical Computation of the Feed-Forward Network
Let
\[
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
\]

1. **Compute \(xW_1 + b_1\)**  
   - Row 1: \([1, 0]\)
     \[
     [1, 0]
     \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
     = [1, 1],
     \]
     then add \([0, 1]\) to get \([1, 2]\).
   - Row 2: \([0, 1]\)
     \[
     [0,1]\times
     \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}
     = [0, 1],
     \]
     plus \([0, 1]\) = \([0, 2]\).
   - Row 3: \([1,1]\)
     \[
     [1,1]\times
     \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}
     = [1, 2],
     \]
     plus \([0, 1]\) = \([1, 3]\).

   So
   \[
   X_1 =
   \begin{bmatrix}
   1 & 2\\
   0 & 2\\
   1 & 3
   \end{bmatrix}.
   \]

2. **ReLU activation**  
   \(\max(0, X_1)\) leaves nonnegative elements unchanged. All entries are already \(\ge0\), so
   \[
   \text{ReLU}(X_1) = X_1.
   \]

3. **Multiply by \(W_2\) and add \(b_2\)**  
   \[
   W_2 =
   \begin{bmatrix}
   1 & 0\\
   2 & 1
   \end{bmatrix},\quad
   b_2 = [1, -1].
   \]
   \[
   X_2 = X_1 W_2.
   \]
   - Row 1 of \(X_1\): \([1,2]\)
     \[
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
     \]
     So \([5,2]\).
   - Row 2: \([0,2]\)
     \[
     [0,2]
     \begin{bmatrix}1\\2\end{bmatrix}=4,\quad
     [0,2]
     \begin{bmatrix}0\\1\end{bmatrix}=2.
     \]
   - Row 3: \([1,3]\)
     \[
     [1,3]\begin{bmatrix}1\\2\end{bmatrix}=1+6=7,\quad
     [1,3]\begin{bmatrix}0\\1\end{bmatrix}=0+3=3.
     \]

   Thus
   \[
   X_2 = \begin{bmatrix}
   5 & 2\\
   4 & 2\\
   7 & 3
   \end{bmatrix}.
   \]
   Add \(b_2=[1,-1]\):
   \[
   X_2 + b_2 =
   \begin{bmatrix}
   6 & 1\\
   5 & 1\\
   8 & 2
   \end{bmatrix}.
   \]

**Final Output**:
\[
\begin{bmatrix}
6 & 1\\
5 & 1\\
8 & 2
\end{bmatrix}.
\]

---

## Training and Optimization

### Optimizer and Learning Rate Scheduling
Transformers commonly use Adam or AdamW, combined with a piecewise learning rate scheduling strategy:

\[
l_{\text{rate}} = d_{\text{model}}^{-0.5}
\cdot
\min\bigl(\text{step}_\text{num}^{-0.5},\;
\text{step}_\text{num}\times \text{warmup}_\text{steps}^{-1.5}\bigr),
\]
where:
- \(d_{\text{model}}\) is the hidden dimension.
- \(\text{step}_\text{num}\) is the current training step.
- \(\text{warmup}_\text{steps}\) is the number of warmup steps.

#### Example: Learning Rate Calculation
Let:
- \(d_{\text{model}} = 4\),
- \(\text{step}_\text{num} = 1000\),
- \(\text{warmup}_\text{steps} = 2000\).

1. **Compute \(d_{\text{model}}^{-0.5}\)**
   \[
   4^{-0.5} = \frac{1}{\sqrt{4}} = 0.5.
   \]

2. **Compute \(\min(\text{step}_\text{num}^{-0.5},\; \text{step}_\text{num} \times \text{warmup}_\text{steps}^{-1.5})\)**
   - \(\text{step}_\text{num}^{-0.5} = 1000^{-0.5} = \frac{1}{\sqrt{1000}} \approx 0.0316.\)
   - \(\text{step}_\text{num} \times \text{warmup}_\text{steps}^{-1.5} = 1000 \times 2000^{-1.5}.\)  
     \(2000^{-1.5} = \frac{1}{2000^{1.5}} \approx \frac{1}{2000\times44.72} \approx 1.118\times10^{-5}.\)  
     Hence \(\approx 1000 \times 1.118\times10^{-5} = 0.01118.\)

   The minimum is \(\min(0.0316, 0.01118) = 0.01118.\)

3. **Combine for final LR**  
   \[
   l_{\text{rate}} = 0.5 \times 0.01118 = 0.00559.
   \]
   So the learning rate \(\approx 0.00559.\)

---

## Conclusion
The Transformer architecture has become a foundational model in modern deep learning, showing remarkable performance in NLP, computer vision, and multimodal applications. Its ability to capture long-range dependencies, combined with high parallelizability and scalability, has inspired a diverse range of research directions and practical systems. Ongoing work continues to explore ways to improve Transformer efficiency, adapt it to new scenarios, and enhance model interpretability.