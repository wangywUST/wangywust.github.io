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

In-Class Question 1: Given layer number $N$ as 6, model dimension $d_{model}$ as 512, feed-forward dimension $d_{ff}$ = 2048, number of attention heads $h$ = 8, what is the total number of learnable parameters in a vanilla Transformer model?

In-Class Question 2: Given layer number $N$ as 6, model dimension $d_{model}$ as 1024, feed-forward dimension $d_{ff}$ = 4096, number of attention heads $h$ = 16, what is the total number of learnable parameters in a vanilla Transformer model?

Reference Tutorial: [Parameter size of vanilla transformer](https://colab.research.google.com/drive/1jdhD6yexq2PBZYg9zg5FG-YRGyL8ZJ0L?usp=sharing)

# Lecture 4: Analysis of Transformer Models: Parameter Count, Computation, Activations, and KV Cache

## 1. Introduction

Recently, OpenAI's ChatGPT has demonstrated exceptional performance, sparking a surge in research on Large Language Models (LLMs). The "large" in large language models is reflected in two aspects: large model parameter scale and large training data scale. Taking GPT-3 as an example, GPT-3 has 175 billion parameters and was trained on 570GB of data. Consequently, training large language models faces two main challenges: memory efficiency and computational efficiency.

Current industry large language models are based on the transformer architecture, with model structures primarily falling into two categories: encoder-decoder (represented by T5) and decoder-only. Specifically, the decoder-only structure can be further divided into Causal LM (represented by the GPT series) and Prefix LM (represented by GLM). Due to the tremendous success of the GPT series, most mainstream large language models adopt the Causal LM structure. Therefore, for the decoder-only framework, to better understand the memory efficiency and computational efficiency of training large language models, this article analyzes the model parameter count, computational requirements, intermediate activations, and KV cache of transformer models using the decoder-only framework.

To facilitate analysis, let's first define some mathematical notation. Let the number of transformer layers be $l$, the hidden dimension be $h$, and the number of attention heads be $a$. The vocabulary size is $V$, the training batch size is $b$, and the sequence length is $s$.

## 2. Model Parameter Count

A transformer model consists of $l$ identical layers, with each layer divided into two parts: a self-attention block and an MLP block.

The self-attention block's model parameters include weight matrices $W_Q$, $W_K$, $W_V$ and their biases, as well as the output weight matrix $W_O$ and its bias. All four weight matrices have the shape $[h, h]$, and the four biases have the shape $[h]$. The parameter count of the self-attention block is $4h^2 + 4h$.

The MLP block consists of two linear layers. Generally, the first linear layer maps the dimension from $h$ to $4h$, and the second linear layer maps it back from $4h$ to $h$. The weight matrix $W_1$ of the first linear layer has the shape $[h, 4h]$, and its bias has the shape $[4h]$. The weight matrix $W_2$ of the second linear layer has the shape $[4h, h]$, and its bias has the shape $[h]$. The parameter count of the MLP block is $8h^2 + 5h$.

Both the self-attention block and the MLP block each have a layer normalization, which contains two trainable parameters: a scaling parameter $\gamma$ and a shifting parameter $\beta$, both with the shape $[h]$. The parameter count for the two layer normalizations is $4h$.

In total, each transformer layer has a parameter count of $12h^2 + 13h$.

Besides this, the word embedding matrix also has a significant number of parameters. The word vector dimension is usually equal to the hidden dimension $h$, so the parameter count of the word embedding matrix is $Vh$. The weight matrix of the final output layer is usually parameter-shared with the word embedding matrix.

Regarding position encoding, if trainable position encoding is used, there will be some trainable model parameters, though relatively few. If relative position encoding is used, such as RoPE and ALiBi, it does not include trainable model parameters. We ignore this part of the parameters.

In summary, an $l$-layer transformer model has a trainable parameter count of $l(12h^2 + 13h) + Vh$. When the hidden dimension $h$ is large, we can ignore the linear terms, and the model parameter count is approximately $12lh^2$.

Next, let's estimate the parameter counts of different versions of the LLaMA model:

| Actual Parameter Count | Hidden Dimension h | Layer Count l | 12lh² |
|------------------------|--------------------|--------------|--------------------|
| 6.7B                   | 4096               | 32           | 6,442,450,944      |
| 13.0B                  | 5120               | 40           | 12,582,912,000     |
| 32.5B                  | 6656               | 60           | 31,897,681,920     |
| 65.2B                  | 8192               | 80           | 64,424,509,440     |

### 2.1 Memory Usage Analysis During Training

During neural network training, the major memory consumers are divided into four parts: model parameters, intermediate activations generated during the forward pass, gradients computed during the backward pass, and optimizer states. Here we focus on analyzing the memory usage of parameters, gradients, and optimizer states, while the memory usage of intermediate activations will be discussed in detail later. When training large models, the AdamW optimizer is typically used, along with mixed precision training to accelerate training, and our analysis of memory usage is based on these premises.

In one training iteration, each trainable model parameter corresponds to one gradient and two optimizer states (first-order momentum and second-order momentum of the Adam optimizer gradient). Let the number of model parameters be $\Phi$, then the number of gradient elements is $\Phi$, and the number of AdamW optimizer elements is $2\Phi$. A float16 data type element occupies 2 bytes, while a float32 data type element occupies 4 bytes. In mixed precision training, float16 model parameters are used for forward and backward passes, computing float16 gradients; when the optimizer updates model parameters, float32 optimizer states, float32 gradients, and float32 model parameters are used to update the model parameters. Therefore, for each trainable model parameter, it occupies 
$$(2 + 4) + (2 + 4) + (4 + 4) = 20\text{ bytes}$$.
Using the AdamW optimizer and mixed precision training to train a large model with $\Phi$ parameters, the memory size occupied by model parameters, gradients, and optimizer states is $20\Phi\text{ bytes}$.

### 2.2 Memory Usage Analysis During Inference

During the inference phase of neural networks, there are no optimizer states or gradients, and there is no need to save intermediate activations. Without gradients, optimizer states, and intermediate activations, the memory usage during model inference is far less than during the training phase. During model inference, the main memory consumer is the model parameters. If float16 is used for inference, the approximate memory usage of model parameters during inference is approximately $2\Phi\text{ bytes}$. If KV cache is used to accelerate the inference process, the KV cache also consumes memory, which will be discussed in detail below. Additionally, input data also needs to be loaded onto the GPU, along with some intermediate results (which are quickly released after use during inference), but the memory occupied by these parts is very small and can be ignored.

## 3. Computational Requirements (FLOPs) Estimation

FLOPs, floating point operations, represent the number of floating-point computations and measure the amount of computation.

How do we calculate the FLOPs of matrix multiplication?

For $A \in \mathbb{R}^{n \times m}, B \in \mathbb{R}^{m \times l}$, computing $AB$ requires $n \times m \times l$ multiplication operations and $n \times l \times (m-1)$ addition operations, totaling approximately $2nml$ floating-point operations, requiring $2nml$ FLOPs. For $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$, computing $AB$ requires $2mnp$ floating-point operations.

In one training iteration, assuming the input data has the shape $[b, s]$, let's first analyze the computation in the self-attention block. The computation formulas are as follows:

$Q = xW_Q, K = xW_K, V = xW_V$

$x_{out} = \text{softmax}(\frac{QK^T}{\sqrt{h}}) \cdot V \cdot W_O + x$

1. Computing $Q, K, V$: The matrix multiplication's input and output shapes are $[b, s, h] \times [h, h] \rightarrow [b, s, h]$. The computational requirement is $3 \times 2bsh^2 = 6bsh^2$.

2. For $QK^T$ matrix multiplication, the input and output shapes are 
   $[b, \text{head\_num}, s, \text{per\_head\_hidden\_size}] \times [b, \text{head\_num}, \text{per\_head\_hidden\_size}, s] \rightarrow [b, \text{head\_num}, s, s]$. 
   The computational requirement is $2bs^2h$.

3. Computing the weighted $V$ on $score$, the matrix multiplication's input and output shapes are 
   $[b, \text{head\_num}, s, s] \times [b, \text{head\_num}, s, \text{per\_head\_hidden\_size}] \rightarrow [b, \text{head\_num}, s, \text{per\_head\_hidden\_size}]$. 
   The computational requirement is $2bs^2h$.

4. For the linear mapping after attention, the matrix multiplication's input and output shapes are $[b, s, h] \times [h, h] \rightarrow [b, s, h]$. The computational requirement is $2bsh^2$.

Next, let's analyze the computation in the MLP block. The computation formulas are as follows:

$x = f_{gelu}(x_{out}W_1)W_2 + x_{out}$

1. For the first linear layer, the matrix multiplication's input and output shapes are $[b, s, h] \times [h, 4h] \rightarrow [b, s, 4h]$. The computational requirement is $8bsh^2$.

2. For the second linear layer, the matrix multiplication's input and output shapes are $[b, s, 4h] \times [4h, h] \rightarrow [b, s, h]$. The computational requirement is $8bsh^2$.

Summing up the above computations, the total computational requirement for each transformer layer is approximately $24bsh^2 + 4bs^2h$.

Additionally, another major computational requirement is for logits computation, mapping the hidden vectors to the vocabulary size. The matrix multiplication's input and output shapes are $[b, s, h] \times [h, V] \rightarrow [b, s, V]$. The computational requirement is $2bshV$.

Therefore, for an $l$-layer transformer model, with input data of shape $[b, s]$, the computational requirement for one training iteration is $l \times (24bsh^2 + 4bs^2h) + 2bshV$.

### 3.1 Relationship Between Computational Requirements and Parameter Count

When the hidden dimension $h$ is large and much larger than the sequence length $s$, we can ignore the linear terms, and the computational requirement can be approximated as $24bsh^2 \times l$. We mentioned earlier that when the model parameter count is $12lh^2$ and the number of input tokens is $bs$, the equation $\frac{24bsh^2l}{12lh^2} = 2bs$ holds. We can approximately consider that: in one forward pass, for each token, for each model parameter, 2 floating-point operations need to be performed, i.e., one multiplication operation and one addition operation.

One training iteration includes both forward and backward passes, and the computational requirement of the backward pass is twice that of the forward pass. Therefore, the combined coefficient of forward pass + backward pass is $1 + 2 = 3$. In one training iteration, for each token, for each model parameter, $2 \times 3 = 6$ floating-point operations need to be performed.

Next, we can estimate the computational requirements for training GPT3-175B. For GPT3, each token, each parameter performs 6 floating-point operations, and multiplying by the parameter count and total tokens gives the total computational requirement. GPT3's model parameter count is $174,600M$, and the training data amount is $300B$ tokens.

$6 \times 174600 \times 10^6 \times 300 \times 10^9 = 3.1428 \times 10^{23} \text{ flops}$

<div style="text-align: center;">
  <img src="./Course/lecture_4_fig_1.png" width="50%">
  <p style="margin-top: 10px;">Large Language Model's Costs</p>
</div>

### 3.2 Training Time Estimation

The model parameter count and total training tokens determine the computational requirements for training a transformer model. Given the hardware GPU type, we can estimate the required training time. Given the computational requirements, the training time (i.e., the time it takes for GPUs to complete so many flops of computation) depends not only on the GPU type but also on GPU utilization. When calculating end-to-end training GPU utilization, we need to consider not only the computation time of forward and backward passes but also the time for CPU data loading, optimizer updates, multi-card communication, and logging. Generally speaking, GPU utilization is typically between 0.3 and 0.55.

As mentioned earlier, in one forward pass, for each token, for each model parameter, 2 floating-point operations are performed. Using activation recomputation technology to reduce intermediate activation memory (which will be discussed in detail below) requires an additional forward pass, so the coefficient of forward pass + backward pass + activation recomputation is $1 + 2 + 1 = 4$. In one training iteration using activation recomputation, for each token, for each model parameter, $2 \times 4 = 8$ floating-point operations need to be performed. Given the training tokens, hardware environment configuration, the computation time for training a transformer model is:

$\text{Training Time} \approx \frac{8 \times \text{tokens count} \times \text{model parameter count}}{\text{GPU count} \times \text{GPU peak performance in flops} \times \text{GPU utilization}}$

Taking GPT3-175B as an example, on 1024 A100 GPUs with 40GB memory, training 175B parameters of GPT3 on 300B tokens of data. The peak performance of a 40GB memory A100 is 312 TFLOPS, assuming a GPU utilization of 0.45, then the required training time is 34 days, which aligns with the training time reported in [7].

Taking LLaMA-65B as an example, on 2048 A100 GPUs with 80GB memory, training a 65B parameter model on 1.4TB tokens of data. The peak performance of an 80GB memory A100 is 624 TFLOPS, assuming a GPU utilization of 0.3, then the required training time is 21 days, which aligns with the actual training time reported in [4].

## 4. Intermediate Activation Analysis

Besides model parameters, gradients, and optimizer states, a major consumer of memory is the intermediate activation values computed during the forward pass, which need to be saved for use in the backward pass to compute gradients. Here, activations refer to all tensors that are computed during the forward pass and needed during the backward pass. These activations do not include model parameters and optimizer states but do include the mask matrices needed for dropout operations.

When analyzing the memory usage of intermediate activations, we only consider the major consumers of activation memory and ignore some small buffers. For example, for layer normalization, computing gradients requires the layer's input, the mean $\mu$ and variance $\sigma^2$ of the input. The input contains $bsh$ elements, while the input's mean and variance each contain $bs$ elements. Since $h$ is usually quite large (in the thousands), we have $bsh \gg bs$. Therefore, for layer normalization, the intermediate activation is approximated as $bsh$, not $bsh + 2bs$.

Large models during training typically use mixed precision training, and intermediate activation values are generally of float16 or bfloat16 data types. When analyzing the memory usage of intermediate activations, we assume that intermediate activation values are saved in float16 or bfloat16 data format, with each element occupying 2 bytes. The only exception is the mask matrix for dropout operations, where each element only occupies 1 byte.

In the following analysis, the unit is bytes, not element count.

Each transformer layer includes a self-attention block and an MLP block, each corresponding to a layer normalization connection.

Let's first analyze the intermediate activations of the self-attention block. The computation formulas for the self-attention block are as follows:

1. For $Q, K, V$, we need to save their common input $x$, which is the intermediate activation. The shape of input $x$ is $[b, s, h]$, with $bsh$ elements, occupying $2 \times bsh = 2bsh$ bytes of memory.

2. For the $QK^T$ matrix multiplication, we need to save intermediate activations $Q, K$, both tensors with shape $[b, s, h]$, occupying a total of $2 \times 2 \times bsh = 4bsh$ bytes of memory.

3. For the $\text{softmax}()$ function, we need to save its input $QK^T$, occupying $2bs^2a$ bytes of memory, where $a$ represents the number of attention heads.

   The shape of $Q$ is: $[b, \text{head\_num}, s, \text{per\_head\_hidden\_size}]$
   The shape of $K^T$ is: $[b, \text{head\_num}, \text{per\_head\_hidden\_size}, s]$
   The shape of $QK^T$ is: $[b, \text{head\_num}, s, s]$, with $bs^2a$ elements, occupying $2bs^2a$ bytes of memory.

4. After computing the $\text{softmax}()$ function, a dropout operation is performed. We need to save a mask matrix, with the same shape as $QK^T$, occupying $bs^2a$ bytes of memory.

5. Computing attention on $V$, i.e., $\text{score} \cdot V$, we need to save $\text{score}$, with size $2bs^2a$; and $V$, with size $2bsh$. Together they occupy $2bs^2a + 2bsh$ bytes of memory.

6. Computing the output mapping and a dropout operation. The input mapping needs to save its input, with size $2bsh$; dropout needs to save the mask matrix, with size $bsh$. Together they occupy $3bsh$ bytes of memory.

Therefore, summing up the intermediate activations, the self-attention block's intermediate activations occupy $11bsh + 5bs^2a$ bytes of memory.

Next, let's look at the intermediate activations of the MLP block. The computation formulas for the MLP block are as follows:

1. The first linear layer needs to save its input, occupying $2bsh$ bytes of memory.
2. The activation function needs to save its input, occupying $8bsh$ bytes of memory.
3. The second linear layer needs to save its input, occupying $8bsh$ bytes of memory.
4. Finally, there is a dropout operation, needing to save the mask matrix, occupying $bsh$ bytes of memory.

For the MLP block, the intermediate activations to be saved total $19bsh$ bytes.

Additionally, the self-attention block and MLP block each correspond to a layer normalization. Each layer norm needs to save its input, with size $2bsh$. The intermediate activations for 2 layer norms total $4bsh$ bytes.

In summary, each transformer layer's intermediate activations occupy $34bsh + 5bs^2a$ bytes of memory. For an $l$-layer transformer model, there's also the embedding layer and the final output layer. The embedding layer doesn't need intermediate activations. Overall, when the hidden dimension $h$ is large and the layer count $l$ is deep, this part of intermediate activation is very small and can be ignored. Therefore, for an $l$-layer transformer model, the intermediate activations occupy approximately $(34bsh + 5bs^2a) \times l$ bytes of memory.

### 4.1 Comparing Memory Size of Intermediate Activations and Model Parameters

In one training iteration, the memory occupied by model parameters (or gradients) is only related to the model parameter count and parameter data type, not to the size of the input data. The memory occupied by optimizer states is similar, related to the optimizer type and model parameter count, but unrelated to the size of the input data. However, intermediate activation values are positively correlated with the size of the input data (batch size $b$ and sequence length $s$). As the batch size $b$ and sequence length $s$ increase, the memory occupied by intermediate activations increases accordingly.

When we encounter Out Of Memory (OOM) issues when training neural networks, we usually try to reduce the batch size to avoid memory issues. This approach actually reduces the memory occupied by intermediate activations, not the memory of model parameters, gradients, and optimizers.

Taking GPT3-175B as an example, let's intuitively compare the memory size of model parameters and intermediate activations. GPT3's model configuration is as follows. We assume mixed precision training is used, with model parameters and intermediate activations both using float16 data type, each element occupying 2 bytes.

| Model Name | Parameter Count | Layer Count | Hidden Dimension | Attention Heads |
|------------|----------------|------------|-----------------|----------------|
| GPT3 | 175B | 96 | 12288 | 96 |

GPT3's model parameter count is 175B, occupying $2 \times 175 \times 10^9 \text{ bytes} = 350GB$ of memory. GPT3 requires 350GB of memory.

GPT3's sequence length $s$ is 2048. Comparing the intermediate activations for different batch sizes $b$:

When $b = 1$, intermediate activations occupy 
$(34bsh + 5bs^2a) \times l = 275,414,777,856 \text{ bytes} \approx 275GB$, 
about 0.79 times the model parameter memory.

When $b = 64$, intermediate activations occupy 
$(34bsh + 5bs^2a) \times l = 17,626,545,782,784 \text{ bytes} \approx 17.6TB$, 
about 50 times the model parameter memory.

When $b = 128$, intermediate activations occupy 
$(34bsh + 5bs^2a) \times l = 35,253,091,565,568 \text{ bytes} \approx 35.3TB$, 
about 101 times the model parameter memory.

We can see that as the batch size $b$ increases, the memory occupied by intermediate activations far exceeds the model parameter memory. Activation recomputation technology is typically used to reduce intermediate activations, theoretically reducing intermediate activation memory from $O(n)$ to $O(\sqrt{n})$, at the cost of increasing the time of an additional forward computation, essentially a "time-space trade-off."

## 5. KV Cache

During inference, a common strategy to accelerate transformer model inference is to use KV cache. A typical large model generative inference includes two phases:

1. Prefill phase: Input a prompt sequence, generate key cache and value cache (KV cache) for each transformer layer.
2. Decoding phase: Use and update the KV cache, generating tokens one by one, with the current generated token depending on previously generated tokens.

The weight matrices of the $i$-th transformer layer are $W_Q^i, W_K^i, W_V^i, W_O^i, W_1^i, W_2^i$. Among them, the 4 weight matrices of the self-attention block are $W_Q^i, W_K^i, W_V^i, W_O^i \in \mathbb{R}^{h \times h}$, and the 2 weight matrices of the MLP block are $W_1^i \in \mathbb{R}^{h \times 4h}, W_2^i \in \mathbb{R}^{4h \times h}$.

**Prefill Phase**

Assuming the input to the $i$-th transformer layer is $x^i$, the key, value, query, and output of the self-attention block are denoted as $x_K^i, x_V^i, x_Q^i, x_{out}^i$, where $x_K^i, x_V^i, x_Q^i, x_{out}^i \in \mathbb{R}^{b \times s \times h}$.

The computation process for key cache and value cache is:

$x_K^i = x^i \cdot W_K^i$

$x_V^i = x^i \cdot W_V^i$

The remaining computation process for the $i$-th transformer layer is:

$x_Q^i = x^i \cdot W_Q^i$

$x_{out}^i = \text{softmax}(\frac{x_Q^i x_K^{i^T}}{\sqrt{h}}) \cdot x_V^i \cdot W_O^i + x^i$

$x^{i+1} = f_{gelu}(x_{out}^i \cdot W_1^i) \cdot W_2^i + x_{out}^i$

**Decoding Phase**

Given the current generated token's vector representation $t^i \in \mathbb{R}^{b \times 1 \times h}$ in the $i$-th transformer layer. The inference computation is divided into two parts: updating the KV cache and computing the output of the $i$-th transformer layer.

The computation process for updating the key cache and value cache is:

$x_K^i \leftarrow \text{Concat}(x_K^i, t^i \cdot W_K^i)$

$x_V^i \leftarrow \text{Concat}(x_V^i, t^i \cdot W_V^i)$

The remaining computation process for the $i$-th transformer layer is:

$x_Q^i = t^i \cdot W_Q^i$

$x_{out}^i = \text{softmax}(\frac{x_Q^i x_K^{i^T}}{\sqrt{h}}) \cdot x_V^i \cdot W_O^i + t^i$

$t^{i+1} = f_{gelu}(x_{out}^i \cdot W_1^i) \cdot W_2^i + x_{out}^i$

### 5.1 Memory Usage Analysis of KV Cache

Assuming the length of the input sequence is $s$, the length of the output sequence is $n$, and using float16 to store the KV cache, then the peak memory usage of the KV cache is $l(s + n) \times h \times l \times 2 \times 2 = 4lh(s + n)$ bytes. Here, the first 2 represents K/V cache, and the second 2 represents float16 occupying 2 bytes.

Taking GPT3 as an example, let's compare the memory size of KV cache and model parameters. GPT3 model occupies 350GB of memory. Assuming batch size $b = 64$, input sequence length $s = 512$, output sequence length $n = 32$, then the KV cache occupies $4lh(s + n) = 164,282,490,072 \text{ bytes} \approx 164GB$ of memory, about 0.5 times the model parameter memory.

## 6. Conclusion

This article first introduced how to calculate the parameter count of transformer models. Based on the parameter count, we can further estimate the memory usage of model parameters, gradients, and optimizer states. Next, the article estimated the computational requirements for training iterations, given the training tokens, and further estimated the computation time of training iterations based on computational requirements and card performance. Then, the article analyzed the memory size of intermediate activation values generated during the forward computation process of transformer models. The memory size of intermediate activations is positively correlated with input data size and can far exceed the memory occupied by model parameters. Finally, the article introduced the common acceleration strategy in the transformer model inference process: using KV cache. In summary, analyzing the parameter count, computational requirements, intermediate activations, and KV cache of transformer models helps understand the memory efficiency and computational efficiency during large model training and inference.

## 7. References

1. Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text transformer[J]. The Journal of Machine Learning Research, 2020, 21(1): 5485-5551.
2. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.
3. Brown T, Mann B, Ryder N, et al. Language models are few-shot learners[J]. Advances in neural information processing systems, 2020, 33: 1877-1901.
4. Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models[J]. arXiv preprint arXiv:2302.13971, 2023.
5. Sheng Y, Zheng L, Yuan B, et al. High-throughput generative inference of large language models with a single gpu[J]. arXiv preprint arXiv:2303.06865, 2023.
6. Korthikanti V, Casper J, Lym S, et al. Reducing activation recomputation in large transformer models[J]. arXiv preprint arXiv:2205.05198, 2022.
7. Narayanan D, Shoeybi M, Casper J, et al. Efficient large-scale language model training on gpu clusters using megatron-lm[C]//Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2021: 1-15.
8. Smith S, Patwary M, Norick B, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model[J]. arXiv preprint arXiv:2201.11990, 2022.