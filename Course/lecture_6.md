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

# Lecture 6: Efficient Text Generation of Decoder-Only Transformers: KV-Cache

## 1. KV Cache

For faster generative inference, transformers often use a **KV cache**, which stores keys and values from **previous** tokens so that each new token only attends to the previously computed K and V rather than recomputing them from scratch.

### Inference Without KV Cache

During the generation process without KV Cache, each new token is produced as follows:

#### Initial Token:

Start with the initial token (e.g., a start-of-sequence token). Compute its Q, K, and V vectors, and apply the attention mechanism to generate the first output token.

#### Subsequent Tokens:

For each new token, recompute the Q, K, and V vectors for the entire sequence (including all previous tokens), and apply the attention mechanism to generate the next token. This approach leads to redundant computations, as the K and V vectors for previously processed tokens are recalculated at each step, resulting in increased computational load and latency.

### Inference With KV Cache

The KV Cache technique addresses the inefficiencies of the above method by storing the Key and Value vectors of previously processed tokens:

#### Initial Token:

Compute the Q, K, and V vectors for the initial token and store the K and V vectors in the cache.

#### Subsequent Tokens:

For each new token, compute its Q, K, and V vectors. Instead of recomputing K and V for all previous tokens, retrieve them from the cache. Apply the attention mechanism using the current Q vector and the cached K and V vectors to generate the next token. By caching the K and V vectors, the model avoids redundant computations, leading to faster inference times.

With KV Cache, A typical inference process has two phases:

1. **Prefill Phase**: The full prompt sequence ($s$ tokens) is fed into the model, generating the key and value cache for each layer.  
2. **Decoding Phase**: Tokens are generated one by one (or in small batches), each time updating and using the cached keys and values.

### 1.1 Memory Usage of the KV Cache

Suppose the input sequence length is $s$, and we want to generate $n$ tokens. Let $b$ be the inference batch size (number of parallel sequences). We store $K, V \in \mathbb{R}^{b \times (s+n) \times h}$ in float16. Each element is 2 bytes, and we have both $K$ and $V$, so the memory cost per layer is:

$$
2 \;\text{(for K and V)} \;\times\; b(s+n)h \;\times\; 2\,\text{bytes} = 4\,b\,(s+n)\,h.
$$

For $l$ layers, the total KV cache memory is:
$$
4\,l\,b\,h\,(s+n).
$$

#### GPT-3 Example

Recall GPT-3 has around 350 GB of parameters (in float16). Suppose we do inference with batch size $b=64$, prompt length $s=512$, and we generate $n=32$ tokens:

- **Model parameters**: 350 GB  
- **KV cache**:  
  $$
  4\,l\,b\,h\,(s+n)\approx 164\,\text{GB}
  $$
  which is nearly half the parameter size under these settings.

---

## 2. Conclusion

In this class, we explored how to estimate and analyze key aspects of inference for large language models:

   - A powerful mechanism for fast autoregressive decoding, storing keys and values for each token in float16 to avoid recomputing them.  
   - The total KV cache scales with $b(s+n)h$ per layer.

By dissecting these components, we gain a clearer picture of **why** training large language models requires extensive memory and computation, and **how** various strategies (e.g., activation recomputation, KV cache) are applied to optimize hardware resources. Such understanding is crucial for practitioners to make informed decisions about scaling laws, distributed training setups, and memory-saving techniques.

---