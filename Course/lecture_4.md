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

Reference Tutorial: [Copy of parameter size of vanilla transformer](https://colab.research.google.com/drive/1jdhD6yexq2PBZYg9zg5FG-YRGyL8ZJ0L?usp=sharing)