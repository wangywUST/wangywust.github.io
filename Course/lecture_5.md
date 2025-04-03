# Lecture 5: Decoder-only Transformer (LLM) vs Vanilla Transformer: A Detailed Comparison

## Introduction

Modern Large Language Models (LLMs) are primarily based on decoder-only transformer architectures, while the original transformer model ("vanilla transformer") uses an encoder-decoder structure. This class will explore the differences between these two architectures in detail, including their respective advantages, disadvantages, and application scenarios.

## Vanilla Transformer Architecture

In 2017, Vaswani et al. introduced the original transformer architecture in their paper "Attention is All You Need."

### Key Features

- **Dual-module Design**: Consists of both encoder and decoder components
- **Encoder**:
  - Processes the input sequence
  - Composed of multiple layers of self-attention and feed-forward networks
  - Each token can attend to all other tokens in the sequence
- **Decoder**:
  - Generates the output sequence
  - Contains self-attention layers, encoder-decoder attention layers, and feed-forward networks
  - Uses masked attention to ensure predictions only depend on already generated tokens

### Workflow

1. Encoder receives and processes the complete input sequence
2. Decoder generates output tokens one by one
3. When generating each token, the decoder accesses the complete representation from the encoder through cross-attention

### Application Scenarios

Mainly used for sequence-to-sequence (seq2seq) tasks, such as:
- Machine translation
- Text summarization
- Dialogue systems

## Decoder-only Transformer (LLM) Architecture

Modern LLMs like the GPT (Generative Pre-trained Transformer) series adopt a simplified decoder-only architecture.

### Key Features

- **Single-module Design**: Only retains the decoder part of the transformer (but removes the cross-attention layer)
- **Autoregressive Generation**: Predicts the next token based on previous tokens
- **Masked Self-attention**: Ensures each position can only attend to positions before it
- **Scale Expansion**: Parameter count is typically much larger than vanilla transformers

### Workflow

1. The model receives a partial sequence as input (prompt)
2. Using an autoregressive approach, it predicts and generates subsequent tokens one by one
3. Each newly generated token is added to the input for predicting the next token

### Advantages

- **Simplified Architecture**: Removing the encoder simplifies the design
- **Unified Framework**: Views all NLP tasks as text completion problems
- **Long-text Generation**: Particularly suitable for open-ended generation tasks
- **Scalability**: Proven to scale to hundreds of billions of parameters

## Key Differences Comparison

| Feature | Vanilla Transformer | Decoder-only Transformer |
|---------|---------------------|--------------------------|
| Architecture | Encoder-Decoder | Decoder only |
| Attention Mechanism | Encoder: Bidirectional attention<br>Decoder: Unidirectional masked attention + cross-attention | Only unidirectional masked self-attention |
| Information Processing | Encoder encodes the entire input<br>Decoder can access complete encoded information | Can only access previously generated tokens |
| Task Adaptability | Better for explicit transformation tasks | Better for open-ended generation tasks |
| Inference Process | Input processed at once, then output generated step by step | Autoregressive generation, each step depends on previously generated content |
| Parameter Efficiency | Higher for specific tasks | Requires more parameters to achieve similar performance |
| Main Representatives | BERT (encoder-only), T5, BART | GPT series, LLaMA, Claude |

## Technical Details

### Positional Encoding

Both architectures use positional encoding, but implementation differs:
- Vanilla: Uses fixed sine and cosine functions
- Modern LLMs: Typically use learnable positional encodings or Rotary Position Embedding (RoPE)

### Pre-training Methods

- Vanilla (BERT): Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- Decoder-only: Autoregressive language modeling, predicting the next token

### Attention Mechanism

```
// Bidirectional self-attention calculation in Vanilla transformer (simplified)
Q = X * Wq
K = X * Wk
V = X * Wv
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

// Masked self-attention in Decoder-only transformer
// The main difference is using a mask matrix to ensure position i can only attend to positions 0 to i
mask = generateCausalMask(seq_length)  // lower triangular matrix
Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + mask) * V
```

## Why Decoder-only Models Became Mainstream?

1. **Simplicity**: Removes complex encoder-decoder interactions
2. **Unified Interface**: Can transform various NLP tasks into the same format
3. **Scalability**: Proven to scale effectively to massive sizes
4. **Generalization Ability**: Achieves remarkable generalization through large-scale pre-training

## Conclusion

While the vanilla transformer architecture excels in specific tasks, the decoder-only architecture has become the preferred choice for modern LLMs due to its simplicity, scalability, and flexibility. Understanding the differences between these architectures is crucial for comprehending current developments in the NLP field.

Each has its advantages, and the choice of architecture should be based on specific task requirements:
- For tasks requiring bidirectional understanding and explicit transformation: Consider vanilla transformers or encoder-only models
- For open-ended generation and general AI capabilities: Decoder-only LLMs are more suitable

Artificial intelligence is developing rapidly, and these architectures continue to evolve, but understanding the fundamental differences will help grasp future development directions.