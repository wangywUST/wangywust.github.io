# Tutorial on Decoding Algorithms in Large Language Models (LLMs)

Decoding algorithms are pivotal in determining how Large Language Models (LLMs) generate text sequences. These methods influence the coherence, diversity, and overall quality of the output. This tutorial delves into various decoding strategies, elucidating their mechanisms and applications.

## 1. Introduction to Decoding in LLMs

Decoding in LLMs refers to the process of generating text based on the model's learned probabilities. Given a context or prompt, the model predicts subsequent tokens to construct coherent and contextually relevant text. The choice of decoding strategy significantly impacts the nature of the generated content.

## 2. Common Decoding Strategies

### 2.1 Greedy Search

Greedy Search selects the token with the highest probability at each step, aiming for immediate optimality.

**Mechanism:**

- **Step 1:** Start with an initial prompt.
- **Step 2:** At each position \( t \), choose the token \( x_t \) that maximizes the conditional probability \( P(x_t \mid x_{1:t-1}) \).
- **Step 3:** Append \( x_t \) to the sequence.
- **Step 4:** Repeat until a stopping criterion is met (e.g., end-of-sequence token).

**Example:**

Given the prompt "The capital of France is", the model might generate "Paris" by selecting the highest-probability token at each step.

**Advantages:**

- Simple and computationally efficient.

**Disadvantages:**

- May produce repetitive or generic text.
- Lacks diversity and can miss alternative plausible continuations.

### 2.2 Beam Search

Beam Search maintains multiple candidate sequences (beams) simultaneously, balancing exploration and exploitation.

**Mechanism:**

- **Step 1:** Initialize with the prompt, creating the initial beam.
- **Step 2:** At each step \( t \), expand each beam by all possible next tokens.
- **Step 3:** Score each expanded sequence using a scoring function, often the sum of log probabilities.
- **Step 4:** Retain the top \( B \) beams based on scores, where \( B \) is the beam width.
- **Step 5:** Repeat until beams reach a stopping criterion.

**Example:**

For a beam width of 3, the model explores three parallel sequences, selecting the most probable completions among them.

**Advantages:**

- Explores multiple hypotheses, reducing the risk of suboptimal sequences.

**Disadvantages:**

- Computationally more intensive than Greedy Search.
- Can still produce repetitive outputs if not combined with other techniques.

### 2.3 Sampling-Based Methods

Sampling introduces randomness into the generation process, allowing for more diverse outputs.

#### 2.3.1 Random Sampling

Tokens are selected randomly based on their conditional probabilities.

**Mechanism:**

- **Step 1:** Compute the probability distribution over the vocabulary for the next token.
- **Step 2:** Sample a token from this distribution.
- **Step 3:** Append the sampled token to the sequence.
- **Step 4:** Repeat until a stopping criterion is met.

**Example:**

Given the prompt "Once upon a time", the model might generate various continuations like "a princess lived" or "a dragon roamed", depending on the sampling.

**Advantages:**

- Produces varied and creative outputs.

**Disadvantages:**

- Can lead to incoherent or less relevant text.
- Quality depends heavily on the underlying probability distribution.

#### 2.3.2 Top-k Sampling

Limits the sampling pool to the top \( k \) tokens with the highest probabilities.

**Mechanism:**

- **Step 1:** Compute the probability distribution for the next token.
- **Step 2:** Select the top \( k \) tokens with the highest probabilities.
- **Step 3:** Normalize the probabilities of these \( k \) tokens.
- **Step 4:** Sample a token from this restricted distribution.
- **Step 5:** Append the sampled token to the sequence.
- **Step 6:** Repeat until a stopping criterion is met.

**Example:**

With \( k = 50 \), the model considers only the top 50 probable tokens at each step, introducing controlled randomness.

**Advantages:**

- Balances diversity and coherence.
- Reduces the chance of selecting low-probability, irrelevant tokens.

**Disadvantages:**

- The choice of \( k \) is crucial; too high or too low can affect output quality.

#### 2.3.3 Top-p (Nucleus) Sampling

Considers the smallest set of top tokens whose cumulative probability exceeds a threshold \( p \).

**Mechanism:**

- **Step 1:** Compute the probability distribution for the next token.
- **Step 2:** Sort tokens by probability in descending order.
- **Step 3:** Select the smallest set of tokens whose cumulative probability is at least \( p \).
- **Step 4:** Normalize the probabilities of these tokens.
- **Step 5:** Sample a token from this distribution.
- **Step 6:** Append the sampled token to the sequence.
- **Step 7:** Repeat until a stopping criterion is met.

**Example:**

With \( p = 0.9 \), the model dynamically adjusts the number of tokens considered at each step, ensuring that 90% of the probability mass is covered.

**Advantages:**

- Adapts the sampling pool size based on the distribution, providing flexibility.
- Often results in more natural and coherent text.

**Disadvantages:**

- Requires careful tuning of \( p \) to balance diversity and coherence.

### 2.4 Temperature Scaling

Temperature scaling adjusts the sharpness of the probability distribution before sampling.

**Mechanism:**

- **Step 1:** Compute the logits (unnormalized probabilities) for the next token.
- **Step 2:** Divide the logits by the temperature \( T \) (a positive scalar).
- **Step 3:** Apply the softmax function to obtain the adjusted probabilities.
- **Step 4:** Sample a token from this adjusted distribution.
- **Step 5:** Append the sampled token to the sequence.
- **Step 6:** Repeat until a stopping criterion is met.

**Example:**

- With \( T = 1 \), the distribution remains unchanged.
- With \( T < 1 \), the distribution becomes sharper, making high-probability tokens more likely.
- With \( T > 1 \), the distribution flattens, allowing for more diverse token selection.

**Advantages:**

- Provides control over the randomness of the output.
- Can be combined with other decoding strategies to fine-tune generation behavior.

**Disadvantages:**

- Setting \( T \) too high can lead to incoherent text; too low can make the output deterministic.