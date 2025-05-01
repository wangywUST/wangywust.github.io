# Lecture 7: Decoding Algorithms in Large Language Models (LLMs)

Decoding algorithms are pivotal in determining how Large Language Models (LLMs) generate text sequences. These methods influence the coherence, diversity, and overall quality of the output. This tutorial delves into various decoding strategies, elucidating their mechanisms and applications.

## 1. Introduction to Decoding in LLMs

Decoding in LLMs refers to the process of generating text based on the model's learned probabilities. Given a context or prompt, the model predicts subsequent tokens to construct coherent and contextually relevant text. The choice of decoding strategy significantly impacts the nature of the generated content.

## 2. Common Decoding Strategies

### 2.1 Greedy Search

Greedy Search selects the token with the highest probability at each step, aiming for immediate optimality.

**Mechanism:**

- **Step 1:** Start with an initial prompt.
- **Step 2:** At each position $t$, choose the token $x_t$ that maximizes the conditional probability $P(x_t \mid x_{1:t-1})$.
- **Step 3:** Append $x_t$ to the sequence.
- **Step 4:** Repeat until a stopping criterion is met (e.g., end-of-sequence token).

**Example:**

- **Prompt:** "The quick brown fox"
  
  1. **Step 1 (t=1):**  
     Model predicts:  
     - "jumps" (0.65)  
     - "runs"  (0.20)  
     - "sleeps" (0.15)  
     → Greedy selects **"jumps"**  
     Sequence: "The quick brown fox jumps"
  
  2. **Step 2 (t=2):**  
     Model predicts:  
     - "over"   (0.70)  
     - "under"  (0.20)  
     - "beside" (0.10)  
     → Greedy selects **"over"**  
     Sequence: "The quick brown fox jumps over"
  
  3. **Step 3 (t=3):**  
     Model predicts:  
     - "the"   (0.80)  
     - "a"     (0.15)  
     - "that"  (0.05)  
     → Greedy selects **"the"**  
     Sequence: "The quick brown fox jumps over the"
  
  4. **Step 4 (t=4):**  
     Model predicts:  
     - "lazy"   (0.60)  
     - "sleepy" (0.25)  
     - "hungry" (0.15)  
     → Greedy selects **"lazy"**  
     Sequence: "The quick brown fox jumps over the lazy"
  
  5. **Step 5 (t=5):**  
     Model predicts:  
     - "dog" (0.85)  
     - "."   (0.10)  
     - "cat" (0.05)  
     → Greedy selects **"dog"**  
     Sequence: "The quick brown fox jumps over the lazy dog"
  
  6. **Stopping Criterion:**  
     The next highest-probability token is the end-of-sequence marker, so generation stops.

**Advantages:**

- Simple and computationally efficient.

**Disadvantages:**

- May produce repetitive or generic text.
- Lacks diversity and can miss alternative plausible continuations.

### 2.2 Beam Search

Beam Search maintains multiple candidate sequences (beams) simultaneously, balancing exploration and exploitation.

**Mechanism:**

- **Step 1:** Initialize with the prompt, creating the initial beam.
- **Step 2:** At each step $t$, expand each beam by all possible next tokens.
- **Step 3:** Score each expanded sequence using a scoring function, often the sum of log probabilities.
- **Step 4:** Retain the top $B$ beams based on scores, where $B$ is the beam width.
- **Step 5:** Repeat until beams reach a stopping criterion.

**Example:**

**Prompt:** “The cat”

**Beam width (B):** 2

---

### Iteration 1 (t=1)

We expand “The cat” with three candidate next tokens and their (fictional) cumulative log-scores:

| Candidate                  | Log-Score |
|----------------------------|-----------|
| The cat **sat**            | –1.2      |
| The cat **is**             | –1.5      |
| The cat **was**            | –1.7      |

**Top 2 beams retained:**
1. **The cat sat** (–1.2)  
2. **The cat is**  (–1.5)

---

### Iteration 2 (t=2)

#### Expanding **“The cat sat”**:
- The cat sat **on** (–1.2 + –0.8 = –2.0)  
- The cat sat **under** (–1.2 + –1.1 = –2.3)  

#### Expanding **“The cat is”**:
- The cat is **sleeping** (–1.5 + –0.9 = –2.4)  
- The cat is **cute**     (–1.5 + –0.6 = –2.1)  

All candidates:

| Candidate                        | Log-Score |
|----------------------------------|-----------|
| The cat sat **on**               | –2.0      |
| The cat is **cute**              | –2.1      |
| The cat sat **under**            | –2.3      |
| The cat is **sleeping**          | –2.4      |

**Top 2 beams retained:**
1. **The cat sat on**      (–2.0)  
2. **The cat is cute**     (–2.1)

---

### Iteration 3 (t=3)

#### Expanding **“The cat sat on”**:
- The cat sat on **the** (–2.0 + –0.5 = –2.5)  
- The cat sat on **a**   (–2.0 + –1.0 = –3.0)  

#### Expanding **“The cat is cute”**:
- The cat is cute **and**   (–2.1 + –0.8 = –2.9)  
- The cat is cute **when**  (–2.1 + –1.2 = –3.3)  

All candidates:

| Candidate                          | Log-Score |
|------------------------------------|-----------|
| The cat sat on **the**             | –2.5      |
| The cat is cute **and**            | –2.9      |
| The cat sat on **a**               | –3.0      |
| The cat is cute **when**           | –3.3      |

**Top 2 beams retained:**
1. **The cat sat on the**      (–2.5)  
2. **The cat is cute and**     (–2.9)

---

### Iteration 4 (t=4)

#### Expanding **“The cat sat on the”**:
- The cat sat on the **mat.** (–2.5 + –0.4 = –2.9)  ← **[STOP: ends with “.”]**  
- The cat sat on the **floor** (–2.5 + –0.7 = –3.2)  

#### Expanding **“The cat is cute and”**:
- The cat is cute and **soft.**   (–2.9 + –0.6 = –3.5)  ← **[STOP: ends with “.”]**  
- The cat is cute and **small**   (–2.9 + –1.0 = –3.9)  

All completed candidates (ending in “.”):

| Completed Sentence                      | Log-Score |
|-----------------------------------------|-----------|
| The cat sat on the mat.                 | –2.9      |
| The cat is cute and soft.               | –3.5      |

**Beam Search outputs (best two complete sentences):**
1. **The cat sat on the mat.**  
2. **The cat is cute and soft.**

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

Limits the sampling pool to the top $k$ tokens with the highest probabilities.

**Mechanism:**

- **Step 1:** Compute the probability distribution for the next token.
- **Step 2:** Select the top $k$ tokens with the highest probabilities.
- **Step 3:** Normalize the probabilities of these $k$ tokens.
- **Step 4:** Sample a token from this restricted distribution.
- **Step 5:** Append the sampled token to the sequence.
- **Step 6:** Repeat until a stopping criterion is met.

**Example:**

With \( k = 50 \), the model considers only the top 50 probable tokens at each step, introducing controlled randomness.

**Advantages:**

- Balances diversity and coherence.
- Reduces the chance of selecting low-probability, irrelevant tokens.

**Disadvantages:**

- The choice of $k$ is crucial; too high or too low can affect output quality.

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

With $p = 0.9$, the model dynamically adjusts the number of tokens considered at each step, ensuring that 90% of the probability mass is covered.

**Advantages:**

- Adapts the sampling pool size based on the distribution, providing flexibility.
- Often results in more natural and coherent text.

**Disadvantages:**

- Requires careful tuning of $p$ to balance diversity and coherence.

### 2.4 Temperature Scaling

Temperature scaling adjusts the sharpness of the probability distribution before sampling.

**Mechanism:**

- **Step 1:** Compute the logits (unnormalized probabilities) for the next token.
- **Step 2:** Divide the logits by the temperature $T$ (a positive scalar).
- **Step 3:** Apply the softmax function to obtain the adjusted probabilities.
- **Step 4:** Sample a token from this adjusted distribution.
- **Step 5:** Append the sampled token to the sequence.
- **Step 6:** Repeat until a stopping criterion is met.

**Example:**

- With $T = 1$, the distribution remains unchanged.
- With $T < 1$, the distribution becomes sharper, making high-probability tokens more likely.
- With $T > 1$, the distribution flattens, allowing for more diverse token selection.

**Advantages:**

- Provides control over the randomness of the output.
- Can be combined with other decoding strategies to fine-tune generation behavior.

**Disadvantages:**

- Setting $T$ too high can lead to incoherent text; too low can make the output deterministic.