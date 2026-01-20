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

Here’s a concrete English example of Beam Search (beam width = 2) generating the sentence **“The cat sat on the mat.”** Step by step, showing the beam prefixes (what’s in cache), candidate next-tokens with their log-probs, and which beams survive at each iteration.

---

#### **Prompt (initial beam)**  
```
“The cat”
```

---

### **Step 1 (t=1)**  
We expand **“The cat”** to all candidate next tokens; here we show the top 4 by log-prob:

| Beam Prefix     | Next Token | Log Prob | Cumulative Score |
|-----------------|------------|----------|------------------|
| “The cat”       |  sat       | –0.10    | –0.10            |
| “The cat”       |  is        | –1.20    | –1.20            |
| “The cat”       |  on        | –1.50    | –1.50            |
| “The cat”       |  meows     | –2.00    | –2.00            |

**Keep top 2 beams** (smallest negative score):  
1. **“The cat sat”** (–0.10)  
2. **“The cat is”**  (–1.20)  

---

### **Step 2 (t=2)**  
Expand each surviving beam:

| Beam Prefix       | Next Token | Log Prob | Cumulative Score |
|-------------------|------------|----------|------------------|
| **“The cat sat”** |  on        | –0.05    | –0.15            |
| **“The cat sat”** |  quietly   | –1.00    | –1.10            |
| **“The cat is”**  |  sleeping  | –0.20    | –1.40            |
| **“The cat is”**  |  hungry    | –0.50    | –1.70            |

**Keep top 2 beams**:  
1. **“The cat sat on”**      (–0.15)  
2. **“The cat sat quietly”** (–1.10)  

---

### **Step 3 (t=3)**  

| Beam Prefix            | Next Token | Log Prob | Cumulative Score |
|------------------------|------------|----------|------------------|
| **“The cat sat on”**      |  the       | –0.02    | –0.17            |
| **“The cat sat on”**      |  a         | –0.30    | –0.45            |
| **“The cat sat quietly”** |  on        | –0.40    | –1.50            |
| **“The cat sat quietly”** |  in        | –0.60    | –1.70            |

**Keep top 2 beams**:  
1. **“The cat sat on the”**    (–0.17)  
2. **“The cat sat on a”**      (–0.45)  

---

### **Step 4 (t=4)**  

| Beam Prefix               | Next Token | Log Prob | Cumulative Score |
|---------------------------|------------|----------|------------------|
| **“The cat sat on the”**  |  mat       | –0.01    | –0.18            |
| **“The cat sat on the”**  |  rug       | –1.00    | –1.17            |
| **“The cat sat on a”**    |  chair     | –0.50    | –0.95            |
| **“The cat sat on a”**    |  bed       | –0.80    | –1.25            |

**Keep top 2 beams**:  
1. **“The cat sat on the mat”** (–0.18)  
2. **“The cat sat on a chair”** (–0.95)  

---

### **Step 5 (t=5)**  
We stop beams when they output a period token “.”. Only the first beam has that option with a high score:

| Beam Prefix                       | Next Token | Log Prob | Cumulative Score |
|-----------------------------------|------------|----------|------------------|
| **“The cat sat on the mat”**      |  .         | –0.01    | –0.19            |
| **“The cat sat on a chair”**      |  .         | –0.05    | –1.00            |

**Keep top 1 completed beam**:  
- **“The cat sat on the mat.”** (–0.19)

---

### **Final Output**  
>The cat sat on the mat.

This illustrates how Beam Search maintains multiple prefixes, scores them, and finally selects the highest-scoring complete sentence.

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

In-class Question: Even setting the temperature value as 0, sometimes we can see LLMs to output different outputs given the same prefix. What is the possible cause? Should the LLMs' outputs always be consistent given the temperature as 0?