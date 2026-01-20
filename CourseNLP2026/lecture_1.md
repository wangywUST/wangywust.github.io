## Lecture 1: Overview of NLP

1. [What is language?](#what-is-language)
2. [What is a language model?](#what-is-a-language-model)
3. [What are large language models?](#what-are-large-language-models)

### What is Language?

Language is a systematic means of communicating ideas or feelings using conventionalized signs, sounds, gestures, or marks.

<div style="text-align: center;">
  <img src="./CourseNLP2026/fig_1.jpg" width="50%">
  <p style="margin-top: 10px;">More than 7,000 languages are spoken around the world today, shaping how we describe and perceive the world around us. Source: https://www.snexplores.org/article/lets-learn-about-the-science-of-language</p>
</div>

#### Text in Language

Text represents the written form of language, converting speech and meaning into visual symbols. Key aspects include:

##### Basic Units of Text

Text can be broken down into hierarchical units:
- Characters: The smallest meaningful units in writing systems
- Words: Combinations of characters that carry meaning
- Sentences: Groups of words expressing complete thoughts
- Paragraphs: Collections of related sentences
- Documents: Complete texts serving a specific purpose

##### Text Properties

Text demonstrates several key properties:
- Linearity: Written symbols appear in sequence
- Discreteness: Clear boundaries between units
- Conventionality: Agreed-upon meanings within a language community
- Structure: Follows grammatical and syntactic rules
- Context: Meaning often depends on surrounding text

Question 1: Could you give some examples in English that a word has two different meanings across two sentences?

Based on the above properties shared by different langauges, the NLP researchers develop a unified Machine Learning technique to model language data -- Large Language Models. Let's start to learn this unfied language modeling technique.

### What is a Language Model?

#### Mathematical Definition

A language model is fundamentally a probability distribution over sequences of words or tokens. Mathematically, it can be expressed as:

$$P(w_1, w_2, ..., w_n) = \prod_i P(w_i|w_1, ..., w_{i-1})$$

where:
- $$w_1, w_2, ..., w_n$$ represents a sequence of words or tokens
- The conditional probability of word $$w_i$$ given all previous words is:

  $$P(w_i|w_1, ..., w_{i-1})$$

For practical implementation, this often takes the form:

$$P(w_t|context) = \text{softmax}(h(context) \cdot W)$$

where:
- Target word: $$w_t$$
- Context encoding function: $$h(context)$$
- Weight matrix: $$W$$
- softmax normalizes the output into probabilities

#### **Example 1: Sentence Probability Calculation**

Consider the sentence: "I love chocolate."

The language model predicts the following probabilities:
- $$P(\text{'I'}) = 0.2$$  
- $$P(\text{'love'}|\text{'I'}) = 0.4$$  
- $$P(\text{'chocolate'}|\text{'I love'}) = 0.5$$  

The total probability of the sentence is calculated as:  
$$P(\text{'I love chocolate'}) = P(\text{'I'}) \cdot P(\text{'love'}|\text{'I'}) \cdot P(\text{'chocolate'}|\text{'I love'})$$  
$$P(\text{'I love chocolate'}) = 0.2 \cdot 0.4 \cdot 0.5 = 0.04$$  

Thus, the probability of the sentence "I love chocolate" is **0.04**.

---

#### **Example 2: Dialogue Probability Calculation**

For the dialogue:  
A: "Hello, how are you?"  
B: "I'm fine, thank you."

The model provides the following probabilities:
- **Speaker A's Sentence:**  
  1. $$P(\text{'Hello'}) = 0.3$$  
  2. $$P(\text{','}|\text{'Hello'}) = 0.8$$  
  3. $$P(\text{'how'}|\text{'Hello ,'}) = 0.5$$  
  4. $$P(\text{'are'}|\text{'Hello , how'}) = 0.6$$  
  5. $$P(\text{'you'}|\text{'Hello , how are'}) = 0.7$$  

  $$P(\text{'Hello, how are you?'}) = 0.3 \cdot 0.8 \cdot 0.5 \cdot 0.6 \cdot 0.7 = 0.0504$$  

- **Speaker B's Sentence:**  
  1. $$P(\text{'I'}) = 0.4$$  
  2. $$P(\text{'m'}|\text{'I'}) = 0.5$$  
  3. $$P(\text{'fine'}|\text{'I m'}) = 0.6$$  
  4. $$P(\text{','}|\text{'I m fine'}) = 0.7$$  
  5. $$P(\text{'thank'}|\text{'I m fine ,'}) = 0.8$$  
  6. $$P(\text{'you'}|\text{'I m fine , thank'}) = 0.9$$  

  $$P(\text{'I\'m fine, thank you.'}) = 0.4 \cdot 0.5 \cdot 0.6 \cdot 0.7 \cdot 0.8 \cdot 0.9 = 0.06048$$  

- **Total Probability for the Dialogue:**  
  Combine the probabilities for both sentences:  
  $$P(\text{'Hello, how are you? I\'m fine, thank you.'}) = P(\text{'Hello, how are you?'}) \cdot P(\text{'I\'m fine, thank you.'})$$  
  $$P(\text{'Hello, how are you? I\'m fine, thank you.'}) = 0.0504 \cdot 0.06048 = 0.003048192$$  

Thus, the total probability of the dialogue is approximately **0.00305**.

---

#### **Example 3: Partial Sentence Generation**

Consider the sentence: "The dog barked loudly."

The probabilities assigned by the language model are:
- $$P(\text{'The'}) = 0.25$$  
- $$P(\text{'dog'}|\text{'The'}) = 0.4$$  
- $$P(\text{'barked'}|\text{'The dog'}) = 0.5$$  
- $$P(\text{'loudly'}|\text{'The dog barked'}) = 0.6$$  

Question 2: Calculate the total probability of the sentence $$P(\text{'The dog barked loudly'})$$ using the given probabilities.

### The Transformer Model: Revolutionizing Language Models

The emergence of the Transformer architecture marked a paradigm shift in how machines process and understand human language. Unlike its predecessors, which struggled with long-range patterns in text, this groundbreaking architecture introduced mechanisms that revolutionized natural language processing (NLP).

#### The Building Blocks of Language Understanding

##### From Text to Machine-Readable Format

Before any sophisticated processing can occur, raw text must be converted into a format that machines can process. This happens in two crucial stages:

1. **Text Segmentation**
The first challenge is breaking down text into meaningful units. Imagine building with LEGO blocks - just as you need individual blocks to create complex structures, language models need discrete pieces of text to work with. These pieces, called tokens, might be:
- Complete words
- Parts of words
- Individual characters
- Special symbols

For instance, the phrase "artificial intelligence" might become ["art", "ificial", "intel", "ligence"], allowing the model to recognize patterns even in unfamiliar words.

2. **Numerical Representation**
Once we have our text pieces, each token gets transformed into a numerical vector - essentially a long list of numbers. Think of this as giving each word or piece its own unique mathematical "fingerprint" that captures its meaning and relationships with other words.

##### Adding Sequential Understanding

One of the most innovative aspects of Transformers is how they handle word order. Rather than treating text like a bag of unrelated words, the architecture adds precise positional information to each token's representation.

Consider how the meaning changes in these sentences:
- "The cat chased the mouse"
- "The mouse chased the cat"

The words are identical, but their positions completely change the meaning. The Transformer's positional encoding system ensures this crucial information isn't lost.

#### The Heart of the System: Information Processing

##### Context Through Self-Attention

The true magic of Transformers lies in their attention mechanism. Unlike humans who must read text sequentially, Transformers can simultaneously analyze relationships between all words in a text. This is similar to how you might solve a complex puzzle:

1. First, you look at all the pieces simultaneously
2. Then, you identify which pieces are most likely to connect
3. Finally, you use these relationships to build the complete picture

In language, this means the model can:
- Resolve pronouns ("She picked up her book" - who is "her" referring to?)
- Understand idiomatic expressions ("kicked the bucket" means something very different from "kicked the ball")
- Grasp long-distance dependencies ("The keys, which I thought I had left on the kitchen counter yesterday morning, were actually in my coat pocket")

#### Real-World Applications and Impact

The Transformer architecture has enabled breakthrough applications in:

1. **Cross-Language Communication**
- Real-time translation systems
- Multilingual document processing

2. **Content Creation and Analysis**
- Automated report generation
- Text summarization
- Content recommendations

3. **Specialized Industry Applications**
- Legal document analysis
- Medical record processing
- Scientific literature review

#### The Road Ahead

As this architecture continues to evolve, we're seeing:
- More efficient processing methods
- Better handling of specialized domains
- Improved understanding of contextual nuances
- Enhanced ability to work with multimodal inputs

The Transformer architecture represents more than just a technical advancement - it's a fundamental shift in how machines can understand and process human language. Its impact continues to grow as researchers and developers find new ways to apply and improve upon its core principles.

The true power of Transformers lies not just in their technical capabilities, but in how they've opened new possibilities for human-machine interaction and understanding. As we continue to refine and build upon this architecture, we're moving closer to systems that can truly understand and engage with human language in all its complexity and nuance.

### What are large language models?

Large language models are transformers with billions to trillions of parameters, trained on massive amounts of text data. These models have several distinguishing characteristics:

1. **Scale**: Models contain billions of parameters and are trained on hundreds of billions of tokens
2. **Architecture**: Based on the Transformer architecture with self-attention mechanisms
3. **Emergent abilities**: Complex capabilities that emerge with scale
4. **Few-shot learning**: Ability to adapt to new tasks with few examples

- **Definition**: Large Language Models are artificial intelligence systems trained on vast amounts of text data, containing hundreds of billions of parameters. Unlike traditional AI models, they can understand and generate human-like text across a wide range of tasks and domains.

- **Scale and Architecture**:
  - Typically contain >1B parameters (Some exceed 500B)
  - Built on Transformer architecture with attention mechanisms
  - Require massive computational resources for training
  - Examples: GPT-3 (175B), PaLM (540B), LLaMA (65B)

- **Key Capabilities**:
  - Natural language understanding and generation
  - Task adaptation without fine-tuning
  - Complex reasoning and problem solving
  - Knowledge storage and retrieval
  - Multi-turn conversation

## Historical Evolution

### 1. Statistical Language Models (SLM) - 1990s
- **Core Technology**: Used statistical methods to predict next words based on previous context
- **Key Features**: 
  - N-gram models (bigram, trigram)
  - Markov assumption for word prediction
  - Used in early IR and NLP applications
- **Limitations**:
  - Curse of dimensionality
  - Data sparsity issues
  - Limited context window
  - Required smoothing techniques

### 2. Neural Language Models (NLM) - 2013
- **Core Technology**: Neural networks for language modeling
- **Key Advances**:
  - Distributed word representations
  - Multi-layer perceptron and RNN architectures
  - End-to-end learning
  - Better feature extraction
- **Impact**:
  - Word2vec and similar embedding models
  - Improved generalization
  - Reduced need for feature engineering

### 3. Pre-trained Language Models (PLM) - 2018
- **Core Technology**: Transformer-based models with pre-training
- **Key Innovations**:
  - BERT and bidirectional context modeling
  - GPT and autoregressive modeling
  - Transfer learning approach
  - Fine-tuning paradigm
- **Benefits**:
  - Context-aware representations
  - Better task performance
  - Reduced need for task-specific data
  - More efficient training

### 4. Large Language Models (LLM) - 2020+
- **Core Technology**: Scaled-up Transformer models
- **Major Breakthroughs**:
  - Emergence of new abilities with scale
  - Few-shot and zero-shot learning
  - General-purpose problem solving
  - Human-like interaction capabilities
- **Key Examples**:
  - GPT-3: First demonstration of powerful in-context learning
  - ChatGPT: Advanced conversational abilities
  - GPT-4: Multimodal capabilities and improved reasoning
  - PaLM: Enhanced multilingual and reasoning capabilities

## Key Features of LLMs

### Scaling Laws

1. **KM Scaling Law (OpenAI)**:
   - Describes relationship between model performance (measured by cross entropy loss $L$) and three factors:
     - Model size ($N$)
     - Dataset size ($D$)
     - Computing power ($C$)
   - Mathematical formulations:
     - $L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$, where $\alpha_N \sim 0.076$, $N_c \sim 8.8 \times 10^{13}$
     - $L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$, where $\alpha_D \sim 0.095$, $D_c \sim 5.4 \times 10^{13}$
     - $L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$, where $\alpha_C \sim 0.050$, $C_c \sim 3.1 \times 10^8$
   - Predicts diminishing returns as model/data/compute scale increases
   - Helps optimize resource allocation for training

2. **Chinchilla Scaling Law (DeepMind)**:
   - **Mathematical formulation**:
     - $L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$
     - where $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$
   - **Optimal compute allocation**:
     - $N_{opt}(C) = G\left(\frac{C}{6}\right)^a$
     - $D_{opt}(C) = G^{-1}\left(\frac{C}{6}\right)^b$
     - where $a = \frac{\alpha}{\alpha+\beta}$, $b = \frac{\beta}{\alpha+\beta}$
   - Suggests equal scaling of model and data size
   - More efficient compute utilization than KM scaling law
   - Demonstrated superior performance with smaller models trained on more data

### Emergent Abilities

1. **In-context Learning**
   - **Definition**: Ability to learn from examples in the prompt
   - **Characteristics**:
     - No parameter updates required
     - Few-shot and zero-shot capabilities
     - Task adaptation through demonstrations
   - **Emergence Point**: 
     - GPT-3 showed first strong results

Question 3: Design a few-shot prompt that can classify the film topic by the film name. It must be able to correctly classify more than 5 films proposed by other students. Using ChatGPT as the test LLM.

2. **Instruction Following**
   - **Definition**: Ability to understand and execute natural language instructions
   - **Requirements**:
     - Instruction tuning
     - Multi-task training
     - Natural language task descriptions

3. **Step-by-step Reasoning**
   - **Definition**: Ability to break down complex problems
   - **Techniques**:
     - Chain-of-thought prompting
     - Self-consistency methods
     - Intermediate step generation
   - **Benefits**:
     - Better problem solving
     - More reliable answers
     - Transparent reasoning process

## Technical Elements

### Architecture

1. **Transformer Base**
   - **Components**:
     - Multi-head attention mechanism
     - Feed-forward neural networks
     - Layer normalization
     - Positional encoding
   - **Variations**:
     - Decoder-only (GPT-style)
     - Encoder-decoder (T5-style)
     - Modifications for efficiency

2. **Scaling Considerations**
   - **Hardware Requirements**:
     - Distributed training systems
     - Memory optimization
     - Parallel processing
   - **Architecture Choices**:
     - Layer count
     - Hidden dimension size
     - Attention head configuration

### Training Process

1. **Pre-training**
   - **Data Preparation**:
     - Web text
     - Books
     - Code
     - Scientific papers
   - **Objectives**:
     - Next token prediction
     - Masked language modeling
     - Multiple auxiliary tasks

2. **Adaptation Methods**
   - **Instruction Tuning**:
     - Natural language task descriptions
     - Multi-task learning
     - Task generalization
   - **RLHF**:
     - Human preference learning
     - Safety alignment
     - Behavior optimization

### Utilization Techniques

1. **Prompting Strategies**
   - **Basic Prompting**:
     - Direct instructions
     - Few-shot examples
     - Zero-shot prompts
   - **Advanced Methods**:
     - Chain-of-thought
     - Self-consistency
     - Tool augmentation

2. **Application Patterns**
   - **Task Types**:
     - Generation
     - Classification
     - Question answering
     - Coding
   - **Integration Methods**:
     - API endpoints
     - Model serving
     - Application backends

## Major Milestones

### ChatGPT (2022)
1. **Technical Achievements**
   - Advanced dialogue capabilities
   - Robust safety measures
   - Consistent persona
   - Tool integration

2. **Impact**
   - Widespread adoption
   - New application paradigms
   - Industry transformation
   - Public AI awareness

### GPT-4 (2023)
1. **Key Advances**
   - Multimodal understanding
   - Enhanced reliability
   - Better reasoning
   - Improved safety

2. **Technical Features**
   - Predictable scaling
   - Vision capabilities
   - Longer context window
   - Advanced system prompting

## Challenges and Future Directions

### Current Challenges

1. **Computational Resources**
   - **Training Costs**:
     - Massive energy requirements
     - Expensive hardware needs
     - Limited accessibility
   - **Infrastructure Needs**:
     - Specialized facilities
     - Cooling systems
     - Power management

2. **Data Requirements**
   - **Quality Issues**:
     - Data cleaning
     - Content filtering
     - Bias mitigation
   - **Privacy Concerns**:
     - Personal information
     - Copyright issues
     - Regulatory compliance

3. **Safety and Alignment**
   - **Technical Challenges**:
     - Hallucination prevention
     - Truthfulness
     - Bias detection
   - **Ethical Considerations**:
     - Harm prevention
     - Fairness
     - Transparency

### Future Directions

1. **Improved Efficiency**
   - **Architecture Innovation**:
     - Sparse attention
     - Parameter efficiency
     - Memory optimization
   - **Training Methods**:
     - Better scaling laws
     - Efficient fine-tuning
     - Reduced compute needs

2. **Enhanced Capabilities**
   - **Multimodal Understanding**:
     - Vision-language integration
     - Audio processing
     - Sensor data interpretation
   - **Reasoning Abilities**:
     - Logical deduction
     - Mathematical problem solving
     - Scientific reasoning

3. **Safety Development**
   - **Alignment Techniques**:
     - Value learning
     - Preference optimization
     - Safety bounds
   - **Evaluation Methods**:
     - Robustness testing
     - Safety metrics
     - Bias assessment

## Summary

- LLMs represent a fundamental shift in AI capabilities
- Scale and architecture drive emergent abilities
- Continuing rapid development in capabilities
- Balance between advancement and safety
- Growing impact on society and technology
- Need for responsible development and deployment

## References and Further Reading
- Scaling Laws Papers
- Emergent Abilities Research
- Safety and Alignment Studies
- Technical Documentation
- Industry Reports

Paper Reading: [A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223)