## Lecture 1: Overview of LLMs

1. [What is a language model?](#what-is-a-language-model)
1. [What are large language models?](#what-are-large-language-models)
1. [Grading of This Course](#grading-of-this-course)
1. [In-Course Questions](#in-course-questions)

# What is Language?

Language is a systematic means of communicating ideas or feelings using conventionalized signs, sounds, gestures, or marks.

## Text in Language

Text represents the written form of language, converting speech and meaning into visual symbols. Key aspects include:

### Basic Units of Text

Text can be broken down into hierarchical units:
- Characters: The smallest meaningful units in writing systems
- Words: Combinations of characters that carry meaning
- Sentences: Groups of words expressing complete thoughts
- Paragraphs: Collections of related sentences
- Documents: Complete texts serving a specific purpose

### Text Properties

Text demonstrates several key properties:
- Linearity: Written symbols appear in sequence
- Discreteness: Clear boundaries between units
- Conventionality: Agreed-upon meanings within a language community
- Structure: Follows grammatical and syntactic rules
- Context: Meaning often depends on surrounding text

Based on the above properties shared by different langauges, the NLP researchers develop a unified Machine Learning technique to model language data -- Large Language Models. Let's start to learn this unfied language modeling technique.

![Words in documents that get filtered out of C4](../images/c4-excluded.png)

# What is a Language Model?

## Mathematical Definition

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

# Why Use Conditional Probability in Language Models?

### Core Insight
From a classification perspective, the number of categories directly impacts the learning difficulty - more categories require exponentially more training data to achieve adequate coverage.

### Comparing Two Approaches

#### Joint Probability Approach
When modeling $$P(w_1,...,w_n)$$ directly:
- Needs to predict $$V^n$$ categories
- Requires seeing enough samples of each possible sentence
- Most long sequences may never appear in training data
- Makes learning practically impossible

#### Conditional Probability Approach
When modeling $$P(w_i|w_1,...,w_{i-1})$$:
- Only predicts $$V$$ categories at each step
- Each word position provides a training sample
- Same words in different contexts contribute learning signals
- Dramatically improves data efficiency

### Numerical Example
Consider a language model with:
- Vocabulary size $$V = 10,000$$
- Sequence length $$n = 5$$

Then:
- Joint probability: Must learn $$10,000^5$$ categories
- Conditional probability: Must learn $$10,000$$ categories at each step

### Why This Matters
1. Training Data Requirements
- More categories require more training examples
- Each category needs sufficient representation
- Data requirements grow exponentially with category count

2. Learning Efficiency
- Smaller category spaces are easier to model
- More efficient use of training data
- Each word occurrence contributes to learning

3. Statistical Coverage
- Impossible to see all possible sequences
- But possible to see all words in various contexts
- Makes learning feasible with finite training data

### Conclusion
The conditional probability formulation cleverly transforms an intractable large-scale classification problem into a series of manageable smaller classification problems. This is the fundamental reason why language models can learn effectively from finite training data.

## Real-world Application: Text Completion

### The Prefix-based Generation Task
In practical applications, we often:
- Have a fixed prefix of text
- Need to predict/generate the continuation
- Don't need to generate text from scratch

### Examples
1. Auto-completion
- Code completion in IDEs
- Search query suggestions
- Email text completion

2. Text Generation
- Story continuation
- Dialogue response generation
- Document completion

### Why Conditional Probability Helps
The formulation $$P(w_i|w_1,...,w_{i-1})$$ naturally fits this scenario because:
- We can directly condition on the given prefix
- No need to model the probability of the prefix itself
- Can focus computational resources on predicting what comes next

### Comparison with Joint Probability
The joint probability $$P(w_1,...,w_n)$$ would be less suitable because:
- Would need to model probability of the fixed prefix
- Wastes computation on already-known parts
- Doesn't directly give us what we want (continuation probability)

This alignment between the mathematical formulation and practical use cases is another key advantage of the conditional probability approach in language modeling.

# The Transformer Model: Revolutionizing Language Models

The emergence of the Transformer architecture marked a paradigm shift in how machines process and understand human language. Unlike its predecessors, which struggled with long-range patterns in text, this groundbreaking architecture introduced mechanisms that revolutionized natural language processing (NLP).

## The Building Blocks of Language Understanding

### From Text to Machine-Readable Format

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

### Adding Sequential Understanding

One of the most innovative aspects of Transformers is how they handle word order. Rather than treating text like a bag of unrelated words, the architecture adds precise positional information to each token's representation.

Consider how the meaning changes in these sentences:
- "The cat chased the mouse"
- "The mouse chased the cat"

The words are identical, but their positions completely change the meaning. The Transformer's positional encoding system ensures this crucial information isn't lost.

## The Heart of the System: Information Processing

### Context Through Self-Attention

The true magic of Transformers lies in their attention mechanism. Unlike humans who must read text sequentially, Transformers can simultaneously analyze relationships between all words in a text. This is similar to how you might solve a complex puzzle:

1. First, you look at all the pieces simultaneously
2. Then, you identify which pieces are most likely to connect
3. Finally, you use these relationships to build the complete picture

In language, this means the model can:
- Resolve pronouns ("She picked up her book" - who is "her" referring to?)
- Understand idiomatic expressions ("kicked the bucket" means something very different from "kicked the ball")
- Grasp long-distance dependencies ("The keys, which I thought I had left on the kitchen counter yesterday morning, were actually in my coat pocket")

### Information Refinement

After the attention mechanism identifies relevant connections, the information passes through a series of specialized neural networks. These networks:
- Combine and transform the gathered context
- Extract higher-level patterns
- Refine the understanding of each piece of text

## Generation and Decision Making

The final stage involves converting all this processed information into useful output. Whether the task is:
- Completing a sentence
- Translating text
- Answering a question
- Summarizing a document

The model uses a probability distribution system to select the most appropriate output. This is similar to a skilled writer choosing the perfect word from their vocabulary, considering both meaning and context.

## Real-World Applications and Impact

The Transformer architecture has enabled breakthrough applications in:

1. **Cross-Language Communication**
- Real-time translation systems
- Multilingual document processing
- Cultural context adaptation

2. **Content Creation and Analysis**
- Automated report generation
- Text summarization
- Content recommendations

3. **Specialized Industry Applications**
- Legal document analysis
- Medical record processing
- Scientific literature review

## The Road Ahead

As this architecture continues to evolve, we're seeing:
- More efficient processing methods
- Better handling of specialized domains
- Improved understanding of contextual nuances
- Enhanced ability to work with multimodal inputs

The Transformer architecture represents more than just a technical advancement - it's a fundamental shift in how machines can understand and process human language. Its impact continues to grow as researchers and developers find new ways to apply and improve upon its core principles.

The true power of Transformers lies not just in their technical capabilities, but in how they've opened new possibilities for human-machine interaction and understanding. As we continue to refine and build upon this architecture, we're moving closer to systems that can truly understand and engage with human language in all its complexity and nuance.

# What are large language models?

Large language models are transformers with billions to trillions of parameters, trained on massive amounts of text data. These models have several distinguishing characteristics:

1. **Scale**: Models contain billions of parameters and are trained on hundreds of billions of tokens
2. **Architecture**: Based on the Transformer architecture with self-attention mechanisms
3. **Emergent abilities**: Complex capabilities that emerge with scale
4. **Few-shot learning**: Ability to adapt to new tasks with few examples