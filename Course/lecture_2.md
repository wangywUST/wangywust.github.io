# Lecture 2: Understanding Tokenization in Language Models

Tokenization is a fundamental concept in Natural Language Processing (NLP) that involves breaking down text into smaller units called tokens. These tokens can be words, subwords, or characters, depending on the tokenization strategy used. The choice of tokenization method can significantly impact a model's performance and its ability to handle various languages and vocabularies.

## Common Tokenization Approaches

1. **Word-based Tokenization**
   - Splits text at word boundaries (usually spaces and punctuation)
   - Simple and intuitive but struggles with out-of-vocabulary words
   - Requires a large vocabulary to cover most words
   - Examples: Early versions of BERT used WordPiece tokenization

2. **Character-based Tokenization**
   - Splits text into individual characters
   - Very small vocabulary size
   - Can handle any word but loses word-level meaning
   - Typically results in longer sequences

3. **Subword Tokenization**
   - Breaks words into meaningful subunits
   - Balances vocabulary size and semantic meaning
   - Better handles rare words
   - Popular methods include:
     - Byte-Pair Encoding (BPE)
     - WordPiece
     - Unigram
     - SentencePiece

Let's dive deep into one of the most widely used subword tokenization methods: Byte-Pair Encoding (BPE).

# Byte-Pair Encoding (BPE) Tokenization

Reference Tutorial: [Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)

Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model. It's used by many Transformer models, including GPT, GPT-2, Llama1, Llama2, Llama3, RoBERTa, BART, and DeBERTa.

## Training Algorithm

BPE training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words. As a very simple example, let's say our corpus uses these five words:

```
"hug", "pug", "pun", "bun", "hugs"
```

The base vocabulary will then be `["b", "g", "h", "n", "p", "s", "u"]`. For real-world cases, that base vocabulary will contain all the ASCII characters, at the very least, and probably some Unicode characters as well. If an example you are tokenizing uses a character that is not in the training corpus, that character will be converted to the unknown token. That's one reason why lots of NLP models are very bad at analyzing content with emojis.

> The GPT-2 and RoBERTa tokenizers (which are pretty similar) have a clever way to deal with this: they don't look at words as being written with Unicode characters, but with bytes. This way the base vocabulary has a small size (256), but every character you can think of will still be included and not end up being converted to the unknown token. This trick is called byte-level BPE.

After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning merges, which are rules to merge two elements of the existing vocabulary together into a new one. So, at the beginning these merges will create tokens with two characters, and then, as training progresses, longer subwords.

At any step during the tokenizer training, the BPE algorithm will search for the most frequent pair of existing tokens (by "pair," here we mean two consecutive tokens in a word). That most frequent pair is the one that will be merged, and we rinse and repeat for the next step.

Going back to our previous example, let's assume the words had the following frequencies:

```python
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

meaning "hug" was present 10 times in the corpus, "pug" 5 times, "pun" 12 times, "bun" 4 times, and "hugs" 5 times. We start the training by splitting each word into characters (the ones that form our initial vocabulary) so we can see each word as a list of tokens:

```python
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

Then we look at pairs. The pair ("h", "u") is present in the words "hug" and "hugs", so 15 times total in the corpus. It's not the most frequent pair, though: that honor belongs to ("u", "g"), which is present in "hug", "pug", and "hugs", for a grand total of 20 times in the vocabulary.

Thus, the first merge rule learned by the tokenizer is ("u", "g") -> "ug", which means that "ug" will be added to the vocabulary, and the pair should be merged in all the words of the corpus. At the end of this stage, the vocabulary and corpus look like this:

```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

Now we have some pairs that result in a token longer than two characters: the pair ("h", "ug"), for instance (present 15 times in the corpus). The most frequent pair at this stage is ("u", "n"), however, present 16 times in the corpus, so the second merge rule learned is ("u", "n") -> "un". Adding that to the vocabulary and merging all existing occurrences leads us to:

```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

Now the most frequent pair is ("h", "ug"), so we learn the merge rule ("h", "ug") -> "hug", which gives us our first three-letter token. After the merge, the corpus looks like this:

```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]
Corpus: ("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```

And we continue like this until we reach the desired vocabulary size.

## Tokenization Inference

Tokenization inference follows the training process closely, in the sense that new inputs are tokenized by applying the following steps:

1. Splitting the words into individual characters
2. Applying the merge rules learned in order on those splits

Let's take the example we used during training, with the three merge rules learned:

```python
("u", "g") -> "ug"
("u", "n") -> "un"
("h", "ug") -> "hug"
```

The word "bug" will be tokenized as `["b", "ug"]`. "mug", however, will be tokenized as `["[UNK]", "ug"]` since the letter "m" was not in the base vocabulary. Likewise, the word "thug" will be tokenized as `["[UNK]", "hug"]`: the letter "t" is not in the base vocabulary, and applying the merge rules results first in "u" and "g" being merged and then "h" and "ug" being merged.

## Implementing BPE

Now let's take a look at an implementation of the BPE algorithm. This won't be an optimized version you can actually use on a big corpus; we just want to show you the code so you can understand the algorithm a little bit better.

I present the colab link for you to reproduce this part's experiments easily: [Colab BPE](https://colab.research.google.com/drive/13F5T8aikeumAu2Oxpaabczvqhxkih8mF?usp=sharing)

### Training BPE

First we need a corpus, so let's create a simple one with a few sentences:

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens."
]
```

Next, we need to pre-tokenize that corpus into words. Since we are replicating a BPE tokenizer (like GPT-2), we will use the gpt2 tokenizer for the pre-tokenization:

```python
from transformers import AutoTokenizer

# init pre tokenize function
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
pre_tokenize_function = gpt2_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str

# pre tokenize
pre_tokenized_corpus = [pre_tokenize_function(text) for text in corpus]
```

We have the output

```python
[
    [('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11)), ('ĠHugging', (11, 19)), ('ĠFace', (19, 24)), ('ĠCourse', (24, 31)), ('.', (31, 32))], 
    [('This', (0, 4)), ('Ġchapter', (4, 12)), ('Ġis', (12, 15)), ('Ġabout', (15, 21)), ('Ġtokenization', (21, 34)), ('.', (34, 35))], 
    [('This', (0, 4)), ('Ġsection', (4, 12)), ('Ġshows', (12, 18)), ('Ġseveral', (18, 26)), ('Ġtokenizer', (26, 36)), ('Ġalgorithms', (36, 47)), ('.', (47, 48))], 
    [('Hopefully', (0, 9)), (',', (9, 10)), ('Ġyou', (10, 14)), ('Ġwill', (14, 19)), ('Ġbe', (19, 22)), ('Ġable', (22, 27)), ('Ġto', (27, 30)), ('Ġunderstand', (30, 41)), ('Ġhow', (41, 45)), ('Ġthey', (45, 50)), ('Ġare', (50, 54)), ('Ġtrained', (54, 62)), ('Ġand', (62, 66)), ('Ġgenerate', (66, 75)), ('Ġtokens', (75, 82)), ('.', (82, 83))]
]
```

Then we compute the frequencies of each word in the corpus as we do the pre-tokenization:

```python
from collections import defaultdict
word2count = defaultdict(int)
for split_text in pre_tokenized_corpus:
    for word, _ in split_text:
        word2count[word] += 1
```

The obtained word2count is as follows:

```python
defaultdict(<class 'int'>, {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1, 'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1, 'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1, 'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1})
```

The next step is to compute the base vocabulary, formed by all the characters used in the corpus:

```python
vocab_set = set()
for word in word2count:
    vocab_set.update(list(word))
vocabs = list(vocab_set)
```

The obtained base vocabulary is as follows:

```python
['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b']
```

We now need to split each word into individual characters, to be able to start training:

```python
word2splits = {word: [c for c in word] for word in word2count}
```

The output is:

```python
'This': ['T', 'h', 'i', 's'], 
'Ġis': ['Ġ', 'i', 's'], 
'Ġthe': ['Ġ', 't', 'h', 'e'], 
...
'Ġand': ['Ġ', 'a', 'n', 'd'], 
'Ġgenerate': ['Ġ', 'g', 'e', 'n', 'e', 'r', 'a', 't', 'e'], 
'Ġtokens': ['Ġ', 't', 'o', 'k', 'e', 'n', 's']
```

Now that we are ready for training, let's write a function that computes the frequency of each pair. We'll need to use this at each step of the training:

```python
def _compute_pair2score(word2splits, word2count):
    pair2count = defaultdict(int)
    for word, word_count in word2count.items():
        split = word2splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair2count[pair] += word_count
    return pair2count
```

The output is 

```python
defaultdict(<class 'int'>, {('T', 'h'): 3, ('h', 'i'): 3, ('i', 's'): 5, ('Ġ', 'i'): 2, ('Ġ', 't'): 7, ('t', 'h'): 3, ..., ('n', 's'): 1})
```

Now, finding the most frequent pair only takes a quick loop:

```python
def _compute_most_score_pair(pair2count):
    best_pair = None
    max_freq = None
    for pair, freq in pair2count.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair
```

After counting, the current pair with the highest frequency is: ('Ġ', 't'), occurring 7 times. We merge ('Ġ', 't') into a single token and add it to the vocabulary. Simultaneously, we add the merge rule ('Ġ', 't') to our list of merge rules.

```python
merge_rules = []
best_pair = compute_most_score_pair(pair2score)
vocabs.append(best_pair[0] + best_pair[1])
merge_rules.append(best_pair)
```

Now the vocabulary is

```python
['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b', 
'Ġt']
```

Based on the updated vocabulary, we re-split word2count. For implementation, we can directly apply the new merge rule ('Ġ', 't') to the existing word2split. This is more efficient than performing a complete re-split, as we only need to apply the latest merge rule to the existing splits.

```python
def _merge_pair(a, b, word2splits):
    new_word2splits = dict()
    for word, split in word2splits.items():
        if len(split) == 1:
            new_word2splits[word] = split
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        new_word2splits[word] = split
    return new_word2splits
```

The new word2split is

```python
{'This': ['T', 'h', 'i', 's'], 
'Ġis': ['Ġ', 'i', 's'], 
'Ġthe': ['Ġt', 'h', 'e'], 
'ĠHugging': ['Ġ', 'H', 'u', 'g', 'g', 'i', 'n', 'g'],
...
'Ġtokens': ['Ġt', 'o', 'k', 'e', 'n', 's']}
```

As we can see, the new word2split now contains the newly merged token "Ġt".
We repeat this iterative process until the vocabulary size reaches our predefined target size.

```python
while len(vocabs) < vocab_size:
    pair2score = compute_pair2score(word2splits, word2count)
    best_pair = compute_most_score_pair(pair2score)
    vocabs.append(best_pair[0] + best_pair[1])
    merge_rules.append(best_pair)
    word2splits = merge_pair(best_pair[0], best_pair[1], word2splits)
```

Let's say our target vocabulary size is 50. After the above iterations, we obtain the following vocabulary and merge rules:

```python
vocabs = ['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b', 'Ġt', 'is', 'er', 'Ġa', 'Ġto', 'en', 'Th', 'This', 'ou', 'se', 'Ġtok', 'Ġtoken', 'nd', 'Ġis', 'Ġth', 'Ġthe', 'in', 'Ġab', 'Ġtokeni', 'Ġtokeniz']

merge_rules = [('Ġ', 't'), ('i', 's'), ('e', 'r'), ('Ġ', 'a'), ('Ġt', 'o'), ('e', 'n'), ('T', 'h'), ('Th', 'is'), ('o', 'u'), ('s', 'e'), ('Ġto', 'k'), ('Ġtok', 'en'), ('n', 'd'), ('Ġ', 'is'), ('Ġt', 'h'), ('Ġth', 'e'), ('i', 'n'), ('Ġa', 'b'), ('Ġtoken', 'i'), ('Ġtokeni', 'z')]
```

Thus, we have completed the training of our BPE tokenizer based on the given corpus.
This trained tokenizer, consisting of the vocabulary and merge rules, can now be used to tokenize new input text using the learned subword patterns.

### BPE's Inference

During the inference phase, given a sentence, we need to split it into a sequence of tokens. The implementation involves two main steps:

First, we pre-tokenize the sentence and split it into character-level sequences

Then, we apply the merge rules sequentially to form larger tokens

```python
def tokenize(text):
    # pre tokenize
    words = [word for word, _ in pre_tokenize_str(text)]
    # split into char level
    splits = [[c for c in word] for word in words]
    # apply merge rules
    for merge_rule in merge_rules:
        for index, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == merge_rule[0] and split[i + 1] == merge_rule[1]:
                    split = split[:i] + ["".join(merge_rule)] + split[i + 2:]
                else:
                    i += 1
            splits[index] = split
    return sum(splits, [])
```

For example:

```python
>>> tokenize("This is not a token.")
>>> ['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']
```