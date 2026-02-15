# News Classification using RNNs & LSTMs

A highly efficient Natural Language Processing (NLP) pipeline that categorizes news articles into distinct topics using Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs).

This project highlights a crucial concept in modern AI engineering: You don't always need a massive LLM. For specific, well-scoped tasks like text categorization, purpose-built Deep Learning models (like LSTMs) can often achieve comparable or better accuracy than generalized LLMs, but at a fraction of the compute cost, latency, and environmental impact.
## Objective

The goal was to build a fast, lightweight, and highly accurate text classifier.

1. The Problem: Processing millions of news articles daily requires a fast categorization system (e.g., Sports, Business, Technology, Politics). Using API-based LLMs for this task at scale is unnecessarily expensive and slow.
2. The Solution: Training a specialized LSTM network that understands the sequential nature of text and context, running locally with minimal hardware requirements.
3. The Takeaway: Specialized, smaller models provide a superior ROI (Return on Investment) for well-defined classification boundaries.

## Key Concepts & Skills

* Sequence Modeling: Using RNNs and LSTMs to process text where word order drastically changes meaning.
* LSTMs vs. RNNs: Overcoming the "vanishing gradient problem" of standard RNNs using LSTM gates (Forget, Input, Output) to retain long-term dependencies in longer news articles.
* Word Embeddings: Translating text into dense vector spaces where semantically similar words are physically closer together.
* Text Preprocessing: Tokenization, sequence padding, and vocabulary management.

## Methodology / Architecture
1. Text Preprocessing

    Tokenization: Breaking down news headlines and body text into integer sequences.
    Padding: Standardizing the length of all news articles so they can be processed in parallel batches.

2. Model Architecture

    Embedding Layer: Maps the integer-encoded vocabulary into dense vectors, learning semantic relationships on the fly.
    Recurrent Layer (LSTM): Processes the sequences word-by-word. It updates its hidden state, deciding what context to "remember" and what to "forget".
    Dense Head: A final fully connected layer with a Softmax activation to output probabilities for each news category.

3. Training & Optimization

    Utilized Categorical Crossentropy for multiclass classification.
    Compared the training time, parameter count, and accuracy of a Simple RNN versus an LSTM.

## Code Highlight

Here is how the lightweight but powerful LSTM architecture is constructed:
```Python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

vocab_size = 10000
embedding_dim = 64
max_length = 120 # Standardized article length

model = Sequential([
    # 1. Learn word semantics
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    
    # 2. Process sequential context (LSTM handles long-term memory)
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    
    # 3. Classify into specific news categories
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
## Results

The model proves that smaller, task-specific networks are incredibly effective.

* Inference Speed: Milliseconds per article (orders of magnitude faster than an LLM API call).
* Accuracy: Reaches high categorical accuracy on the test set, demonstrating that the LSTM perfectly captures the linguistic patterns of different news sectors.
* Compute Cost: Capable of running inference on a standard CPU, requiring zero specialized GPU infrastructure in production.

<img width="1422" height="626" alt="image" src="https://github.com/user-attachments/assets/4933d5b4-7853-434f-9449-6a44884dc0f1" />


## Dependencies

```Python 3.x
    TensorFlow / Keras
    Scikit-Learn (for train/test splits and metrics)
    NLTK / Spacy (for optional text cleaning like stopword removal)
    Pandas & Matplotlib
```

## How to Run

Clone the repository.
Ensure you have the required libraries installed 
```Bash
    pip install tensorflow pandas scikit-learn matplotlib).
```
Load your news dataset (e.g., AG News or BBC News) into the data directory.
Run news-classification-using-rnns-and-lstms.ipynb.

## Future Improvements

    Next Iteration (GRUs): I will be building the exact same pipeline using Gated Recurrent Units (GRUs) to compare performance and training times, as GRUs offer similar memory benefits to LSTMs but with fewer parameters.

    Pre-trained Embeddings: Swap the custom Embedding layer with GloVe or Word2Vec embeddings to give the model a "head start" on understanding language.

    Bidirectional LSTMs: Allow the network to read the text both forwards and backwards for even deeper context.

## References / Credits

Dataset: [Labelled Newscatcher Dataset](https://www.newscatcherapi.com/blog-posts/topic-labeled-news-dataset)
Concept Reference: Understanding LSTM Networks (Chris Olah's Blog)
