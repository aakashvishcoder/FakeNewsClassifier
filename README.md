# FakeNewsClassifier

A deep learning model that classifies news articles as Fake or Real using a Bidirectional LSTM (BiLSTM). The model leverages sequential word dependencies to effectively capture context from both past and future tokens in a sentence, making it suitable for fake news detection.

---
## Features

- Preprocessing with tokenization, stopword removal, and padding

- Embedding layer for word representations

- Bidirectional LSTM for capturing contextual information from both directions

- Trained on labeled fake news datasets (e.g., Kaggle’s "Fake News Detection")

- Achieves high accuracy with optimized hyperparameters

--- 

## Tech Stack

- Python 3.8+

- PyTorch (Deep Learning Framework)

- Pandas & NumPy (Data Handling)

--- 

## Installation

git clone https://github.com/aakashvishcoder/fakenews-bilstm.git
cd fakenews-bilstm
