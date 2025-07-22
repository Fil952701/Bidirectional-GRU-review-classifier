# Bidirectional-GRU-review-classifier

A deep learning project for sentiment classification of English text reviews into 5 classes (from 0 to 4), using a GRU-based Recurrent Neural Network with transfer learning and data augmentation.

---

## 🚀 Overview

This project aims to classify textual reviews into one of five sentiment classes using a bidirectional GRU-based architecture trained on a real-world English dataset.

Key features:
- Custom GRU architecture with embedding + dropout + batch norm
- Word2Vec embeddings trained from scratch
- Advanced data augmentation: EDA, back-translation, synonym substitution
- Mixed precision training and XLA acceleration
- Transfer Learning with fine-tuning support

---

## 🧠 Model Architecture

The core model consists of:

- Word2Vec-based Embedding layer (trainable)
- Gaussian Noise + Spatial Dropout
- 2 stacked Bidirectional GRU layers
- Global Average Pooling + Dense Layers
- Softmax output over 5 classes

Full architecture and training logs are available in `results/prestazioni_ottime.txt`.

---

## 📁 Folder Structure

```text
gru-review-classifier/
├── dataset/         # Contains the input dataset (labelled sentences)
├── src/             # Core and fine-tuning scripts
├── results/         # Model performance metrics and logs
└── requirements.txt # Python dependencies
