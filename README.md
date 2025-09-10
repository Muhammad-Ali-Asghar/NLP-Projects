# NLP Projects

This repository contains four Natural Language Processing (NLP) projects, each focusing on a specific application of NLP techniques. Below is an overview of each project:

---

## 1. Fake News Detector

### Description
This project aims to classify news articles as either "Fake" or "True" using machine learning techniques. The dataset includes labeled examples of fake and true news articles.

### Dataset
- **Fake.csv**: Contains examples of fake news articles.
- **True.csv**: Contains examples of true news articles.

### Key Features
- Data preprocessing and cleaning.
- Feature extraction using NLP techniques.
- Model training and evaluation.

---

## 2. Sentiment Analysis

### Description
This project focuses on analyzing the sentiment of user reviews. The goal is to classify reviews as positive, negative, or neutral.

### Dataset
- **Reviews.csv**: Contains user reviews along with their sentiment labels.

### Key Features
- Text preprocessing and tokenization.
- Sentiment classification using machine learning models.
- Evaluation of model performance.

---

## 3. Topic Modeling

### Description
This project performs topic modeling on a collection of news articles to identify the underlying topics discussed in the dataset.

### Dataset
- **bbc_news.csv**: A dataset containing news articles from various categories.

### Key Features
- Data preprocessing and vectorization.
- Topic extraction using Latent Dirichlet Allocation (LDA).
- Visualization of topics.

### Script
- **topic_modeling_news.py**: The main script for topic modeling.

---

## 4. News Category Classification

### Description
This project focuses on classifying news articles into predefined categories using machine learning models. The goal is to automate and enhance the process of news categorization.

### Dataset
- **news_category.csv**: A dataset containing news articles labeled with their respective categories.

### Key Features
- Data preprocessing and feature engineering.
- Model training and evaluation using multiple algorithms.
- Comparative analysis of model performance.

### Results
Our machine learning models achieved the following accuracy scores:
- **Logistic Regression**: 88.08%
- **Random Forest**: 86.55%
- **Support Vector Machine (SVM)**: 89.11%

These results highlight the effectiveness of machine learning in automating news categorization.

---

## 5. Additional Notes

Each project is self-contained within its respective folder. To get started with any project, navigate to its folder and follow the instructions provided in the scripts or datasets.

---

## Requirements

To run these projects, ensure you have the following installed:
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `nltk`, `gensim`

Install the required libraries using:
```bash
pip install -r requirements.txt

```

---

## Author

Developed by [Your Name].
