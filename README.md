# 🧠 Twitter Sentiment Analysis with NLTK

This project is a **Sentiment Analysis classifier** built using **NLTK**, the Natural Language Toolkit for Python. It focuses on classifying tweets as either **Positive** or **Negative** by applying a complete NLP pipeline, from data cleaning to model training and testing.

---

## 📦 Project Overview

- ✅ Dataset: [NLTK's twitter_samples](https://www.nltk.org/nltk_data/)
- ✅ Techniques used: Tokenization, Noise Removal, Lemmatization, Stopword Filtering
- ✅ Model: Naive Bayes Classifier
- ✅ Evaluation: Word frequency, model accuracy, most informative features
- ✅ Extras: Live sentiment testing with custom input

---

## 🔧 Stack & Tools

- Python 3.x
- NLTK
- Regex & String Manipulation
- Naive Bayes Classifier (NLTK)
- WordNetLemmatizer
- Frequency Distributions (FreqDist)

---

## 🧪 Pipeline Steps

1. **Data Import** – Using NLTK's built-in Twitter dataset
2. **Tokenization** – Splitting tweets into word-level tokens
3. **Text Cleaning** – Removing URLs, mentions, special characters
4. **Lemmatization** – Reducing words to their base form
5. **Stopword Removal** – Eliminating common non-informative words
6. **Feature Extraction** – Converting tokens into feature dictionaries
7. **Model Training** – Naive Bayes classifier on labeled tweet sets
8. **Evaluation** – Accuracy and top predictive features
9. **Prediction Demo** – Real-time test on custom tweet input

