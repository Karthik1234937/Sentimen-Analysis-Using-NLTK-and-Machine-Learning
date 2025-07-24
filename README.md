# Sentimen-Analysis-Using-NLTK-and-Machine-Learning
A Python-based sentiment analysis model that classifies tweets or text data as Positive, Negative, Neutral, or Irrelevant. It uses NLTK for preprocessing, TF-IDF for feature extraction, and scikit-learn models for classification.


## 📌 Features

- 🔤 Preprocesses raw text using **NLTK**:
  - Tokenization
  - Stopword Removal
  - Lemmatization
  - Punctuation and case normalization
- 📊 TF-IDF Vectorization for converting text to numerical features
- 🤖 Trains and evaluates the following models:
  - SVM
  - Multinomial Naive Bayes
  - Random Forest Classifier
- 📈 Evaluation Metrics:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
- 🌥️ WordClouds for each sentiment category
- 🧪 Custom test cases for live prediction

---

## 📁 Dataset

- Dataset used: **Twitter Training Data**
- Format: CSV with columns for ID, Entity, Sentiment, and Tweet Text
- Each row is labeled with one of the four sentiments:
  - `Positive`
  - `Negative`
  - `Neutral`
  - `Irrelevant`

---

## 🔧 Tech Stack

- **Python 3.8+**
- **NLTK** – for natural language processing
- **scikit-learn** – for machine learning models and metrics
- **Matplotlib / Seaborn** – for visualizations
- **WordCloud** – for sentiment-based word clouds

---


