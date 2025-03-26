
# 🧠 Stock Movement Prediction from News Headlines

This project explores how financial news headlines can be used to predict the directional movement (up/down) of stock prices. Using natural language processing and machine learning, we attempt to classify daily stock trends based on the top 25 headlines for each trading day.

---

## 📂 Dataset Overview

The dataset includes daily records of the top 25 news headlines from 2000 to 2016. Each row consists of:

- **Date**: Market trading day
- **Label**: Binary indicator (1 = stock up, 0 = stock down)
- **Headline 1–25**: News headlines from that day

---

## 🤖 Model Experiments

### 🧪 Model 1 – Baseline Random Forest

- **Preprocessing**:
  - Removed non-letter characters
  - Lowercased headlines
  - Combined all 25 headlines into one string per day
  - Tokenized and stemmed words using `PorterStemmer`
- **Vectorization**: CountVectorizer (Bigram)
- **Model**: RandomForestClassifier (200 estimators)

**Results**:
- Accuracy: **54.2%**
- Strong bias toward predicting "up" days (high recall for class 1, poor for class 0)

---

### 📊 Model 2 – Logistic Regression with TF-IDF + Sentiment

- **Preprocessing**:
  - Replaced stemming with **lemmatization** (WordNetLemmatizer)
  - Added **TextBlob sentiment polarity** score as an extra feature
- **Vectorization**: TF-IDF (Bigram, max_features=7000, custom stopwords keeping negations)
- **Model**: Logistic Regression (`class_weight='balanced'`)
- **Optimization**: GridSearchCV for hyperparameter tuning

**Results**:
- Accuracy: **56.3%**
- More balanced recall and F1 scores for both classes
- High interpretability and fast inference

---

### 🧠 Model 3 – Voting Ensemble (LogReg + Random Forest + SVM)

- **Features**:
  - TF-IDF + Sentiment Score
  - Lemmatized text
- **Models**:
  - Logistic Regression (tuned)
  - Random Forest (tuned)
  - Linear SVM (tuned)
- **Ensemble**: `VotingClassifier` (hard voting)

**Results**:
- Accuracy: **56.3%**
- Slightly improved F1 for class 0 vs. Model 2
- More robust and generalizable due to ensemble approach

---

## 🚀 Planned Improvements (Model 3)

We aim to take Model 3 further by enhancing both features and model architecture:

### 📊 Feature Engineering
- **VADER sentiment** to better capture financial sentiment
- **Named Entity Recognition** (NER) using spaCy
- **Topic Modeling** (LDA/NMF)
- **Event tagging** for market-moving days

### 🧠 Model Architecture
- **XGBoost / LightGBM** for stronger baselines
- **Soft Voting Classifier** to leverage prediction confidence
- **Transformer Embeddings** (e.g., FinBERT or DistilBERT)

### ⚖️ Imbalance Handling
- Use **SMOTE** or **ADASYN**
- Implement **custom loss functions** like Focal Loss

### 🔍 Optimization & CV
- Switch to **RandomizedSearchCV** or Bayesian Optimization
- Use **TimeSeriesSplit** to maintain temporal integrity

### 🧪 Experiment Tracking
- Integrate `MLflow` or `W&B` for experiment logging and reproducibility

---

## 📈 Outcome Goals

With these enhancements, we aim to:
- Surpass **60% accuracy**
- Improve recall for underrepresented classes
- Create a scalable, production-ready model

---

## 🛠️ Requirements

- Python 3.8+
- Libraries: pandas, nltk, scikit-learn, textblob, matplotlib, scipy, numpy, spacy

---

## 📎 License

MIT License © 2025

---
# Stock-Sentiment-Analysis
This is a mini project to solve a problem statement available on Kaggle. This project uses NLP and ML for stock sentiment analysis using news headlines.
Code inspiration taken from videos made by Krish Naik. Video link - https://www.youtube.com/watch?v=h-LGjJ_oANs&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=12
