# Sentiment Analysis Pipeline with GCP and PySpark

A cloud-based NLP sentiment classification pipeline built with **PySpark**, **Google Cloud Storage**, **Word2Vec**, and **Random Forest**.

This project processes Amazon product review text, converts review content into numerical word embeddings, and trains a Spark ML classification model to predict review sentiment.

---

## Overview
The goal was to understand how large-scale text data can be processed using Spark and how NLP features can be used inside a machine learning pipeline.

The pipeline loads Amazon review data from Google Cloud Storage, preprocesses the text, vectorizes it using Word2Vec, trains a Random Forest classifier, evaluates the model using F1 score, and saves the trained model artifacts back to cloud storage.

---

## Project Workflow

```text
Amazon Reviews CSV
        ↓
Google Cloud Storage
        ↓
PySpark DataFrame
        ↓
Text Cleaning + Punctuation Handling
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Word2Vec Feature Generation
        ↓
Random Forest Classification
        ↓
F1 Score Evaluation
        ↓
Model Saved to GCS
```

---

## Features

* Loaded Amazon review data from **Google Cloud Storage**
* Processed review text using **PySpark DataFrames**
* Cleaned text by handling missing values and punctuation
* Tokenized reviews into word-level features
* Removed English stopwords using NLTK
* Converted cleaned text into numerical vectors using **Word2Vec**
* Trained a **Random Forest Classifier** using Spark MLlib
* Evaluated predictions using **F1 score**
* Saved trained model artifacts back to cloud storage

---

## Tech Stack

* **Python**
* **PySpark**
* **Spark MLlib**
* **Google Cloud Platform**
* **Google Cloud Storage**
* **NLTK**
* **Word2Vec**
* **Random Forest Classifier**

---

## Repository Structure

```text
Sentiment-Analysis/
│
├── Sentiment_Analysis_GCP_PySpark_Framework.ipynb
├── README.md
└── images/
    └── sentiment_analysis_thumbnail.png
```

Recommended future structure:

```text
Sentiment-Analysis/
│
├── README.md
├── requirements.txt
├── data/
│   └── sample_reviews.csv
├── notebooks/
│   └── Sentiment_Analysis_GCP_PySpark_Framework.ipynb
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── predict.py
└── models/
    └── .gitkeep
```

---

## Dataset

The project uses Amazon review data stored in Google Cloud Storage.

Expected columns include:

```text
Review
Polarity
```

The `Review` column contains the raw product review text.
The `Polarity` column is used as the sentiment label.

---

## Model Pipeline

The machine learning pipeline uses the following Spark ML stages:

1. **Text preprocessing**

   * Rename columns
   * Drop null review or label values
   * Remove punctuation
   * Tokenize review text
   * Remove stopwords

2. **Feature engineering**

   * Convert cleaned tokens into dense vectors using Word2Vec

3. **Classification**

   * Train a Random Forest classifier on the generated Word2Vec features

4. **Evaluation**

   * Evaluate the model using F1 score

5. **Model persistence**

   * Save the trained model locally
   * Upload model artifacts to Google Cloud Storage

---

## Key Code Components

### Word2Vec Vectorization

```python
word2Vec = Word2Vec(
    inputCol="cleaned_text",
    outputCol="features",
    vectorSize=100,
    minCount=1
)
```

### Random Forest Classifier

```python
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=20
)
```

### Spark ML Pipeline

```python
pipeline = Pipeline(stages=[word2Vec, rf])
```

### Model Evaluation

```python
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

word2vec_rf_f1_score = evaluator.evaluate(word2vec_rf_predictions)
```

---

## Planned improvements

Planned improvements to make this project - 

* Add a small sample dataset for local testing
* Add a `requirements.txt` file
* Convert notebook logic into reusable Python scripts
* Add local execution mode without requiring GCP
* Add optional GCP mode for cloud-based training
* Add a `predict.py` script for testing custom review text
* Add model comparison with Logistic Regression, Naive Bayes, or Gradient-Boosted Trees
* Add hyperparameter tuning
* Add a simple FastAPI endpoint for sentiment prediction
* Add Docker support for easier setup

