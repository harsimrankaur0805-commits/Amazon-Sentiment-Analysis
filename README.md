# Amazon Product Review Sentiment Analysis


## Project Overview
This project focuses on analyzing Amazon product reviews and classifying them into different sentiment categories using Machine Learning techniques. The system performs data preprocessing, feature engineering, sentiment prediction, and provides an interactive dashboard using Streamlit.

The main goal of this project is to understand customer opinions from product reviews and automatically predict whether the sentiment of a review is positive, negative, or neutral.

---

## Problem Statement
Online shopping platforms like Amazon receive a huge number of customer reviews every day. Manually analyzing these reviews is difficult and time-consuming. This project solves that problem by building a machine learning model that can automatically classify review sentiments.

---

## Objectives
- Perform data cleaning and preprocessing on Amazon review data
- Handle missing values and duplicate records
- Apply feature engineering techniques for text data
- Convert text reviews into numerical features using TF-IDF
- Train and evaluate machine learning models for sentiment classification
- Save the best performing model and required files
- Build a Streamlit dashboard for user interaction and prediction

---

## Project Structure
```bash
Amazon_Sentiment_Analysis_Project/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── test_data_hidden.csv
│   ├── processed_data.csv
│   ├── X_balanced.pkl
│   └── y_balanced.pkl
│
├── models/
│   ├── best_sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Model_Training_Evaluation.ipynb
│
├── reports/
├── README.md
└── .gitignore