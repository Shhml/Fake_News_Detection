## 📰 Fake News Detection Using Machine Learning
This project aims to detect fake news articles using Natural Language Processing (NLP) and Machine Learning techniques. With the growing concern over misinformation on social media and online platforms, automated fake news detection systems are essential to help combat this issue.

## 📌 Overview
In this project, we:

- Use a dataset of real and fake news articles (from Kaggle).
- Preprocess and clean the text data using NLP tools like NLTK and Scikit-learn.
- Apply TF-IDF vectorization to convert text to numerical features.
- Train multiple classification models such as:

- Logistic Regression
- RandomForestClassifier
- Naive Bayes (MultinomialNB)
- Evaluate the models using accuracy, confusion matrix, precision, recall, and F1-score.


## 🧠 Technologies Used Python 🐍

- Pandas & NumPy – for data handling
- NLTK – for text preprocessing (stopwords, tokenization, etc.)
- Scikit-learn – for ML models, TF-IDF, and evaluation
- Matplotlib & Seaborn – for visualizations

## 🔍 Dataset

- Source: Kaggle Fake News Dataset
- The dataset contains Features like : index, title, text and label

## ⚙️ Project Workflow

 ## Data Loading
- Combine real and fake news data, label them (0 = fake, 1 = real)

## Data Preprocessing

- Lowercasing
- Removing punctuation and stopwords
- Tokenization and lemmatization (optional)
- Applying TF-IDF vectorization
- Model Training
- Split the data into training and test sets
- Train multiple ML models
- Evaluate model performance
- Model Evaluation
- Accuracy Score
- Confusion Matrix

- Conclusion & Insights

Identify the best-performing model

## 📈 Results

## Model	Accuracy

- Logistic Regression	~94%
- Random Forest Performance	~95%
- Multinomial Naive Bayes	~84%
- Random Forest achieved the highest accuracy on the test dataset.
