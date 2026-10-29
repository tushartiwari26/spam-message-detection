# Spam Message Detection Using Machine Learning

## Overview

This project uses machine learning and natural language processing techniques to classify SMS messages as spam or legitimate messages.

It uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Multinomial Naive Bayes
* Streamlit

## Features

* SMS dataset loading and cleaning
* Text feature extraction using TF-IDF
* Supervised machine learning classification
* Training and testing data separation
* Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
* Interactive web application for testing custom messages

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Remove missing values and duplicate records.
3. Convert message labels into numerical values.
4. Split the dataset into training and testing sets.
5. Convert messages into numerical features using TF-IDF.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate the model on unseen test messages.
8. Save the trained pipeline.
9. Launch the Streamlit application.

## Installation

Install the dependencies:

```
pip install -r requirements.txt
```

## Train the Model

```
python main.py
```

This generates the trained model file, spam_model.pkl.

## Launch the Application

```
python -m streamlit run app.py
```

## Dataset

SMS Spam Collection Dataset:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Limitations

* The model can misclassify legitimate messages as spam and spam messages as legitimate.
* Its performance depends on the training dataset.
* New spam techniques and unfamiliar language patterns may reduce performance.
* The dataset may not represent all languages, regions, or modern messaging scams.
* Model confidence should not be interpreted as a guarantee of correctness.

## Disclaimer

This project is intended for educational purposes and demonstrates a basic supervised machine learning workflow.
