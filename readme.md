# Spam Message Detection Using Machine Learning

## About the Project

This project uses Machine Learning to classify SMS messages as either **Spam** or **Ham (Not Spam)**. It uses the Multinomial Naive Bayes algorithm and TF-IDF vectorization to analyze text messages and predict whether they are spam.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* TF-IDF Vectorization
* Multinomial Naive Bayes

## Dataset

The project uses the SMS Spam Collection dataset, containing 5,572 SMS messages:

* Ham (Not Spam): 4,825 messages
* Spam: 747 messages

## Features

* Classifies messages as Spam or Ham.
* Uses TF-IDF to convert text into numerical features.
* Trains a Multinomial Naive Bayes model.
* Evaluates model performance using classification metrics.
* Provides a simple web interface using Streamlit.

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

This trains the model and saves it as `spam_model.pkl`.

### 3. Launch the Web App

```bash
python -m streamlit run app.py
```

Enter an SMS message into the text box and click **Analyze Message** to see the prediction.

## Project Structure

```text
spam detection/
├── main.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── SMSSpamCollection
```

## Machine Learning Workflow

1. Load and clean the dataset.
2. Convert spam and ham labels into numerical values.
3. Split the data into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train the Multinomial Naive Bayes classifier.
6. Evaluate the model.
7. Use the trained model to predict new messages.

## Future Improvements

* Try additional machine learning algorithms.
* Improve model performance.
* Add more text preprocessing techniques.
* Deploy the application online.

## Author

Tushar Tiwari
