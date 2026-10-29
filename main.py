
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("=" * 50)
print("       SPAM MESSAGE DETECTION SYSTEM")
print("=" * 50)

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("\nDataset loaded successfully!")
print("Total messages:", len(data))

# ==========================================
# 2. CLEAN DATA
# ==========================================

data = data.dropna()
data = data.drop_duplicates()

# Convert labels into numbers:
# ham = 0, spam = 1

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nMessages after cleaning:", len(data))
print("\nLabel distribution:")
print(data["label"].value_counts())

# ==========================================
# 3. SPLIT DATA
# ==========================================

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# ==========================================
# 4. BUILD MACHINE LEARNING PIPELINE
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model trained successfully!")

# ==========================================
# 6. EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Ham (Not Spam)", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as spam_model.pkl")

# ==========================================
# 8. TEST CUSTOM MESSAGES
# ==========================================

test_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hey bro, are we going to the gym today?",
    "URGENT! You have won cash. Call now!",
    "I will meet you at college tomorrow."
]

print("\n" + "=" * 50)
print("CUSTOM MESSAGE PREDICTIONS")
print("=" * 50)

for message in test_messages:

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    confidence = probabilities[prediction] * 100

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print(f"\nMessage: {message}")
    print(f"Prediction: {result}")
    print(f"Model confidence: {confidence:.2f}%")

print("\nProject completed successfully!")

