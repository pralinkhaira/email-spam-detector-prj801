# 📦 Model Files Overview

This document explains the purpose and difference between the two `.pkl` files used in the **Email Spam Detector** project.

---

## 1. 🧠 `spam_model.pkl` – The Trained Machine Learning Model

- **What it is**: A Logistic Regression model trained on vectorized email/SMS messages.
- **What it does**: Predicts whether a message is **Spam (0)** or **Ham (1)**.
- **Input type**: Expects a numerical feature vector (not raw text).
- **Output**: A binary prediction (0 = spam, 1 = ham).

---

## 2. 🧰 `vectorizer.pkl` – The TF-IDF Vectorizer

- **What it is**: A `TfidfVectorizer` object trained on the same dataset used for the model.
- **What it does**: Converts raw text into a format the model can understand (numerical vectors).
- **Input type**: Raw text messages (strings).
- **Output**: Sparse matrix of TF-IDF features.

---

## 🔁 How They Work Together

1. **Text Preprocessing**:
   - `vectorizer.pkl` transforms raw messages like:
     ```
     "Congratulations! You have won a free ticket."
     ```
     into a numerical feature vector using TF-IDF.

2. **Prediction**:
   - `spam_model.pkl` then takes that vector and predicts:
     ```
     ➤ Spam (0)
     ```

---

## 🧪 Usage Example (Python)

```python
import pickle

# Load model and vectorizer
with open('models/vectorizer.pkl', 'rb') as v_file:
    vectorizer = pickle.load(v_file)

with open('models/spam_model.pkl', 'rb') as m_file:
    model = pickle.load(m_file)

# Predict on a sample message
message = ["You’ve won a free lottery! Call now."]
message_vector = vectorizer.transform(message)
prediction = model.predict(message_vector)

print("Spam" if prediction[0] == 0 else "Ham")
```

---

## 🧼 Notes
- These files are binary and should be **excluded from version control** (`.gitignore`).
- Always keep `vectorizer.pkl` and `spam_model.pkl` **in sync**. Retrain both if dataset changes significantly.
