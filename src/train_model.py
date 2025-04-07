# train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data_path = os.path.join("data", "mail_data.csv")
df = pd.read_csv(data_path)

# Replace null values with empty strings
df = df.where((pd.notnull(df)), '')

# Label encoding: spam = 0, ham = 1
df.loc[df['Category'] == 'spam', 'Category'] = 0
df.loc[df['Category'] == 'ham', 'Category'] = 1

# Features and labels
X = df['Message']
y = df['Category'].astype('int')

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, stratify=y, random_state=42
)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Ensure models directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save model
with open(os.path.join(model_dir, "spam_model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully to 'models/' folder.")
