# predict.py
import pickle
import sys

# Load model and vectorizer
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_spam(message):
    data = [message]
    transformed = vectorizer.transform(data)
    prediction = model.predict(transformed)
    return "Ham (Not Spam)" if prediction[0] == 1 else "Spam"

if __name__ == "__main__":
    message = sys.argv[1]
    result = predict_spam(message)
    print(f"Prediction: {result}")