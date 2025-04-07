# 📧 Email Spam Detector - PRJ-801

> ML-based spam classifier for emails and messages.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML Project](https://img.shields.io/badge/ML-Spam_Detection-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Objective
Build a machine learning model to classify messages as **Spam** or **Ham (Not Spam)** and enhance messaging security.

---

## 🚀 Features
- Logistic Regression classifier
- TF-IDF feature extraction
- Data cleaning & preprocessing
- Accuracy evaluation
- Portable prediction script

---

## 📂 Folder Overview
| Folder      | Description                             |
|-------------|-----------------------------------------|
| `data/`     | Dataset file (mail_data.csv)            |
| `notebooks/`| Jupyter notebooks for analysis/training |
| `src/`      | Python scripts for training & inference |
| `models/`   | Trained model artifacts                 |

---

## 🧠 ML Workflow
1. Preprocess data (clean text)
2. Apply TF-IDF vectorization
3. Train logistic regression
4. Evaluate model accuracy
5. Save model for prediction

---

## 💡 Applications
- Email spam filters  
- SMS detection on mobile  
- Social media bot/spam blocking  

---

## 🙋‍♂️ Contributors
- Akshant Saini - [211101003]  
- Vaibhav Chhibber - [211101011]  
- Pralin Khaira - [211101027]  

---

## 📚 References
- Stanford CS229: [SMSSpamDetectionUsingML](https://cs229.stanford.edu/proj2013/ShiraniMehr-SMSSpamDetectionUsingMachineLearningApproach.pdf)  
- IEEE: [Spam Detection Using ML](https://ieeexplore.ieee.org/document/10322491)

---

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/email-spam-detector-prj801.git
cd email-spam-detector-prj801
pip install -r requirements.txt
