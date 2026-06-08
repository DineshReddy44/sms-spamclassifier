# 📱 SMS Spam Classifier

An end-to-end Machine Learning application that uses Natural Language Processing (NLP) to detect whether a given text message is **Spam** or **Ham (Not Spam)**. 

---

## 🎯 Project Overview
With the rise of mobile communication, smishing (SMS phishing) has become a major issue. This project aims to filter out spam messages by analyzing the text data. The model was trained focusing heavily on **Precision**, as classifying a legitimate message as spam (False Positive) is highly undesirable.

## ⚙️ Tech Stack
* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Natural Language Processing:** NLTK (Natural Language Toolkit)
* **Web Framework:** Streamlit (For the user interface)
* **Deployment:** [Optional: Mention if deployed on Heroku, Render, Streamlit Cloud, etc.]

---

## 🚀 How It Works
The project follows a standard NLP pipeline:
1. **Data Cleaning:** Handling missing values and removing duplicate entries.
2. **Exploratory Data Analysis (EDA):** Analyzing word counts, character counts, and the distribution of Spam vs. Ham.
3. **Text Preprocessing:** 
   * Converting text to lowercase
   * Tokenization
   * Removing special characters, punctuation, and stop words
   * Stemming (reducing words to their root form)
4. **Vectorization:** Converting text into numerical data using **TF-IDF Vectorizer**.
5. **Model Building:** Training various models (Multinomial Naive Bayes, Random Forest, SVM) and selecting the best performer.

---

## 💻 Installation and Setup

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/DineshReddy44/sms-spam-classifier.git](https://github.com/DineshReddy44/sms-spam-classifier.git)
cd sms-spam-classifier
```
## 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Run the Application
streamlit run app.py
