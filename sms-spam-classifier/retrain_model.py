import pickle
import re
import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'spam.csv'
VECTORIZER_FILE = BASE_DIR / 'vectorizer.pkl'
MODEL_FILE = BASE_DIR / 'model.pkl'

ps = PorterStemmer()


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(ps.stem(word) for word in words)


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'target', 'v2': 'text'})
    df['target'] = df['target'].map({'spam': 1, 'ham': 0})
    df['transformed_text'] = df['text'].apply(transform_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text'])
    y = df['target'].values

    model = MultinomialNB()
    model.fit(X, y)

    accuracy = accuracy_score(y, model.predict(X))
    print(f'Training accuracy: {accuracy:.4f}')

    with VECTORIZER_FILE.open('wb') as f:
        pickle.dump(tfidf, f)
    with MODEL_FILE.open('wb') as f:
        pickle.dump(model, f)

    print('Saved:', VECTORIZER_FILE, MODEL_FILE)
