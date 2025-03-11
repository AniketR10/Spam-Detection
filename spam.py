import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
    data = data[['label', 'tweet']]
    data.rename(columns={'tweet': 'message', 'label': 'spam'}, inplace=True)
    data['message'] = data['message'].apply(preprocess_text)
    data['spam'] = data['spam'].map({1: 'spam', 0: 'ham'})
    return data

@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['spam'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer, X_test, y_test

data = load_and_preprocess_data()
model, vectorizer, X_test, y_test = train_model(data)


st.title("Spam Detection App")
user_input = st.text_area("Enter a message to check if it's spam or not:")
if st.button("Predict"):
    input_processed = preprocess_text(user_input)
    input_vectorized = vectorizer.transform([input_processed])
    prediction = model.predict(input_vectorized)[0]
    st.write(f"Prediction: {prediction}")
