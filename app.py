from flask import Flask, render_template, request
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
import string
from nltk.corpus import stopwords
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nlpcloud

# model load spam classification
spam_model = joblib.load('models/spammodeljune1.pkl')

# news classifier
n_clf = joblib.load('models/newsjune1.pkl')

app = Flask(__name__)


# home page
@app.route('/')
def index():
    return render_template('home.html')


# sentiment analysis
@app.route('/nlpsentiment')
def sentiment_nlp():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST', 'GET'])
def sentiment():
    if request.method == 'POST':
        message = request.form['message']
        c_sentence = TextBlob(message).correct()

        print(" Corrected sen : ", c_sentence)

        analysisPol = TextBlob(message).polarity

        print(" Analysis Pol : ", analysisPol)
        pred = ""
        if analysisPol < 0.0:
            pred = 'Negative'
        elif analysisPol > 0.0:
            pred = 'Positive'
        else:
            pred = 'Neutral'

        print("Prediction : ", pred)

        return render_template('sentiment.html', prediction=pred)


# spam
@app.route('/nlpspam')
def spam_nlp():
    return render_template('spam.html')


# spam classification
@app.route('/spam', methods=['POST', 'GET'])
def spam():
    if request.method == 'POST':
        message = request.form['message']
        data = pd.read_csv("spam.csv", encoding="latin-1")
        data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        data['class'] = data['class'].map({'ham': 0, 'spam': 1})
        X = data['message']
        y = data['class']
        cv = CountVectorizer()
        X = cv.fit_transform(X)

        data = [message]
        vect = cv.transform(data).toarray()
        pred = spam_model.predict(vect)
        if pred == 0:
            ans = "ham"
        else:
            ans = "spam"
        return render_template('spam.html', prediction=ans)


# summarize
@app.route('/nlpsummarize')
def summarize_nlp():
    return render_template('summarize.html')


@app.route('/summarize', methods=['POST', 'GET'])
def sum_route():
    if request.method == 'POST':
        text = request.form['message']
        client = nlpcloud.Client("bart-large-cnn", "299877c9eee221a6bbbe3f211e4533df1ae3127a")
        summary_text = client.summarization(text)
        summary = summary_text['summary_text']
        return render_template('summarize.html', prediction=summary)


# news classifier
@app.route('/newsclf')
def news_classifier():
    return render_template('news.html')


@app.route('/newsclassifier', methods=['POST', 'GET'])
def news_clf():
    if request.method == 'POST':
        message = request.form['message']
        data = pd.read_csv("exactdata.csv", encoding="latin-1", names=['message', 'class'])
        X = data['message']
        cv = CountVectorizer()
        X = cv.fit_transform(X)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = n_clf.predict(vect)
        if my_prediction == 'b':
            ans = "b"
        elif my_prediction == 't':
            ans = "t"
        elif my_prediction == 'e':
            ans = "e"
        elif my_prediction == 'm':
            ans = "m"
        return render_template('news.html', prediction=ans)


if __name__ == '__main__':
    app.run(debug=True)
