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
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text)

        # Creating a frequency table to keep the
        # score of each word

        freqTable = dict()
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

        # Creating a dictionary to keep the score
        # of each sentence
        sentences = sent_tokenize(text)
        sentenceValue = dict()

        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]

        # Average value of a sentence from the original text

        average = int(sumValues / len(sentenceValue))

        # Storing sentences into our summary.
        summary = ''
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence


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
