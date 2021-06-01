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



# model load spam classification
spam_model = joblib.load('models/spammodeljune1.pkl')

# news classifier
n_clf = joblib.load('models/newsjune1.pkl')

app = Flask(__name__)


def process(text):
    global punctuation
    stopwords = list(STOP_WORDS)

    nlp = spacy.load('en_core_web_sm')

    # if you get error on this line

    # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz

    doc = nlp(text)
    tokens = [token.text for token in doc]

    punctuation = punctuation + '\n'

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    from heapq import nlargest

    select_length = int(len(sentence_tokens) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    return summary


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
        message = request.form['message']

        sum_message = process(message)
        return render_template('summarize.html', prediction=sum_message)


#news classifier
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
