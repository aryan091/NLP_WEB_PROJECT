import warnings
from flask import Flask, render_template, request
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings('ignore')
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
import joblib
import string
from nltk.corpus import stopwords


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


sen_model = keras.models.load_model('models/best_model2.hdf5')

# model load spam classification
spam_model = joblib.load('models/mymodel2May.pkl')

# news classifier
n_clf = joblib.load('models/exactnewsclassifier.pkl')

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
        max_words = 5000
        max_len = 200
        tokenizer = Tokenizer(num_words=max_words)

        sentiments = ['Neutral', 'Negative', 'Positive']
        sequence = tokenizer.texts_to_sequences([message])
        test = pad_sequences(sequence, maxlen=max_len)
        print("SENTIMENT IS :", sentiments[np.around(sen_model.predict(test), decimals=0).argmax(axis=1)[0]])
        pred = sentiments[np.around(sen_model.predict(test), decimals=0).argmax(axis=1)[0]]
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
        pred = spam_model.predict([message])
        return render_template('spam.html', prediction=pred)


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


# news classifier
@app.route('/newsclf')
def news_classifier():
    return render_template('news.html')


@app.route('/newsclassifier', methods=['POST', 'GET'])
def news_clf():
    if request.method == 'POST':
        message = request.form['message']

        pred = n_clf.predict([message])
        print("pred--", pred)
        return render_template('news.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
