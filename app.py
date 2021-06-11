from flask import Flask, render_template, request, send_file, current_app, send_from_directory
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
import os

# model load spam classification
from werkzeug.utils import redirect

spam_model = joblib.load('models/spammodeljune1.pkl')

# news classifier
n_clf = joblib.load('models/newsjune1.pkl')

app = Flask(__name__)

app.config["UPLOAD_PATH"] = 'uploads\\'


@app.route("/upload_file_news", methods=["GET", "POST"])
def upload_file_news():
    if request.method == "POST":
        f = request.files['file_name']
        if f.filename == '':
            print('no filename')
            return redirect(request.url)
        f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
        readfilenews(f.filename)
        return render_template("upload_file_NEWS.html", msg="File has been Uploaded Successfully", prediction=True)
    return render_template("upload_file_NEWS.html", msg="Please choose a file")


@app.route("/upload_file_smtanal", methods=["GET", "POST"])
def upload_file_smtanal():
    if request.method == "POST":
        f = request.files['file_name']
        if f.filename == '':
            print('no filename')
            return redirect(request.url)
        f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
        readfilesmt(f.filename)
        return render_template("upload_file_SMT.html", msg="File has been Uploaded Successfully", prediction=True)
    return render_template("upload_file_SMT.html", msg="Please choose a file")


@app.route("/upload_file_se", methods=["GET", "POST"])
def upload_file_se():
    if request.method == "POST":
        f = request.files['file_name']
        if f.filename == '':
            print('no filename')
            return redirect(request.url)
        f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
        readfilese(f.filename)
        return render_template("upload_file_SE.html", prediction=True)
    return render_template("upload_file_SE.html", msg="Please choose a file")


def writefile(file, line):
    f = open('uploads/' + file + ".txt", 'a')
    f.truncate(0)
    f.write(line + "\n")
    f.close()
    return


def readfilenews(file):
    with open("uploads\\" + file, "r+") as script:
        for line in script.readlines():
            data = pd.read_csv("exactdata.csv", encoding="latin-1", names=['message', 'class'])
            X = data['message']
            cv = CountVectorizer()
            X = cv.fit_transform(X)

            data = [line]
            vect = cv.transform(data).toarray()
            my_prediction = n_clf.predict(vect)
            if my_prediction == 'b':
                ans = "b"
                print(current_app.root_path)
                writefile(ans, line)
            elif my_prediction == 't':
                ans = "t"
                writefile(ans, line)
            elif my_prediction == 'e':
                ans = "e"
                writefile(ans, line)
            elif my_prediction == 'm':
                ans = "m"
                writefile(ans, line)

    return render_template('upload_file_NEWS.html', prediction=True)


def readfilesmt(file):
    with open("uploads/" + file, "r+") as script:
        for line in script.readlines():
            c_sentence = TextBlob(line).correct()

            print(" Corrected sen : ", c_sentence)

            analysisPol = TextBlob(line).polarity

            print(" Analysis Pol : ", analysisPol)
            pred = ""
            if analysisPol < 0.0:
                pred = 'Negative'
                writefile("neg", line)
            elif analysisPol > 0.0:
                pred = 'Positive'
                writefile("pos", line)
            else:
                pred = 'Neutral'
                writefile("net", line)

            print("Prediction : ", pred)

    return render_template('upload_file_SMT.html', prediction=True)


def readfilese(file):
    with open("uploads\\" + file, "r+") as script:
        for line in script.readlines():
            data = pd.read_csv("spam.csv", encoding="latin-1")
            data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
            data['class'] = data['class'].map({'ham': 0, 'spam': 1})
            X = data['message']
            y = data['class']
            cv = CountVectorizer()
            X = cv.fit_transform(X)

            data = [line]
            vect = cv.transform(data).toarray()
            pred = spam_model.predict(vect)
            if pred == 0:
                ans = "ham"
                writefile("ham", line)
            else:
                ans = "spam"
                writefile("spam", line)

    return render_template('upload_file_SMT.html', prediction=True)



@app.route('/download_b')
def download_file_b():
    f1 = "uploads/b.txt"
    return send_file(f1, as_attachment=True)


@app.route('/download_t')
def download_file_t():
    f2 = "uploads/t.txt"
    return send_file(f2, as_attachment=True)


@app.route('/download_m')
def download_file_m():
    f3 = "uploads/m.txt"
    return send_file(f3, as_attachment=True)


@app.route('/download_e')
def download_file_e():
    f4 = "uploads/e.txt"
    return send_file(f4, as_attachment=True)


@app.route('/download_p')
def download_file_p():
    f5 = "uploads/pos.txt"
    return send_file(f5, as_attachment=True)


@app.route('/download_n')
def download_file_n():
    f6 = "uploads/neg.txt"
    return send_file(f6, as_attachment=True)


@app.route('/download_net')
def download_file_net():
    f7 = "uploads/net.txt"
    return send_file(f7, as_attachment=True)


@app.route('/download_ham')
def download_file_ham():
    f8 = "uploads/ham.txt"
    return send_file(f8, as_attachment=True)


@app.route('/download_spam')
def download_file_spam():
    f9 = "uploads/spam.txt"
    return send_file(f9, as_attachment=True)


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
