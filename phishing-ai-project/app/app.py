# app.py
#
# Authors: Connor Fletcher, Frank Sosta
#
# Purpose:  Simple Flask web app that loads your trained phishing detector
#           and lets the user enter a URL to classify as "Phishing" or "Legitimate".

from flask import Flask, render_template, request
import pickle

# load in the trained model
# laod in the TF-IDF vectorizer
model = pickle.load(open("C:/Users/confl/OneDrive/Desktop/phishing-ai-project/model.pkl", "rb"))
vectorizer = pickle.load(open("C:/Users/confl/OneDrive/Desktop/phishing-ai-project/vectorizer.pkl", "rb"))

"""
needed to hardcode the full file paths because the program
was only able to find the .pkl files this way
(idk why either, but this fixed the issue so we roll with it)
"""

app = Flask(__name__)  # start the Flask web app


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # grab the URL that the user types
        url = request.form.get("url")

        # converts the URL into a TF-IDF vector
        url_tfidf = vectorizer.transform([url])

        # then predicts if the URL is a 0 or a 1
        pred = model.predict(url_tfidf)[0]

        """
        originally these labels were flipped (1 meant good, 0 meant bad)
        so the output was backwards
        swapping them here made the predictions match their intended
        """

        # if the model thinks it's phishing
        if pred == 0:
            prediction = "⚠️ Phishing (Unsafe)"
        else:
            # else the URL is legitmate
            prediction = "✅ Legitimate (Safe)"


    # send the prediction back so the user can see the result
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
