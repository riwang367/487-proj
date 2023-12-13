"""Website index page (run app in this file)"""
import flask
import joblib
import os
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

app = flask.Flask(__name__)  # pylint: disable=invalid-name

class Classifier():
    def __init__(self, clf_file, vect_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.clf = joblib.load(os.path.join(dir_path, clf_file))
        self.vectorizer = joblib.load(os.path.join(dir_path, vect_file))
    
    def make_predict(self, str):
        prediction = self.clf.predict(self.vectorizer.transform([fix_length(str)]))
        return prediction
    

clf = Classifier("./1000_ffnn.joblib", "./1000_ffnn_vectorizer.joblib")

@app.route('/', methods=['GET'])
def show_index():
    context = {
        "fandom": "TODO",
        "fandom_desc": "TODO",
        "resultShowing": False
    }

    return flask.render_template("index.html", **context)

# FIXME: route
@app.route('/', methods=['POST'])
def show_prediction():
    flask.session['text'] = flask.request.form['text']
    fandom = clf.make_predict(text)
    desc = ""
    if fandom == "harrypotter":
        fandom = "Harry Potter"
        desc = "But you already know your Hogwarts house."
    elif fandom == "starwars":
        fandom = "Star Wars"
        desc = "A long time ago, in a galaxy far, far away..."
    elif "leagueoflegends":
        fandom = "League of Legends"
        desc = "Pain is temporary, victory is forever."
    elif "pokemon":
        fandom = "Pokemon"
        desc = "Hello there! Welcome to the world of POKEMON!"
    elif "gameofthrones":
        fandom = "Game of Thrones"
        desc = "Winter is coming... and hopefully a novel with it."
    elif "himym":
        fandom = "How I Met Your Mother"
        desc = "And this, kids, is not how I met your mother."
    elif "mylittlepony":
        fandom = "My Little Pony"
        desc = "Friendship is magic... and so are you!"
    elif "startrek":
        fandom = "Star Trek"
        desc = "Stardate 77390.7, a new ensign joins the crew."

    context = {
        "fandom": fandom,
        "fandom_desc": desc,
        "resultShowing": True
    }
    return flask.render_template("index.html", **context)


# Helpers from process
def fix_length(line):
    words = word_tokenize(line)
    if len(words) > 254:
        words = words[:254]
    else:
        words.extend(["_" for _ in range(254 - len(words))])
    return " ".join(words)