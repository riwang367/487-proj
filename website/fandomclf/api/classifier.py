"""Run a classifier from a saved joblib"""
import joblib
import os
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

class Classifier():
    def __init__(self, clf_file, vect_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.clf = joblib.load(os.path.join(dir_path, clf_file))
        self.vectorizer = joblib.load(os.path.join(dir_path, vect_file))
    
    def make_predict(self, str):
        prediction = self.clf.predict(self.vectorizer.transform([fix_length(str)]))
        return prediction

def main():
    # set up files/needed objs
    clf = Classifier('./5000_ffnn.joblib', './5000_vectorizer.joblib')

    test = "Hail to the victors!"
    prediction = clf.make_predict(test)
    print(f"{test}: {prediction}")

    return prediction


if __name__ == "__main__":
    main()


# Helpers from process
def fix_length(line):
    words = word_tokenize(line)
    if len(words) > 254:
        words = words[:254]
    else:
        words.extend(["_" for _ in range(254 - len(words))])
    return " ".join(words)