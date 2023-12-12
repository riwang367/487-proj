import joblib
import os
from nltk.tokenize import word_tokenize

# Helpers from process
def fix_length(line):
    words = word_tokenize(line)
    if len(words) > 254:
        words = words[:254]
    else:
        words.extend(["_" for _ in range(254 - len(words))])
    return " ".join(words)

def make_predict(clf, vectorizer, str):
    prediction = clf.predict(vectorizer.transform([fix_length(str)]))
    return prediction

def main():
    # set up files/needed objs
    dir_path = os.path.dirname(os.path.realpath(__file__))
    joblib_file = os.path.join(dir_path, '5000_ffnn.joblib')
    clf = joblib.load(joblib_file)
    vectorizer = joblib.load(os.path.join(dir_path, '5000_vectorizer.joblib'))

    test = "Hail to the victors!"
    prediction = make_predict(clf, vectorizer, test)
    print(f"{test}: {prediction}")

    return prediction


if __name__ == "__main__":
    main()
