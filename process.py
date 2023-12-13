import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader
from joblib import dump

import nltk
# nltk.download('punkt')
nltk.download('stopwords')


class FFNN():
    """Feed-forward neural network classifier."""

    def __init__(self):
        self.glove = gensim.downloader.load('glove-wiki-gigaword-200')

    def make(self, train):
        """Make & return model."""
        # get hyperparameters
        # self.cross_val(train)
        self.vectorizer = TfidfVectorizer(max_df=.8, min_df=2)
        X = self.vectorizer.fit_transform(train['fixed'])
        y = train['cat']
        self.fit(X, y, {
            'hidden_layers': [528, 528, 528, 528],
            'learning_rate': 0.001,
            'alpha': 0.001
        })
        return self.clf

    def cross_val(self, train):
        learning_rate = [1e-2, 1e-3]
        alpha = [1e-2, 1e-3]
        layers = [4, 8]
        neurons = [256, 528, 1024]
        best = {}
        mean = 0

        self.vectorizer = TfidfVectorizer(max_df=.8, min_df=2)
        X = self.vectorizer.fit_transform(train['fixed'])
        y = train['cat']

        for num in layers:
            for n in neurons:
                reps = [n for _ in range(num)]
                for perm in itertools.product(learning_rate, alpha):
                    params = {
                        "hidden_layers": reps,
                        "learning_rate": perm[0],
                        "alpha": perm[1]
                    }

                    self.fit(X, y, params)
                    cross_val = cross_val_score(
                        self.clf, X, y, cv=5).mean()
                    if cross_val > mean:
                        best = params
                        mean = cross_val

        print(best)
        self.fit(X, y, best)
        self.best_params = best
        return best

    def get_glove(self, line):
        words = np.array([self.glove[w] for w in word_tokenize(line)])
        return np.mean(words, axis=0)

    def fit(self, X, y, params):
        """Fit to MLPClassifier."""
        self.clf = MLPClassifier(
            hidden_layer_sizes=params["hidden_layers"],
            learning_rate_init=params["learning_rate"],
            alpha=params["alpha"]
        ).fit(X, y)

    def test(self, test):
        """Test model made."""
        prediction = self.clf.predict(self.vectorizer.transform(test['fixed']))
        accuracy = accuracy_score(test['cat'], prediction)
        f1 = f1_score(test['cat'], prediction, average='macro')
        return accuracy, f1

    def make_predict(self, line):
        """Use the FFNN to predict an input."""
        prediction = self.clf.predict(
            self.vectorizer.transform([fix_length(line)]))
        return prediction

    def save_joblib(self):
        """Save classifier and vectorizer for future use."""
        dump(self.clf, "ffnn.joblib")
        dump(self.vectorizer, "vectorizer.joblib")


class NB():
    """Naive Bayes classifier."""

    def make(self, train):
        stops = list(stopwords.words('english'))
        self.vectorizer = CountVectorizer(stop_words=stops)
        X = self.vectorizer.fit_transform(train['fixed'])
        y = train['cat']
        self.clf = MultinomialNB()
        self.clf.fit(X, y)

    def test(self, test):
        X = self.vectorizer.transform(test['fixed'])
        y = test['cat']
        prediction = self.clf.predict(X)
        accuracy = accuracy_score(y, prediction)
        f1 = f1_score(test['cat'], prediction, average='macro')
        return accuracy, f1
    
    def make_predict(self, line):
        """Use the NB classifier to predict an input."""
        prediction = self.clf.predict(
            self.vectorizer.transform([fix_length(line)]))
        return prediction

    def save_joblib(self):
        """Save classifier and vectorizer for future use."""
        dump(self.clf, "bayes.joblib")
        dump(self.vectorizer, "bayes_vectorizer.joblib")


def main():
    """Do things."""
    print("1) Prep")
    dataset = "datasets/final/1000.csv"  # Change to correct dataset
    train, test = load_data(dataset)

    # train multimodal naive bayes
    print("2) Naive Bayes")
    bayes = NB()
    bayes.make(train)
    nb_accuracy, nb_f1 = bayes.test(test)
    print(f"Naive Bayes: accuracy {nb_accuracy}, f1 {nb_f1}")
    bayes.save_joblib()
    print("joblibs saved")
    
    # train multimodal FFNN
    print("3) FFNN")
    ffnn = FFNN()
    ffnn.make(train)
    ffnn_accuracy, ffnn_f1 = ffnn.test(test)
    print(f"FFNN: accuracy {ffnn_accuracy}, f1 {ffnn_f1}")
    ffnn.save_joblib()
    print("joblibs saved")

    # evaluation
    print("4) Evaluation:")
    eval_dataset = load_eval("datasets/final/eval.csv")
    nb_e_accuracy, nb_e_f1 = bayes.test(eval_dataset)
    ffnn_e_accuracy, ffnn_e_f1 = ffnn.test(eval_dataset)
    nb_prediction = bayes.make_predict("hail to the victors")
    ffnn_prediction = ffnn.make_predict("hail to the victors")
    print(f"Hail to the victors: NB = {nb_prediction}, FFNN = {ffnn_prediction}")

    # write to an output file
    with open("result.txt", "a") as file:
        file.write(f"Results for {dataset}--------------\n")
        file.write(f"NB: accuracy {nb_accuracy}, f1 {nb_f1}\n")
        file.write(f"NB Eval: accuracy {nb_e_accuracy}, f1 {nb_e_f1}\n")
        file.write(f"FFNN: accuracy {ffnn_accuracy}, f1 {ffnn_f1}\n")
        file.write(f"FFNN Eval: accuracy {ffnn_e_accuracy}, f1 {ffnn_e_f1}\n\n")

    return

# HELPERS


def load_data(filename):
    random_state = 42
    data = pd.read_csv(filename)
    data['fixed'] = data['text'].apply(fix_length)
    # Split into training & testing data
    [df_train, df_test] = train_test_split(
        data, train_size=0.90, test_size=0.10, random_state=random_state)
    return df_train, df_test


def load_eval(filename):
    data = pd.read_csv(filename)
    data['fixed'] = data['text'].apply(fix_length)
    return data


def fix_length(line):
    words = word_tokenize(line)
    if len(words) > 254:
        words = words[:254]
    else:
        words.extend(["_" for _ in range(254 - len(words))])
    return " ".join(words)


if __name__ == "__main__":
    main()
