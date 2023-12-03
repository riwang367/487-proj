import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from nltk.tokenize import word_tokenize
import gensim.downloader
from tqdm import tqdm

import nltk
nltk.download('punkt')


class FFNN():
    """Feed-forward neural network classifier."""

    def __init__(self):
        self.glove = gensim.downloader.load('glove-wiki-gigaword-200')

    def make(self, train):
        """Make & return model."""
        # turn training set into GloVe vector set
        train['glove'] = train['text'].apply(self.get_glove)
        # get hyperparameters
        self.cross_val(train)
        # get final classifier
        return self.clf

    def cross_val(self, train):
        learning_rate = [1e-2, 1e-3, 1e-4]
        alpha = [1e-2, 1e-3, 1e-4]
        layers = [4]
        neurons = [528]
        best = {}
        mean = 0

        self.vectorizer = TfidfVectorizer(max_df=.8, min_df=2)
        X = self.vectorizer.fit_transform(train['text'])
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
                    # X=train.glove, y=train.cat, params
                    self.fit(X, y, params)
                    cross_val = cross_val_score(
                        self.clf, X, y, cv=5).mean()
                    if cross_val > mean:
                        best = params
                        mean = cross_val
        print(best)
        self.fit(X, y, best)
        return best

    def get_glove(self, line):
        num_word = 0
        feature = np.zeros((1, 200))
        for word in word_tokenize(line):
            if word in self.glove:
                feature = feature + self.glove[word]
                num_word += 1

        # Avoid nan
        if num_word == 0:
            feature = np.zeros((1, 200))
        else:
            feature = feature / num_word
        return feature

    def fit(self, X, y, params):
        """Fit to MLPClassifier."""
        self.clf = MLPClassifier(
            hidden_layer_sizes=params["hidden_layers"],
            learning_rate_init=params["learning_rate"],
            alpha=params["alpha"]
        ).fit(X, y)

    def test(self, test):
        """Test model made."""
        test['glove'] = test['text'].apply(self.get_glove)
        # prediction = self.clf.predict(test['glove'])
        prediction = self.clf.predict(self.vectorizer.fit_transform(test['text']))
        accuracy = accuracy_score(test['cat'], prediction)
        f1 = f1_score(test['cat'], prediction, average='macro')
        return accuracy, f1


class NB():
    """Naive Bayes classifier."""
    # tokenizing?

    def enum_col(self, data):
        array = np.array([[e] for e in data['enum']])
        return array

    def make(self, train):
        self.vectorizer = TfidfVectorizer()
        self.clf = MultinomialNB()
        self.clf.fit(
            self.vectorizer.fit_transform(train['text']))

    def test(self, test):
        prediction = self.clf.predict(
            self.vectorizer.fit_transform(test['text']))
        accuracy = accuracy_score(test['cat'], prediction)
        f1 = f1_score(test['cat'], prediction, average='macro')
        return accuracy, f1


def main():
    """Do things."""
    print("1) Prep")
    train, test = load_data("datasets/final/debug.csv")
    # train multimodal naive bayes
    # print("2) Naive Bayes")
    # bayes = NB()
    # bayes.make(train)
    # accuracy, f1 = bayes.test(test)
    # print(f"Naive Bayes: accuracy {accuracy}, f1 {f1}")
    # train multimodal FFNN
    print("3) FFNN")
    ffnn = FFNN()
    ffnn.make(train)
    accuracy, f1 = ffnn.test(test)
    print(f"FFNN: accuracy {accuracy}, f1 {f1}")
    # evaluation
    return ffnn#, bayes

# HELPERS


def load_data(filename):
    random_state = 42
    data = pd.read_csv(filename)
    # Split into training & testing data
    [df_train, df_test] = train_test_split(
        data, train_size=0.90, test_size=0.10, random_state=random_state)
    return df_train, df_test


if __name__ == "__main__":
    main()
