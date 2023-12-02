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
        self.get_glove(train)
        # get hyperparameters
        self.cross_val(train)
        # get final classifier
        return self.clf

    def cross_val(self, train):
        learning_rate = [1e-2, 1e-3, 1e-4]
        alpha = [1e-2, 1e-3, 1e-4]
        layers = [2, 4, 8, 16]
        neurons = [64, 128, 256, 528]
        best = {}
        mean = 0
        for num in layers:
            for n in neurons:
                reps = [n * num]
                for perm in itertools.product(learning_rate, alpha):
                    params = {
                        "hidden_layers": reps,
                        "learning_rate": perm[0],
                        "alpha": perm[1]
                    }
                    # X=train.cat, y=train.glove, params
                    self.fit(train['cat'], train['glove'], params)
                    cross_val = cross_val_score(
                        self.clf, train['cat'], train['glove'], cv=5).mean()
                    if cross_val > mean:
                        best = params
                        mean = cross_val
        return best

    def get_glove(self, train):
        train.insert(2, 'glove', np.zeros((1, 200)))
        for i in range(train.shape[0]):
            num_word = 0
            feature = np.zeroes(1, 200)
            for word in word_tokenize(train['text'][i]):
                if word in self.glove:
                    feature = feature + self.glove[word]
                    num_word += 1
            feature = feature / num_word
            train['glove'][i] = feature

    def fit(self, X, y, params):
        """Fit to MLPClassifier."""
        self.clf = MLPClassifier(
            hidden_layer_sizes=params["hidden_layers"],
            learning_rate_init=params["learning_rate"],
            alpha=params["alpha"]
        ).fit(X, y)

    def test(self, test):
        """Test model made."""
        self.get_glove(test)
        prediction = self.clf.predict(test['glove'])
        accuracy = accuracy_score(test['cat'], prediction)
        f1 = f1_score(test['cat'], prediction, average='macro')
        return accuracy, f1


class NB():
    """Naive Bayes classifier."""

    def make(self, train):
        self.vectorizer = TfidfVectorizer()
        self.clf = MultinomialNB()
        self.clf.fit(
            train['cat'], self.vectorizer.fit_transform(train['text']))

    def test(self, test):
        prediction = self.clf.predict(
            self.vectorizer.fit_transform(test['text']))
        accuracy = accuracy_score(test['cat'], prediction)
        f1 = f1_score(test['cat'], prediction, average='macro')
        return accuracy, f1


def main():
    """Do things."""
    train, test = load_data("datasets/final/1000.csv")
    # train multimodal naive bayes
    bayes = NB()
    bayes.make(train)
    accuracy, f1 = bayes.test(test)
    print(f"Naive Bayes: accuracy {accuracy}, f1 {f1}")
    # train multimodal FFNN
    ffnn = FFNN()
    ffnn.make(train)
    accuracy, f1 = ffnn.test(test)
    print(f"FFNN: accuracy {accuracy}, f1 {f1}")
    # evaluation
    return ffnn, bayes

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
