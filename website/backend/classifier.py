import pickle
import pandas as pd


def main():
    clf = pd.read_pickle(r'ffnn_clf.pkl')

    prediction = predict("Hail to the victors!", clf)
    print(prediction)

    return prediction


def predict(input, clf):
    return clf.predict(input)


if __name__ == "__main__":
    main()
