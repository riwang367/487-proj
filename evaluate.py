import pickle
import joblib
import csv


def main():
    # with open("website/backend/ffnn_clf.pkl", "rb") as pickle_file:
    #     clf = pickle.load(pickle_file)

    clf = joblib.load('website/backend/ffnn_clf.pkl')

    with open("eval.csv", "r") as file:
        for line in csv.reader(file):
            _, cat, text = line
            prediction = predict(text, clf)

    # print(f"Accuracy {accuracy}, f1 {f1}")


def predict(input, clf):
    return clf.predict(input)


if __name__ == "__main__":
    main()
