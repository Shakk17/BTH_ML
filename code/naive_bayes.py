# NAIVE BAYES SUPERVISED CLASSIFICATION ALGORITHM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self):
        self.data = None
        self.x_train = None
        self.y_train = None
        self.features = None
        self.target = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        # Instantiate the Gaussian classifier
        self.gnb = GaussianNB()

    # Import dataset.
    def load_data(self, dataset_name):
        self.data = pd.read_csv(dataset_name)
        print("Imported dataset '{}'".format(dataset_name))

    # Train classifier on a training dataset.
    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        # Train classifier.
        self.gnb.fit(
            self.x_train.values,
            self.y_train
        )
        print("Training classifier.")

    # Test classifier on a test dataset.
    def test(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

        # Test classifier.
        self.y_pred = self.gnb.predict(self.x_test)
        print("Testing classifier.")
        performance = 100 * (1 - (self.y_test != self.y_pred).sum() / self.x_test.shape[0])
        print("Performance: {}".format(performance))

    # Print results.
    def print_results(self):
        print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(
            self.x_test.shape[0],
            (self.y_test != self.y_pred).sum(),
            100 * (1 - (self.y_test != self.y_pred).sum() / self.x_test.shape[0])
            )
        )


nb = NaiveBayes()

nb.load_data("data/spambase.csv")

features = list(nb.data.drop("spam", axis=1))
target = "spam"

x = nb.data[features]
y = nb.data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

nb.train(x_train, y_train)
nb.test(x_test, y_test)
nb.print_results()

