# DECISION TREE SUPERVISED CLASSIFICATION ALGORITHM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self):
        self.data = None
        self.x_train = None
        self.y_train = None
        self.features = None
        self.target = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None

        self.dt = DecisionTreeClassifier(criterion='entropy')

    # Import dataset.
    def load_data(self, dataset_name):
        self.data = pd.read_csv(dataset_name)
        print("Imported dataset '{}'".format(dataset_name))

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        self.dt.fit(
            self.x_train.values,
            self.y_train
        )
        print("Training classifier.")

    def test(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

        self.y_pred = self.dt.predict(self.x_test)
        print("Accuracy is {}".format(self.dt.score(
            self.x_test,
            self.y_test
        )))


dt = DecisionTree()

dt.load_data("data/spambase.csv")

features = list(dt.data.drop("spam", axis=1))
target = "spam"

x = dt.data[features]
y = dt.data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

dt.train(x_train, y_train)
dt.test(x_test, y_test)
