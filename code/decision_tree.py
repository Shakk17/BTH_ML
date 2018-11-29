# DECISION TREE SUPERVISED CLASSIFICATION ALGORITHM
from time import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


class DecisionTree:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.train_time = 0
        self.accuracy = 0
        self.f_measure = 0

        self.dt = DecisionTreeClassifier(criterion='entropy')

    def train(self, x_train, y_train):
        print("DECISION TREE: Training...")
        self.x_train = x_train
        self.y_train = y_train

        t0 = time()
        self.dt.fit(self.x_train, self.y_train)
        t1 = time()
        self.train_time = round(t1 - t0, 3)

    def test(self, x_test, y_test):
        print("DECISION TREE: Testing...")
        self.x_test = x_test
        self.y_test = y_test

        self.y_pred = self.dt.predict(self.x_test)
        self.accuracy = round(accuracy_score(self.y_test, self.y_pred), 4)
        self.f_measure = round(f1_score(self.y_test, self.y_pred), 4)

    # Print results.
    def print_results(self):
        print("Training time: {} sec".format(self.train_time))
        print("Accuracy: {}".format(self.accuracy))
        print("F-measure: {}".format(self.f_measure))


'''
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data/spambase.csv")
features = list(dataset.drop("spam", axis=1))
target = "spam"
x = dataset[features]
y = dataset[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
dt = DecisionTree()
dt.train(x_train, y_train)
dt.test(x_test, y_test)
dt.print_results()
'''

