# K-NEAREST NEIGHBORS SUPERVISED CLASSIFICATION ALGORITHM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNeighbors:
    def __init__(self, k):
        self.data = None
        self.x_train = None
        self.y_train = None
        self.features = None
        self.target = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None

        self.kn = KNeighborsClassifier(n_neighbors=k)

    # Import dataset.
    def load_data(self, dataset_name):
        self.data = pd.read_csv(dataset_name)
        print("Imported dataset '{}'".format(dataset_name))

    # Feature scaling.
    def preprocess(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test

        ss = StandardScaler()
        ss.fit(self.x_train)

        self.x_train = ss.transform(self.x_train)
        self.y_test = ss.transform(self.x_test)

    def train(self, y_train):
        self.y_train = y_train

        self.kn.fit(
            self.x_train,
            self.y_train
        )
        print("Training classifier.")

    def test(self, y_test):
        self.y_test = y_test

        self.y_pred = self.kn.predict(self.x_test)
        accuracy = self.kn.score(
            self.x_test,
            self.y_test
        )
        print("Accuracy is {}".format(accuracy))
        return accuracy


max_acc = 0
max_n = 0
for k in range(1, 50):
    kn = KNeighbors(k)
    kn.load_data("data/spambase.csv")

    features = list(kn.data.drop("spam", axis=1))
    target = "spam"

    x = kn.data[features]
    y = kn.data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    kn.preprocess(x_train, x_test)
    kn.train(y_train)
    acc = kn.test(y_test)
    if acc > max_acc:
        max_acc = acc
        max_n = k
