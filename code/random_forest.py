# RANDOM FOREST SUPERVISED CLASSIFICATION ALGORITHM
from time import time

from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.train_time = 0
        self.accuracy = 0
        self.f_measure = 0

        self.rf = RandomForestClassifier(n_estimators=100)

    def train(self, x_train, y_train):
        print("RANDOM FOREST: Training...")
        self.x_train = x_train
        self.y_train = y_train

        t0 = time()
        self.rf.fit(self.x_train, self.y_train)
        t1 = time()
        self.train_time = round(t1 - t0, 3)

    def test(self, x_test, y_test):
        print("RANDOM FOREST: Testing...")
        self.x_test = x_test
        self.y_test = y_test

        self.y_pred = self.rf.predict(self.x_test)
        self.accuracy = self.rf.score(self.x_test, self.y_test)
        print("Accuracy is {}".format(self.accuracy))


'''
dataset = pd.read_csv("data/spambase.csv")
features = list(dataset.drop("spam", axis=1))
target = "spam"
x = rf.data[features]
y = rf.data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
rf = RandomForest()
rf.train(x_train, y_train)
rf.test(x_test, y_test)
'''
