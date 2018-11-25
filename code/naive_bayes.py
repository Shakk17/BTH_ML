# NAIVE BAYES SUPERVISED CLASSIFICATION ALGORITHM

from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = 0
        # Instantiate the Gaussian classifier
        self.gnb = GaussianNB()

    # Train classifier on a training dataset.
    def train(self, x_train, y_train):
        print("NAIVE BAYES: Training...")
        self.x_train = x_train
        self.y_train = y_train

        # Train classifier.
        self.gnb.fit(self.x_train, self.y_train)

    # Test classifier on a test dataset.
    def test(self, x_test, y_test):
        print("NAIVE BAYES: Testing...")
        self.x_test = x_test
        self.y_test = y_test

        # Test classifier.
        self.y_pred = self.gnb.predict(self.x_test)
        self.accuracy = 1 - (self.y_test != self.y_pred).sum() / self.x_test.shape[0]
        print("Performance: {}".format(self.accuracy))

    # Print results.
    def print_results(self):
        print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(
            self.x_test.shape[0],
            (self.y_test != self.y_pred).sum(),
            self.accuracy
            )
        )

'''
dataset = pd.read_csv("data/spambase.csv")
features = list(dataset.drop("spam", axis=1))
target = "spam"
x = dataset[features]
y = dataset[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

nb = NaiveBayes()
nb.train(x_train, y_train)
nb.test(x_test, y_test)
nb.print_results()'''

