# COMPARE COMPUTATIONAL AND PREDICTIVE PERFORMANCE.
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from naive_bayes import NaiveBayes

dataset = pd.read_csv("data/spambase.csv")
features = list(dataset.drop("spam", axis=1))
target = "spam"
x = dataset[features]
y = dataset[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

nb = NaiveBayes()

skf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in skf.split(x_train, y_train):
    nb.train(x_train.values[train_index], y_train.values[train_index])
    nb.test(x_train.values[test_index], y_train.values[test_index])
    nb.print_results()

