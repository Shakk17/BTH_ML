# COMPARE COMPUTATIONAL AND PREDICTIVE PERFORMANCE.
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from decision_tree import DecisionTree
from naive_bayes import NaiveBayes
from random_forest import RandomForest

dataset = pd.read_csv("data/spambase.csv")
features = list(dataset.drop("spam", axis=1))
target = "spam"
x = dataset[features]
y = dataset[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

nb = NaiveBayes()
nb_accuracy = []
dt = DecisionTree()
dt_accuracy = []
rf = RandomForest()
rf_accuracy = []

skf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in skf.split(x_train, y_train):
    # NAIVE BAYES
    nb.train(x_train.values[train_index], y_train.values[train_index])
    nb.test(x_train.values[test_index], y_train.values[test_index])
    nb_accuracy.append(nb.accuracy)
    nb.print_results()
    # DECISION TREE
    dt.train(x_train.values[train_index], y_train.values[train_index])
    dt.test(x_train.values[test_index], y_train.values[test_index])
    dt_accuracy.append(dt.accuracy)
    # RANDOM FOREST
    rf.train(x_train.values[train_index], y_train.values[train_index])
    rf.test(x_train.values[test_index], y_train.values[test_index])
    rf_accuracy.append(rf.accuracy)
    
print("NB accuracy avg: {:05.4f}, st_dev: {:05.4f}".format(
    np.mean(nb_accuracy),
    np.std(nb_accuracy)
))
print("DT accuracy avg: {:05.4f}, st_dev: {:05.4f}".format(
    np.mean(dt_accuracy),
    np.std(dt_accuracy)
))
print("RF accuracy avg: {:05.4f}, st_dev: {:05.4f}".format(
    np.mean(rf_accuracy),
    np.std(rf_accuracy)
))
