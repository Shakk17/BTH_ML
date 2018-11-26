# COMPARE COMPUTATIONAL AND PREDICTIVE PERFORMANCE.
import math as math
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabulate import tabulate

from decision_tree import DecisionTree
from naive_bayes import NaiveBayes
from random_forest import RandomForest


def rank(array):
    rank_array = [0 for x in range(len(array))]
    for i in range(len(array)):
        (r, s) = (1, 1)
        for j in range(len(array)):
            if j != i and array[j] > array[i]:
                r += 1
            if j != i and array[j] == array[i]:
                s += 1
        rank_array[i] = r + (s - 1) / 2
    return rank_array


# The performance of two classifiers is significantly different if the corresponding average
# ranks differ by at least the critical difference
def reject_null_hyp(value, cd):
    if value > cd:
        print("There is A performance difference between the two algorithms.")
    else:
        print("There is NO performance difference between the two algorithms.")


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

headers = ["Fold", "Naive Bayes", "Decision Tree", "Random Forest"]
table = np.array([list(range(1, 11)),
                  nb_accuracy,
                  dt_accuracy,
                  rf_accuracy
                  ])
stat_table = [
    ["avg", "st_dev"],
    [np.mean(nb_accuracy), np.std(nb_accuracy)],
    [np.mean(dt_accuracy), np.std(dt_accuracy)],
    [np.mean(rf_accuracy), np.std(rf_accuracy)]
]
table = np.append(table, stat_table, axis=1)

# FRIEDMAN TEST
ranking = []
for row in range(10):
    array = np.array([
        nb_accuracy[row],
        dt_accuracy[row],
        rf_accuracy[row]
    ])
    ranking.append(rank(array))

ranking_avg = []
for col in np.transpose(ranking):
    ranking_avg.append(np.average(col))
table = table.T
print(tabulate(table, headers, tablefmt="fancy_grid", floatfmt=".4f", stralign="center"))
print("Average Rank: {}".format(ranking_avg))

# NEMENYI TEST
# Number of algorithms.
k = 3
# Q value for alpha=0.05 and k=3
q = 2.343
# Calculate critical difference.
cd = q * math.sqrt((k*(k+1))/60)
print("Critical difference: {}".format(cd))
print("Difference between Naive Bayes and Decision Tree: {}".format(
    abs(ranking_avg[0]-ranking_avg[1])
))
reject_null_hyp(abs(ranking_avg[0]-ranking_avg[1]), cd)
print("Difference between Decision Tree and Random Forest: {}".format(
    abs(ranking_avg[1]-ranking_avg[2])
))
reject_null_hyp(abs(ranking_avg[1]-ranking_avg[2]), cd)
print("Difference between Naive Bayes and Random Forest: {}".format(
    abs(ranking_avg[0]-ranking_avg[2])
))
reject_null_hyp(abs(ranking_avg[0]-ranking_avg[2]), cd)
