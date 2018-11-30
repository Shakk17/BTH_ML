# COMPARE COMPUTATIONAL AND PREDICTIVE PERFORMANCE.
import math as math
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import friedmanchisquare
from prettytable import PrettyTable

from decision_tree import DecisionTree
from naive_bayes import NaiveBayes
from random_forest import RandomForest


def rank(array, type):
    rank_array = [0 for x in range(len(array))]
    for i in range(len(array)):
        (r, s) = (1, 1)
        for j in range(len(array)):
            if j != i and array[j] > array[i]:
                r += 1
            if j != i and array[j] == array[i]:
                s += 1
        rank_array[i] = r + (s - 1) / 2
    if type == "TRAINING TIME":
        return rank_array[::-1]
    return rank_array


# The performance of two classifiers is significantly different if the corresponding average
# ranks differ by at least the critical difference
def reject_null_hyp(value, cd):
    if value > cd:
        print("There is A performance difference between the two algorithms.")
    else:
        print("There is NO performance difference between the two algorithms.")


def print_table(table, headers, rank):
    p_table = PrettyTable(headers)
    fold = 1
    for t_row in table:
        if rank is not None:
            i = 0
            t_rank = []
            for j in range(3):
                t_rank.append("{} ({})".format(t_row[j+1], rank[i][j]))
            i += 1
            t_rank.insert(0, str(fold))
            p_table.add_row(t_rank)
        else:
            p_table.add_row(t_row)
        fold += 1
    print(p_table)


def transpose(table):
    return [[table[j][i] for j in range(len(table))] for i in range(len(table[0]))]


dataset = pd.read_csv("data/spambase.csv")
features = list(dataset.drop("spam", axis=1))
target = "spam"
x = dataset[features]
y = dataset[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

nb = NaiveBayes()
nb_dict = {"TRAINING TIME": [], "ACCURACY": [], "F-MEASURES": []}
dt = DecisionTree()
dt_dict = {"TRAINING TIME": [], "ACCURACY": [], "F-MEASURES": []}
rf = RandomForest()
rf_dict = {"TRAINING TIME": [], "ACCURACY": [], "F-MEASURES": []}

skf = StratifiedKFold(n_splits=10, shuffle=True)
i_fold = 0
for train_index, test_index in skf.split(x_train, y_train):
    i_fold += 1
    print("++ FOLD {} ++".format(i_fold))
    # NAIVE BAYES
    nb.train(x_train.values[train_index], y_train.values[train_index])
    nb.test(x_train.values[test_index], y_train.values[test_index])
    nb_dict["TRAINING TIME"].append(nb.train_time)
    nb_dict["ACCURACY"].append(nb.accuracy)
    nb_dict["F-MEASURES"].append(nb.f_measure)
    # DECISION TREE
    dt.train(x_train.values[train_index], y_train.values[train_index])
    dt.test(x_train.values[test_index], y_train.values[test_index])
    dt_dict["TRAINING TIME"].append(dt.train_time)
    dt_dict["ACCURACY"].append(dt.accuracy)
    dt_dict["F-MEASURES"].append(dt.f_measure)
    # RANDOM FOREST
    rf.train(x_train.values[train_index], y_train.values[train_index])
    rf.test(x_train.values[test_index], y_train.values[test_index])
    rf_dict["TRAINING TIME"].append(rf.train_time)
    rf_dict["ACCURACY"].append(rf.accuracy)
    rf_dict["F-MEASURES"].append(rf.f_measure)

# Print statistics.
headers = ["Fold", "Naive Bayes", "Decision Tree", "Random Forest"]
table_types = ["TRAINING TIME", "ACCURACY", "F-MEASURES"]
ranking_avg = {}
# Table containing training times for each fold, for each algorithm.
for type in table_types:
    print("\n++ {} ++".format(type))
    # FOLDS
    table = [list(range(1, 11)),
             nb_dict[type],
             dt_dict[type],
             rf_dict[type]]
    # Ranking of results for each fold.
    ranking = []
    for row in range(10):
        array = np.array([
            nb_dict[type][row],
            dt_dict[type][row],
            rf_dict[type][row]])
        ranking.append(rank(array, type))
    print_table(transpose(table), headers, ranking)
    # STATISTICS
    stat_table = [
        ["avg", "std"],
        [np.around(np.mean(nb_dict[type]), 4), np.around(np.std(nb_dict[type]), 4)],
        [np.around(np.mean(dt_dict[type]), 4), np.around(np.std(dt_dict[type]), 4)],
        [np.around(np.mean(rf_dict[type]), 4), np.around(np.std(rf_dict[type]), 4)]]
    stat_table = transpose(stat_table)
    # FRIEDMAN TEST
    ranking_avg = [np.average(col) for col in np.transpose(ranking)]
    ranking_avg.insert(0, "rank")
    stat_table.append(ranking_avg)
    print_table(stat_table, headers, None)

    # NEMENYI TEST
    # Number of algorithms.
    k = 3
    # Q value for alpha=0.05 and k=3
    q = 2.343
    # Friedman test.
    stat, p = friedmanchisquare(nb_dict[type], dt_dict[type], rf_dict[type])
    alpha = 0.05
    if p > alpha:
        print('FRIEDMAN TEST: Same distributions (fail to reject H0)')
    else:
        print('FRIEDMAN TEST: Different distributions (reject H0)')
    # Calculate critical difference.
    cd = round(q * math.sqrt((k * (k + 1)) / 60), 4)
    print("\nNEMENYI TEST: Critical difference: {}".format(cd))
    print("Difference between Naive Bayes and Decision Tree: {}".format(
        abs(ranking_avg[1] - ranking_avg[2])
    ))
    reject_null_hyp(abs(ranking_avg[1] - ranking_avg[2]), cd)
    print("Difference between Decision Tree and Random Forest: {}".format(
        abs(ranking_avg[2] - ranking_avg[3])
    ))
    reject_null_hyp(abs(ranking_avg[2] - ranking_avg[3]), cd)
    print("Difference between Naive Bayes and Random Forest: {}".format(
        abs(ranking_avg[1] - ranking_avg[3])
    ))
    reject_null_hyp(abs(ranking_avg[1] - ranking_avg[3]), cd)
