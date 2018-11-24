# COMPARE COMPUTATIONAL AND PREDICTIVE PERFORMANCE.
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from naive_bayes import NaiveBayes


class Evaluation:
    def __init__(self, dataset_name):
        # Import dataset.
        self.dataset = pd.read_csv(dataset_name)
        print("Imported dataset '{}'".format(dataset_name))

    # Run stratified ten-fold cross-validation tests.
    def stratified_10f_x_validation(self, tr_dataset, tr_feat, target):
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train, test in skf.split(tr_dataset[tr_feat], tr_dataset[target]):
            print("%s %s" % (train, test))


ev = Evaluation("data/spambase.csv")
nb = NaiveBayes()

