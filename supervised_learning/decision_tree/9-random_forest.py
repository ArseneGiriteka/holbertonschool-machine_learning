#!/usr/bin/env python3
"""
This module contains the Random_Forest class, an ensemble of
randomly-split Decision_Tree objects that vote on predictions.
"""
import numpy as np
from scipy import stats

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """This is a random forest class made of several decision trees"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializer method of Random_Forest class"""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict the class of each individual by majority vote among
        the trees of the forest"""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return stats.mode(predictions, axis=0)[0]

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the forest by fitting n_trees random decision trees"""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : \
{self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Return the proportion of correct predictions"""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size
