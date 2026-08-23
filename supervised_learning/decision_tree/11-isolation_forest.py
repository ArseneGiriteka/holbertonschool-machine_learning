#!/usr/bin/env python3
"""
This module contains the Isolation_Random_Forest class, an ensemble of
Isolation_Random_Tree objects used to detect outliers.
"""
import numpy as np

Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """This is an isolation random forest class used for outlier
    detection, made of several isolation random trees"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializer method of Isolation_Random_Forest class"""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Return the mean isolation depth of each individual across
        every tree of the forest"""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Train the forest by fitting n_trees random isolation trees"""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """Return the n_suspects rows in explanatory that have the
        smallest mean depth, along with their depths"""
        depths = self.predict(explanatory)
        suspect_indices = np.argsort(depths)[:n_suspects]
        return explanatory[suspect_indices], depths[suspect_indices]
