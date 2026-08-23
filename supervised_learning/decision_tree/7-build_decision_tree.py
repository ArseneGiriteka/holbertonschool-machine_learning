#!/usr/bin/env python3
"""
This module contains a tree major class of decision trees:
- Node
- Leaf
- Decision_tree
"""
import numpy as np
import math


class Node:
    """This is a node class despit it is a decision of leaf node"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """This is a node class initializer method"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes"""
        if only_leaves:
            return self.left_child.count_nodes_below(only_leaves=only_leaves)\
                + self.right_child.count_nodes_below(only_leaves=only_leaves)
        else:
            return 1 +\
                self.left_child.count_nodes_below(only_leaves=only_leaves)\
                + self.right_child.count_nodes_below(only_leaves=only_leaves)

    def max_depth_below(self):
        """This method compute the maximum depth a decision tree can
        reach"""
        if self.is_leaf:
            return self.depth

        depth = self.left_child.max_depth_below()

        if depth < self.right_child.max_depth_below():
            depth = self.right_child.max_depth_below()

        return depth

    def get_leaves_below(self):
        """Will get all leaves below this node in a list"""
        return self.left_child.get_leaves_below() +\
            self.right_child.get_leaves_below()

    def left_child_add_prefix(self, text):
        """This method print the left child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """This prints the right child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def update_bounds_below(self):
        """Update"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        for child, is_left\
                in [(self.left_child, True), (self.right_child, False)]:
            if is_left:
                child.lower[self.feature] = max(
                    child.lower.get(self.feature, -np.inf), self.threshold)
            else:
                child.upper[self.feature] = min(
                    child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """This method will update the indicator function of this node"""
        def is_large_enough(x):
            return np.all(
                np.array([x[:, key] > self.lower[key]
                          for key in self.lower.keys()]),
                axis=0)

        def is_small_enough(x):
            return np.all(
                np.array([x[:, key] <= self.upper[key]
                          for key in self.upper.keys()]),
                axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Make a prediction for a single sample"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def __str__(self):
        """This method print out the state of Node"""
        lines = ""
        if self.is_leaf:
            lines += "    " + "+--->" + f"leaf [value={self.value}]"
            return lines
        elif self.is_root:
            lines = f"root [feature={self.feature}, " +\
                f"threshold={self.threshold}]\n"
        else:
            lines += f"-> node [feature={self.feature}, " +\
                f"threshold={self.threshold}]\n"

        lines += self.left_child.left_child_add_prefix(
            self.left_child.__str__())
        lines += self.right_child.right_child_add_prefix(
            self.right_child.__str__())

        return lines


class Leaf(Node):
    """This is a Leaf node class"""
    def __init__(self, value, depth=None):
        """Initializer method of Leaf node class"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def count_nodes_below(self, only_leaves=False):
        """Counts how many nodes are there"""
        return 1

    def max_depth_below(self):
        """Compute how deep this node can go"""
        return self.depth

    def get_leaves_below(self):
        """Return the self(Leaf) object"""
        return [self]

    def update_bounds_below(self):
        """Nothing"""
        pass

    def pred(self, x):
        """Make a prediction for a single sample"""
        return self.value

    def __str__(self):
        """This method will be used to print out the state of Leaf"""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """This is a Decision tree node class"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initiçalizer method of Decision tree class"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def count_nodes(self, only_leaves=False):
        """Counts how many nodes are there"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def depth(self):
        """How deep can reach this tree"""
        return self.root.max_depth_below()

    def get_leaves(self):
        """Return the leafs is this tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """update bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """Make a prediction for a single sample"""
        return self.root.pred(x)

    def update_predict(self):
        """This method will update the predict function of this tree"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def np_extrema(self, arr):
        """Return the min and max of an array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Pick a random feature and threshold to split a node"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the given data"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
                - Depth                     : {self.depth()}
                - Number of nodes           : {self.count_nodes()}
                - Number of leaves          : \
                    {self.count_nodes(only_leaves=True)}
                - Accuracy on training data : \
                    {self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        """Recursively build the tree from a given node"""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = (np.sum(left_population) < self.min_pop
                        or node.depth + 1 == self.max_depth
                        or np.unique(
                            self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (np.sum(right_population) < self.min_pop
                         or node.depth + 1 == self.max_depth
                         or np.unique(
                             self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Build a leaf child holding the most represented class"""
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Build an internal node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Return the proportion of correct predictions"""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size

    def __str__(self):
        """This method will be used to print out the state of
        Decision_tree"""
        return self.root.__str__()
