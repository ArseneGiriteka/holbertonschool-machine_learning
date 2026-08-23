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
            return 1

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

    def __str__(self):
        """This method will be used to print out the state of
        Decision_tree"""
        return self.root.__str__()
