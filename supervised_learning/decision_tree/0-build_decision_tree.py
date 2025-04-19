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

    def max_depth_below(self):
        """This method compute the maximum depth a decision tree can
        reach"""
        if self.is_leaf:
            return 1

        depth = self.left_child.max_depth_below()

        if depth < self.right_child.max_depth_below():
            depth = self.right_child.max_depth_below()

        return depth


class Leaf(Node):
    """This is a Leaf node class"""
    def __init__(self, value, depth=None):
        """Initializer method of Leaf node class"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Compute how deep this node can go"""
        return self.depth


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

    def depth(self):
        """How deep can reach this tree"""
        return self.root.max_depth_below()
