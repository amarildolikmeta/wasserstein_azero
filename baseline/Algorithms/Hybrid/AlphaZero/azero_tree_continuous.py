from math import sqrt
from random import shuffle

import numpy as np
import logging

from .azero_node import AzeroNode
from .azero_tree import AzeroTree, my_argmax

# Use default (root) logger
logger = logging.getLogger()

class AzeroTreeContinuous(AzeroTree):
    def __init__(self, env, args, verbose=False):
        self.alpha = args["alpha"]  # Used as exponent in the progressive widening c_pw*node.N^(alpha)
        self.c_widening = args["c_pw"]  # Used as coefficient in the progressive widening c_pw*node.N^(alpha)
        self.random_action_frac = args["random_action_frac"]
        super().__init__(env, args, verbose)

    def traverse(self, node):
        """
        Traverse the tree starting from node until a leaf node is reached or a node in between needs to be expanded.
        Args:
            node: the starting node

        Returns: first non explored node

        """
        while self.required_number_of_children(node) <= len(node.children):
            node = self.best_uct(node)
        return node

    def required_number_of_children(self, node):
        return max(1, np.ceil(self.c_widening * (node.N ** self.alpha)))

    def add_child_actions(self, node):
        """
        Add an action to node if needed (progressive widening).
        Initialize the action and prior of the new child randomly with probability "random_action_frac".
        Args:
            node:

        Returns: True if a new action is added to node, False otherwise

        """
        # add a child action
        # if node.S["nn_input"] is None:
        #     self.env.set_S(node.parent)
        #     S_, reward, done, info = self.env.step(node.action)
        #     node.S = S_
        #     node.is_terminal = done
        #     node.r = reward
        rand_num = np.random.random()
        if rand_num > self.random_action_frac:
            # Add random action and prior
            a, prior = self.brain.sample(node.S["nn_input"])
            a = a[0]
        else:
            # Query the policy to get the action and the prior
            a = self.env.sample_action()
            # prior = self.brain.log_prob(node.S["nn_input"], a)
            prior = self.brain.log_prob(node.S["nn_input"], a[None, :])
        prior = np.exp(prior)
        # a = np.squeeze(a, axis=0)
        if a.ndim == 0:
            a = a[None]
        if np.isnan(np.sum(a)):
            logger.error("What")
        self.env.set_S(node)
        is_valid = self.env.is_valid_action(a)
        child = AzeroNode(node, a, is_valid=is_valid)
        node.P.append(prior)
        node.children.append(child)
        return child

    def best_uct(self, node):
        """

        Args:
            node:

        Returns:

        """
        UTCs = [(child.Q + self.c_utc * node.P[i] * sqrt(node.N) / (child.N + 1), i) for i, child in
                enumerate(node.children)]
        UTCs.sort(key=lambda x: x[0], reverse=True)
        # Shuffle node with same utc
        if len(UTCs) > 1 and UTCs[0][0] == UTCs[1][0]:  # No need to shuffle scalar or different values
            idx = 0
            for idx, el in enumerate(UTCs):
                if el[0] != UTCs[0][0]: break

            sublist = UTCs[:idx]
            shuffle(sublist)
            UTCs[:idx] = sublist
        # Assicurati che il nodo sia valido (corrisponda ad azione valida)
        for utc, idx in UTCs:
            child = node.children[idx]
            if child.is_valid_action and not child.is_masked:
                return child
        logger.error("what")

    def expand(self, node):
        """

        Args:
            node:

        Returns:

        """
        child = self.add_child_actions(node)
        action = child.action
        self.env.set_S(node)
        S_, reward, done, info = self.env.step(action)
        child.S = S_
        child.is_terminal = done
        child.r = reward
        solved_state = False
        if "solved" in info:
            solved_state = info["solved"]
            child.is_solved = info["solved"]

        vs = self.brain.predict_value(child.S["nn_input"])
        # Mask non-solved children
        if solved_state and not child.is_solved: child.is_masked = True
        if not child.is_terminal:
            child.v = vs[0]
        else:
            child.v = 0
        child.Q = child.r + self.gamma * child.v
        return child

    def get_probabilities(self, HER_type):
        """
        Used to get P HER
        """
        if HER_type == "PosteriorFutureAllP":
            P = np.zeros(len(self.probabilities))
            P[my_argmax(self.probabilities)] = 1
            return P[self.last_action]

        if HER_type == "PosteriorFutureNoisyP":
            return np.array(self.noisy_probabilities).copy()[self.last_action]

        return self.probabilities[self.last_action]

    def get_best_action(self, depth=20, mode=None):
        """
        pitting: Add dirichlet noise
        train: Add dirichlet noise + noisy probabilities

        Args:
            depth:
            mode:

        Returns:

        """

        for _ in range(depth):
            node = self.traverse(self.root)
            if not node.is_terminal: node = self.expand(node)
            self.backpropagate(node)

        children_N = np.asarray([child.N for child in self.root.children])
        parent_N = np.sum(children_N)
        probabilities = children_N / parent_N

        self.probabilities = probabilities

        if mode == "train":
            if self.root.depth > self.n_stochastic_depth:
                noisy_probabilities = np.zeros(len(probabilities))
                noisy_probabilities[my_argmax(probabilities)] = 1
            else:
                noisy_probabilities = probabilities.copy()
        else:
            noisy_probabilities = np.zeros(len(probabilities))
            noisy_probabilities[my_argmax(probabilities)] = 1

        if self.verbose > 1:
            self.print_infos(noisy_probabilities=noisy_probabilities, probabilities=probabilities)
        self.noisy_probabilities = noisy_probabilities
        self.env.set_S(self.root)
        while True:
            action = np.random.choice(range(len(noisy_probabilities)), p=noisy_probabilities)
            a = self.root.children[action].action
            self.last_action = action
            if self.env.is_valid_action(a):
                return a, action

    def print_infos(self, node=None, noisy_probabilities=None, probabilities=None):
        import tabulate as tab

        """

        Args:
            node:
            noisy_probabilities:
            probabilities:

        Returns:

        """
        if node is None: node = self.root

        table = {"Name": [],
                 "ID": [],
                 "N": [],
                 "Q": [],
                 "v": [],
                 "UTC": [],
                 "is valid": [],
                 "is terminal": [],
                 "action": []}

        if noisy_probabilities is not None: table["noisy"] = noisy_probabilities
        if probabilities is not None: table["P: N/Ntot"] = probabilities
        table["NN-P"] = node.P

        # get NN value prediction
        nn_inputs = np.asarray([child.S["nn_input"] for child in node.children])
        _, vs = self.brain.predict(nn_inputs)

        table["nn-v"] = vs

        for i, child in enumerate(node.children):
            table["Name"].append(i)
            table["ID"].append(id(child))
            table["N"].append(child.N)
            table["Q"].append(child.Q)
            table["v"].append(child.v)
            table["UTC"].append(child.Q + self.c_utc * node.P[i] * sqrt(node.N) / (child.N + 1))
            table["is valid"].append(child.is_valid_action)
            table["is terminal"].append(child.is_terminal)
            table["action"].append(child.action)

        logger.debug("Q node: {}".format(node.Q))
        logger.debug("N node: {}".format(node.N))
        logger.debug("ID node: {}".format(id(node)))
        logger.debug("node depth {}".format(self.root.depth if self.args["single_player"] else self.root.depth % 2))
        logger.debug(tab.tabulate(table, headers="keys", tablefmt="psql", numalign="center"))
