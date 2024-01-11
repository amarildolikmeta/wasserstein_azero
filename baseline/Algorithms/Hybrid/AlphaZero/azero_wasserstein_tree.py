import copy
from math import sqrt
from random import shuffle

#import igraph as ig
import logging
import numpy as np
import tabulate as tab
from scipy.stats import norm
from .azero_node import AzeroNode
from.azero_tree import AzeroTree, my_argmax
# Use default (root) logger
logger = logging.getLogger()


class AzeroWassersteinTree(AzeroTree):
    def __init__(self, env, args, verbose=0):
        super().__init__(env, args, verbose)
        self.delta = delta = args["delta"]
        self.standard_bound = norm.ppf(delta, loc=0, scale=1)
        self.backpropagation = args["tree_params"]["backpropagation"]
        self.tree_policy = args["tree_params"]["tree_policy"]

    def best_uct(self, node):
        """
        Args:
            node:
        Returns:
        """
        factor = int(self.tree_policy == "mean")
        upper_bounds = [(q + factor * self.standard_bound * node.sigmas[i], i) for i, q in
                        enumerate(node.Qs)]
        upper_bounds.sort(key=lambda x: x[0], reverse=True)

        # Shuffle node with same ucb
        if len(upper_bounds) > 1 and upper_bounds[0][0] == upper_bounds[1][0]:  # No need to shuffle scalar or different values
            idx = 0
            for idx, el in enumerate(upper_bounds):
                if el[0] != upper_bounds[0][0]: break

            sublist = upper_bounds[:idx]
            shuffle(sublist)
            upper_bounds[:idx] = sublist

        # Assicurati che il nodo sia valido (corrisponda ad azione valida)
        for utc, idx in upper_bounds:
            child = node.children[idx]
            if child.is_valid_action and not child.is_masked:
                return child

        # If here something is wrong
        logger.error("error node: {}".format(node.S))
        for child in node.children:
            logger.error(child.is_valid_action)
            logger.error(child.is_masked)
            logger.error(child.action)
        return None

    def expand(self, node):
        """
        Args:
            node:
        Returns:
        """
        if node.done:
            return
        self.env.set_S(node)
        is_valids = [self.env.is_valid_action(action) for action in range(self.env.n_actions)]
        node.children = [AzeroNode(node, action, is_valid=is_valid) for action, is_valid in enumerate(is_valids)]
        tmpS = []  # State of children to be evaluated by nn
        any_solved_state = False  # If True it exists a solved-state children
        for child in node.children:
            self.env.set_S(node)
            S_, reward, done, info = self.env.step(child.action)
            child.S = S_.copy()
            child.is_terminal = done
            child.done = done
            child.r = reward
            if "solved" in info:
                any_solved_state += info["solved"]
                child.is_solved = info["solved"]
            tmpS.append(child.S["nn_input"])

        # It is faster to interrogate policy once
        tmpS = np.asarray(tmpS)
        Qs, sigmas = self.brain.predict(tmpS)
        for child, q, sigma in zip(node.children, Qs, sigmas):
            # Mask non-solved children
            if any_solved_state and not child.is_solved:
                child.is_masked = True
            child.Qs = q
            child.sigmas = sigma
            if child.is_solved or child.is_terminal:
                child.Qs *= 0
                child.sigmas *= 0

    def backpropagate(self, node):
        """
        Args:
            node:
        Returns:
        """
        upper_bounds = node.Qs + self.standard_bound * node.sigmas
        best_arm = my_argmax(upper_bounds)
        result = (node.Qs[best_arm], node.sigmas[best_arm])
        ret = result if not node.is_terminal else (0, 0)
        depth = 1
        while node.parent is not None:
            depth += 1
            if self.backpropagation != "mc":
                if self.backpropagation == "wass":
                    if len(node.children) > 0:
                        counts = np.array([child.N for child in node.children])
                        counts = counts / np.sum(counts)
                        q = np.sum(node.Qs * counts)
                        sigma = np.sum(node.sigmas * counts)
                        result = (q, sigma)
                        ret = result if not node.is_terminal else (0, 0)
                else:
                    if self.backpropagation == "optimistic":
                        values = node.Qs + self.standard_bound * node.sigmas
                    elif self.backpropagation == "mean":
                        values = node.Qs
                    elif self.backpropagation == "posterior":
                        values = np.random.normal(loc=node.Qs, scale=node.sigmas)
                    else:
                        raise ValueError("Backpropagation not implemented")
                    best_arm = my_argmax(values)
                    result = (node.Qs[best_arm], node.sigmas[best_arm])
                    ret = result if not node.is_terminal else (0, 0)
            ret = (node.r + self.gamma * ret[0], self.gamma * ret[1])
            node.N += 1
            node.W += ret[0]
            node.sum_sig += ret[1]
            if self.backpropagation == "wass":
                node.parent.Qs[node.action] = ret[0]
                node.parent.sigmas[node.action] = ret[1]
            else:
                node.parent.Qs[node.action] = node.W / node.N
                node.parent.sigmas[node.action] = node.sum_sig / node.N
            node = node.parent
        # this is root
        node.N += 1
        node.W += ret[0]
        node.sum_sig += ret[1]
        node.v = node.W / node.N
        node.sum_sig = node.sum_sig / node.N
        if depth > self.max_depth:
            self.max_depth = depth
    def get_probabilities(self, HER_type):
        """
        Used to get P HER
        Args:
            HER_type:

        Returns:

        """
        # if HER_type == "PosteriorFutureAllP":
        #     P = np.zeros(len(self.probabilities))
        #     P[my_argmax(self.probabilities)] = 1
        #     # P = 0.1 * P + 0.9*np.asarray(self.probabilities)
        #     return P
        #
        # if HER_type == "PosteriorFutureNoisyP":
        #     return np.array(self.noisy_probabilities).copy()

        return self.probabilities

    def get_best_action(self, depth=20, mode=None, selection="optimistic"):
        """
        pitting: Add dirichlet noise
        train: Add dirichlet noise + noisy probabilities
        Args:
            depth:
            mode:

        Returns:

        """
        if len(self.root.children) == 0:
            if self.root.done or self.root.is_terminal:
                raise ValueError("Planning from a terminal state")
            self.expand(self.root)
            Qs, sigmas = self.brain.predict_one(self.root.S["nn_input"])
            self.root.Qs = Qs
            self.root.sigmas = sigmas

        # Add noise to promote exploration
        # if mode == "train" or mode == "pitting":
        #     self.root.P_no_dirichlet = self.root.P
        #     self.root.P = (1 - self.dirichlet_noise_ratio) * self.root.P + \
        #                   self.dirichlet_noise_ratio * np.random.dirichlet([2 for _ in range(self.env.n_actions)])

        for _ in range(depth):
            try:
                node = self.traverse(self.root)
                self.backpropagate(node)
                if not node.is_terminal: self.expand(node)
            except:
                print("what Happened??")



        children_N = np.asarray([child.N for child in self.root.children])
        parent_N = np.sum(children_N)
        count_probabilities = children_N / parent_N

        upper_bounds = self.root.Qs + self.standard_bound * self.root.sigmas
        if selection == "counts":
            self.probabilities = noisy_probabilities = count_probabilities
        elif selection == "optimistic":
            noisy_probabilities = np.zeros(len(count_probabilities))
            noisy_probabilities[my_argmax(upper_bounds)] = 1.
            self.probabilities = noisy_probabilities
        elif selection == "mean":
            noisy_probabilities = np.zeros(len(count_probabilities))
            noisy_probabilities[my_argmax(self.root.Qs)] = 1.
            self.probabilities = noisy_probabilities


        # if mode == "train":
        #     if self.root.depth > self.n_stochastic_depth:
        #         noisy_probabilities = np.zeros(len(probabilities))
        #         noisy_probabilities[my_argmax(probabilities)] = 1
        #     else:
        #         noisy_probabilities = probabilities.copy()
        # else:
        #     noisy_probabilities = np.zeros(len(probabilities))
        #     noisy_probabilities[my_argmax(probabilities)] = 1

        # if self.verbose > 1:
        #     self.print_infos(noisy_probabilities=noisy_probabilities, probabilities=probabilities)
        self.noisy_probabilities = noisy_probabilities
        self.env.set_S(self.root)
        while True:
            action = np.random.choice(range(len(noisy_probabilities)), p=noisy_probabilities)
            if self.env.is_valid_action(action):
                return action, action

    def print_infos(self, node=None, noisy_probabilities=None, probabilities=None):
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
                 "sigmas": [],
                 "v": [],
                 "Us": [],
                 "is valid": [],
                 "is terminal": []}

        if noisy_probabilities is not None: table["noisy"] = noisy_probabilities
        if probabilities is not None: table["P: N/Ntot"] = probabilities
        table["NN-P"] = node.P

        # get NN value prediction
        nn_inputs = np.asarray([child.S["nn_input"] for child in node.children])
        _, vs = self.brain.predict(nn_inputs)

        table["nn-v"] = vs
        table["Q"] = node.Qs
        table["sigmas"] = node.sigmas
        table["Us"] = node.Qs + self.standard_bound * node.sigmas
        for i, child in enumerate(node.children):
            table["Name"].append(i)
            table["ID"].append(id(child))
            table["N"].append(child.N)
            table["v"].append(child.v)
            table["is valid"].append(child.is_valid_action)
            table["is terminal"].append(child.is_terminal)

        logger.debug("Q node: {}".format(node.Q))
        logger.debug("N node: {}".format(node.N))
        logger.debug("ID node: {}".format(id(node)))
        logger.debug("node depth {}".format(self.root.depth if self.args["single_player"] else self.root.depth % 2))
        logger.debug(tab.tabulate(table, headers="keys", tablefmt="psql", numalign="center"))
