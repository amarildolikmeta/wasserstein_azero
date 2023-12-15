from math import sqrt
from random import shuffle, sample

#import igraph as ig
import logging
import numpy as np
import tabulate as tab

from .azero_node import AzeroNode

# Use default (root) logger
logger = logging.getLogger()


def my_argmax(listy):
    """
    It works like numpy.argmax but it returns a random index if multiple max are found
    Args:
        listy: the listy

    Returns:

    """
    occurencies = np.argwhere(listy == np.amax(listy))
    return occurencies[np.random.randint(0, len(occurencies))][0]


class AzeroTree:
    def __init__(self, env, args, verbose=0):

        self.args = args

        self.root = AzeroNode(None, None, S=env.get_S(), is_single_player=self.args["single_player"])
        self.env = env
        self.probabilities = None
        self.noisy_probabilities = None

        # Set random seed
        self.random_seed = self.args["random_seed"]
        # np.random.seed(self.random_seed)

        self.verbose = verbose
        self.brain_producer = args['brain_producer']
        self.c_utc = args["c_utc"]
        self.gamma = args['gamma']
        self.dirichlet_noise_ratio = args[
            "dirichlet_noise_ratio"]  # new_P = (1-dirichelt_noise_ratio)*P + dirichelt_noise_ratio
        self.n_stochastic_depth = args["n_stochastic_depth"]  # number of step before deterministic action choice
        self.max_depth = 0
        # You can't initialize a brain here, because you need to pickle it later
        self.brain = None

    def reset_env(self):
        self.env.reset()
        #self.env.seed(np.random.randint(100000))
        self.root = AzeroNode(None, None, S=self.env.get_S(), is_single_player=self.args["single_player"])

    def reset(self):
        self.reset_env()
        self.probabilities = None

    def set_env(self, env):
        self.env = env
        self.root = AzeroNode(None, None, S=self.env.get_S(), is_single_player=self.args["single_player"])

    def set_brain(self, brain):
        self.brain = brain

    def set_brain_weights(self, weights):
        if self.brain is None:
            self.brain = self.brain_producer(self.args["obs_dim"], self.args["act_dim"],
                                             **self.args["brain_params"])
        self.brain.set_weights(weights.copy())

    def set_new_root(self, action, new_root_state):

        try:
            # if state was not explored yet
            self.root.children[action]
        except IndexError:
            self.expand(self.root)

        # and depth?? Check it

        self.root = self.root.children[action]
        self.max_depth = 0
        self.root.parent = None
        if self.root.S is None:
            self.root.S = new_root_state

    def traverse(self, node):

        while len(node.children) != 0:
            node = self.best_uct(node)

        return node

    def best_uct(self, node):
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

        # If here something is wrong
        logger.error("error node: {}".format(node.S))
        for child in node.children:
            logger.error(child.is_valid_action)
            logger.error(child.is_masked)
            logger.error(child.action)
        return None

    def expand(self, node):
        self.env.set_S(node)

        is_valids = [self.env.is_valid_action(action) for action in range(self.env.n_actions)]

        node.children = [AzeroNode(node, action, is_valid=is_valid) for action, is_valid in enumerate(is_valids)]

        tmpS = []  # State of children to be evaluated by nn
        any_solved_state = False  # If True it exists a solved-state children

        for child in node.children:

            self.env.set_S(node)

            S_, reward, done, info = self.env.step(child.action)

            child.S = S_
            child.is_terminal = done
            child.r = reward
            if "solved" in info:
                any_solved_state += info["solved"]
                child.is_solved = info["solved"]

            tmpS.append(child.S["nn_input"])

        # It is faster to interrogate policy once
        tmpS = np.asarray(tmpS)
        Ps, vs = self.brain.predict(tmpS)

        for child, v, P in zip(node.children, vs, Ps):

            # Mask non-solved children
            if any_solved_state and not child.is_solved: child.is_masked = True

            if not child.is_terminal:
                child.v = v[0]
                child.P = P
            else:
                child.v = 0
            child.Q = child.r + self.gamma * child.v

    def backpropagate(self, node):
        result = node.v
        ret = result if not node.is_terminal else 0
        depth = 1
        while node.parent is not None:
            depth += 1
            ret = node.r + self.gamma * ret  # Il primo nodo non dovrebbe avere ret = reward?
            node.N += 1
            node.W += ret
            node.Q = node.W / node.N
            node = node.parent
        # this is root
        node.N += 1
        node.W += ret
        node.v = node.W / node.N
        if depth > self.max_depth:
            self.max_depth = depth
    def render(self, path, node=None):
        """

        Args:
            path:
            node:

        Returns:

        """
        visited = list()  # Set to keep track of visited nodes.

        graph = ig.Graph(directed=True)

        if node is None: node = self.root

        AzeroTree.dfs(visited, graph, node)

        graph_depth = len(graph.get_diameter(directed=True))
        n_children = len(node.children)
        N = 600  # Constant

        visual_style = {"vertex_label": ["{} \n Q: {:.2} ; N: {} \n S: {}".format(name, q, n, s) for name, q, n, s in
                                         zip(graph.vs["S"], graph.vs["Q"], graph.vs["N"], graph.vs["solved"])],
                        "vertex_color": graph.vs["color"], "edge_label": ["{:.2}".format(p) for p in graph.es["P"]],
                        "layout": graph.layout_reingold_tilford(root=[0]), "margin": 100,
                        "bbox": (n_children * graph_depth * N, N * graph_depth)}
        # visual_style["vertex_label_size"] = 4
        # visual_style["edge_width"] = [1 + 2 * p / (node.N+1) for p in graph.es["P_Tree"]]

        logger.info("pixel {}x{}".format(n_children * graph_depth * N, N * graph_depth))

        ig.plot(graph, path, **visual_style)

    def bfs(self, max=np.inf):  # function for BFS
        visited = []
        queue = []
        node = self.root
        queue.append(node)
        count = 0
        while queue and count < max:  # Creating loop to visit each node
            node = queue.pop(0)
            if not node.children or len(node.children) == 0:
                children = []
            else:
                children = sample(node.children, len(node.children))
            for child in children: # no bias wrt action index
                visited.append(child)
                queue.append(child)
                count += 1
                if count >= max:
                    break
        return visited

    @staticmethod
    def dfs(visited, graph, node):
        """

        Args:
            visited:
            graph:
            node:

        Returns:

        """
        if id(node) not in visited:

            visited.append(id(node))

            graph.add_vertex()
            IDX_graph_node = len(graph.vs) - 1
            graph_node = graph.vs[IDX_graph_node]

            graph_node["ID_node"] = id(node)
            graph_node["N"] = node.N
            graph_node["Q"] = float(node.Q)
            graph_node["Player"] = node.player
            graph_node["S"] = node.S["current_state"]
            graph_node["solved"] = node.is_solved

            if node.is_solved:
                graph_node["color"] = "green"
            elif (node.is_terminal and not node.is_solved) or not node.is_valid_action:
                graph_node["color"] = "red"
            elif node.is_masked:
                graph_node["color"] = "orange"
            else:
                graph_node["color"] = "blue"

            if node.parent is not None:
                IDX_graph_parent = visited.index(id(node.parent))
                graph.add_edge(IDX_graph_parent, IDX_graph_node)

                IDX_graph_edge = len(graph.es) - 1
                graph_edge = graph.es[IDX_graph_edge]

                idx = node.parent.children.index(node)
                prob = node.parent.P[idx]
                graph_edge["P"] = prob
                graph_edge["P_Tree"] = graph_node["N"]

            for child in node.children:
                AzeroTree.dfs(visited, graph, child)

    def get_probabilities(self, HER_type):
        """
        Used to get P HER
        Args:
            HER_type:

        Returns:

        """
        if HER_type == "PosteriorFutureAllP":
            P = np.zeros(len(self.probabilities))
            P[my_argmax(self.probabilities)] = 1
            # P = 0.1 * P + 0.9*np.asarray(self.probabilities)
            return P

        if HER_type == "PosteriorFutureNoisyP":
            return np.array(self.noisy_probabilities).copy()

        return self.probabilities

    def get_best_action(self, depth=20, mode=None):
        """
        pitting: Add dirichlet noise
        train: Add dirichlet noise + noisy probabilities
        """

        if len(self.root.children) == 0:
            self.expand(self.root)
            P, v = self.brain.predict_one(self.root.S["nn_input"])
            self.root.P = P

        # Add noise to promote exploration
        if mode == "train" or mode == "pitting":
            self.root.P_no_dirichlet = self.root.P
            self.root.P = (1 - self.dirichlet_noise_ratio) * self.root.P + \
                          self.dirichlet_noise_ratio * np.random.dirichlet([2 for _ in range(self.env.n_actions)])

        for _ in range(depth):

            node = self.traverse(self.root)
            self.backpropagate(node)

            if not node.is_terminal: self.expand(node)

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
                 "v": [],
                 "UTC": [],
                 "is valid": [],
                 "is terminal": []}

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
            table["UTC"].append(child.Q + node.P[i] * sqrt(node.N) / (child.N + 1))
            table["is valid"].append(child.is_valid_action)
            table["is terminal"].append(child.is_terminal)

        logger.debug("Q node: {}".format(node.Q))
        logger.debug("N node: {}".format(node.N))
        logger.debug("ID node: {}".format(id(node)))
        logger.debug("node depth {}".format(self.root.depth if self.args["single_player"] else self.root.depth % 2))
        logger.debug(tab.tabulate(table, headers="keys", tablefmt="psql", numalign="center"))
