from collections import deque

import numpy as np

'''
Cambio un po' le carte in tavola rispetto al paper di AlphaGo
Per loro la rete neurale dovrebbe restituire la value (la reward finale rispetto
al giocatore che deve giocare quel nodo). Io invece restituisco la reward di fine
episodio e tengo conto di questo nella backpropagation)
'''


class AzeroNode:
    def __init__(self, parent, action, is_single_player=None, is_valid=True, S=None):
        self.action = action
        self.parent = parent
        self.depth = 0 if parent is None else (parent.depth + 1)
        self.S = S
        self.W = 0  # Sum of Qs passing through this node
        self.sum_sig = 0  # Sum of sigmas passing through this node
        self.Q = 0  # Q of parent action = self.r + gamma * self.V
        self.v = None  # value of node -> \sum_a (Q_a * N_a ) / N
        self.N = 0
        self.r = 0
        self.P = []  # NN prediction + noise (if training)
        self.P_no_dirichlet = []  # NN prediction, without additional exploration
        self.children = []
        self.is_terminal = not is_valid  # True if self is a terminal state (not every leafs are terminal)
        self.is_valid_action = is_valid  # False if self correspond to invalid action
        self.is_masked = False  # True if a brother-node is a solution
        self.is_solved = False  # True if self is a solved state (requires that env return "solved" key in info = env.step())
        self.prediction = None
        self.is_single_player = self.parent.is_single_player if is_single_player is None else is_single_player
        self.player = 1 if self.is_single_player or (self.depth % 2 == 0) else -1


