import numpy as np


class Solver:

    def predict_one(self, state):

        pi = np.zeros(state.shape[0])
        indices = np.argwhere(state == -1)
        pi[indices] = 1 / len(indices)

        return pi, -len(indices)