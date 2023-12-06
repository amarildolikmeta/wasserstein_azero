# Solovay Kitaev algorithm implemented in python https://github.com/cryptogoth/skc-python

import gym
import gym.spaces
import numpy as np
import qutip as qt
import logging

try:
    from gates import GATES
except:
    from .gates import GATES

from gym.utils import seeding
from qutip.qip.operations.gates import gate_expand_1toN, gate_expand_2toN, controlled_gate
from qutip.qip.circuit import QubitCircuit

# Use default (root) logger
logger = logging.getLogger()

class Compiler(gym.Env):
    def __init__(self, n_qubits, base_gates, target_gates, min_target, max_target, max_length_circuit, epsilon,
                 reward_type, norm, concatenate_input, target=None, gym_standard=False, no_glob_phase=False):
        """

        :param base_gates:
        :param target_gates:
        :param min_target:
        :param max_target:
        :param epsilon:
        :param reward_type:
        :param norm:
        :param gym_standard:
        """

        self.n_qubits = n_qubits  # The number of qubits in the circuit
        self.dim_space = 2 ** n_qubits  # The dimension of the unitary space (dim_space*dim_space)

        self.base_gates = GATES[base_gates]
        self.base_targets = GATES[target_gates]

        # Define the action space action. Action = (n, m, q)
        self.n_parameters, self.n_gates, self.n_targets = self.init_action_info()
        self.n_actions = [self.n_parameters, self.n_gates,
                          self.n_targets]  # Number of parameters + gate index + target index

        self.concatenate_input = concatenate_input  # If True it returns [U,T] as nn_input instead of [X] where T = X*U

        self.gym_standard = gym_standard  # True if you are using stable baselines
        self.no_glob_phase = no_glob_phase  # True del global phase from target and current state

        if gym_standard:

            if concatenate_input:
                Range = 6 if (self.dim_space == 2) else 2 * (
                        2 * self.dim_space ** 2)  # Single qubit unitaries can be parametrized
            else:
                Range = 3 if (self.dim_space == 2) else 2 * (
                            self.dim_space ** 2)  # Single qubit unitaries can be parametrized

            high = np.array([np.finfo(np.float32).max for i in range(Range)])  # 2x due to Re and Im part
            self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.epsilon = epsilon
        self.max_length_circuit = max_length_circuit
        self.reward_type = reward_type
        self.norm = norm
        self.min_target = min_target
        self.max_target = max_target

        self.current_state = np.identity(self.dim_space)  # It is the unitary matrix representing the circuit
        self.qt_circuit = QubitCircuit(self.n_qubits)
        qt_gates = [gate.qutip_gate(normalized_params=True) for gate in self.base_gates]
        self.qt_circuit.user_gates = dict(zip(range(len(self.base_gates)), qt_gates))

        self.done = False  # True when the episode is over
        self.solved = False  # True when a solution is found
        self.timestep = 1

        self.actions_sequence = []  # To keep track of the actions performed

        # Define the unitary target to approximate
        if target is not None or target == "None":
            # You keep the unitary target at each episode
            if target == "random_fixed":
                # You create the target randomly
                self.target_unitary = Compiler.rand_unitary(self.min_target, self.max_target, self.base_targets,
                                                            self.norm, self.epsilon, self.dim_space)
            else:
                self.target_unitary = target

            self.fixed_target = True
        else:
            # You change the unitary target every episode
            self.target_unitary = Compiler.rand_unitary(self.min_target, self.max_target, self.base_targets, self.norm,
                                                        self.epsilon, self.dim_space)
            self.fixed_target = False

    def init_action_info(self):
        """
        The action is a list (n, m, q) of n-real parameters, an integer m (gate index), an integer q (target index)
        Returns: a list of n_parameters, n_action, n_target

        Remember:
            - 0 <= m < n_action
            - 0 <= q < n_target

        """

        logger.debug("This is the Compiler init")

        n_parameters = 0
        for gate in self.base_gates:
            if not gate.is_parametric: continue
            if gate.n_parameters > n_parameters:
                n_parameters = gate.n_parameters

        n_action = len(self.base_gates)
        n_target = self.n_qubits

        return n_parameters, n_action, n_target

    def sample_action(self):
        """
        Returns a random action

        Returns:

        """

        transposed_action = np.random.uniform(low=-1, high=1, size=(self.n_parameters + 2, self.n_qubits))
        transposed_action[-2] = np.random.randint(self.n_gates - 1, size=self.n_qubits)
        transposed_action[-1] = np.random.randint(self.n_targets - 1, size=self.n_qubits)

        return np.transpose(transposed_action)

    def reset(self):
        """
        Reset the environment

        Returns: the state S and False

        """

        self.done = False
        self.solved = False
        self.actions_sequence = []
        self.current_state = np.identity(self.dim_space)

        self.timestep = 0

        if not self.fixed_target:
            self.target_unitary = Compiler.rand_unitary(self.min_target, self.max_target, self.base_targets, self.norm,
                                                        self.epsilon, self.dim_space)

        if self.gym_standard: return self.get_S()
        return self.get_S(), self.done

    def get_reward(self, unitary=None, target=None, evaluate=False):
        """
        Return the reward for the current state.
        Using "average_gate_fidelity" to evaluate the distance between
        See qutip.metrics for details.

        Args:
            unitary:
            target:
            evaluate:

        Returns:

        """

        # if not isinstance(unitary, qt.Qobj): unitary = qt.Qobj(unitary)
        # if not isinstance(target, qt.Qobj): target = qt.Qobj(target)

        if not evaluate:
            unitary = self.current_state
            target = self.target_unitary

        distance_from_target = Compiler.distance(unitary, target, self.norm)

        tolerance = 1 - self.epsilon

        if self.norm == "fidelity":
            accuracy = 1 - distance_from_target
        else:
            accuracy = distance_from_target

        if self.reward_type == 0:
            '''
            Using "sparse reward". Best choice if using Hindsight Experience Replay
            '''

            # If solved
            if accuracy < tolerance:
                if not evaluate:
                    self.done = True
                    self.solved = True
                # The return when the goal is reached
                return -1, {"distance": distance_from_target, "solved": True}

            # If performed too many actions
            if not evaluate:
                if self.timestep >= self.max_length_circuit:
                    self.done = True

            # The return when the goal is not reached
            return -1, {"distance": distance_from_target, "solved": False}

        if self.reward_type == 1:
            '''
            The old informative reward
            '''

            # If solved
            if accuracy < tolerance:
                if not evaluate:
                    self.done = True
                    self.solved = True
                # The return when the goal is reached
                return (self.max_length_circuit - self.timestep) + 1, {"distance": distance_from_target, "solved": True}

            # If performed too many actions
            if not evaluate:
                if self.timestep >= self.max_length_circuit:
                    self.done = True

            # The return when the goal is not reached
            return -(1 - distance_from_target) / self.max_length_circuit, {"distance": distance_from_target,
                                                                           "solved": False}

    @staticmethod
    def remove_global_phase(matrix):
        """
        Return a matrix with global phase = 1. It uses Binet theorem
        Args:
            matrix:

        Returns:

        """

        det = np.linalg.det(matrix)
        det_rad = np.angle(det)
        dim = matrix.shape[0]
        new_matrix = np.exp(-1j / dim * det_rad) * matrix
        #new_det = np.linalg.det(new_matrix)

        return new_matrix

    def current_state_to_S(self, current_state=None, target=None, onehot=True):
        """
        Transform the current circuit to neural network input

        Returns X where

        Target = X * current_state

        if a = current_state and b = Target then

        Ax = b
        (Ax)^T = b^T
        x^T A^T = b^T

        Args:
            current_state:
            target:
            onehot: if True it returns a flatten array

        Returns: x

        """

        if current_state is None: current_state = self.current_state
        if target is None: target = self.target_unitary

        if self.no_glob_phase:
            target = Compiler.remove_global_phase(target)
            #current_state = Compiler.remove_global_phase(current_state)

        if self.dim_space != 2:
            x_transposed = np.linalg.solve(current_state.T, target.T)
            x = x_transposed.T
            #x = Compiler.remove_global_phase(x)
        else:

            x = np.zeros((2, 2), dtype=complex)
            (c1, c2), (c3, c4) = current_state
            (t1, t2), (t3, t4) = target

            D1 = c1 * c4 - c2 * c3
            x[0, 0] = (c4 * t1 - c3 * t2) / D1
            x[0, 1] = (c1 * t2 - c2 * t1) / D1
            x[1, 0] = (c4 * t3 - c3 * t4) / D1
            x[1, 1] = (c1 * t4 - c2 * t3) / D1

        if onehot:
            if self.dim_space == 2:

                # TESTING
                th_U, phi1_U, phi2_U = self.get_unitary_parameters(current_state)
                th_T, phi1_T, phi2_T = self.get_unitary_parameters(target)

                if self.concatenate_input:
                    return np.asarray([th_U, phi1_U, phi2_U, th_T, phi1_T, phi2_T])
                else:
                    th, phi1, phi2 = Compiler.get_unitary_parameters(x)
                    return np.asarray([th, phi1, phi2]).flatten()

            # TESTING
            if self.concatenate_input:
                return np.asarray([[a, b, c, d] for a, b, c, d in
                                   zip(current_state.real, current_state.imag, target.real, target.imag)]).flatten()
            else:
                return np.asarray([[a, b] for a, b in zip(x.real, x.imag)]).flatten()
        else:
            return x

    def get_S(self):
        """

        Returns:

        """
        if self.gym_standard: return self.current_state_to_S(self.current_state, self.target_unitary, True)

        S = {}

        S["current_state"] = self.current_state
        S["target_unitary"] = self.target_unitary
        S["goal"] = S["target_unitary"]
        S["nn_input"] = self.current_state_to_S(self.current_state, self.target_unitary, True)
        S["fixed_target"] = self.fixed_target
        S["done"] = self.done
        S["solved"] = self.solved
        S["gym_standard"] = self.gym_standard
        S["timestep"] = self.timestep

        return S

    def is_valid_action(self, action):
        """
        Check if a specific action is valid

        Args:
            action: the action to check

        Returns: bool

        """
        return not self.done

    def set_S(self, node):
        """
        Used to load the environment from a tree node

        Args:
            node: the tree node

        Returns: None

        """

        self.current_state = node.S["current_state"]
        self.target_unitary = node.S["target_unitary"]
        self.fixed_target = node.S["fixed_target"]
        self.done = node.S["done"]
        self.solved = node.S["solved"]
        self.gym_standard = node.S["gym_standard"]

        self.timestep = node.S["timestep"]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass

    @staticmethod
    def get_unitary_parameters(U):
        """
        Used to get the parametrization of single qubit unitaries.

        e^(i phi/2) [e^(i phi1) cos(th), e^(i phi2) sin(th)]
                    [-e^(-i phi2) sin(th), e^(-i phi1) cos(th)]

        Args:
            U: Single qubit unitary

        Returns: three parameters (no global phase)

        """

        if isinstance(U, qt.Qobj): U = U.full()

        det = np.linalg.det(U)
        phi = np.angle(det)

        U = U * np.exp(-1j * phi / 2)

        A = U[0, 0]
        B = U[0, 1]
        C = U[1, 0]
        D = U[1, 1]

        if (A * D).real != 0:
            # abs mandatory! if A*B should be zero but it is > 0 (epsilon) it can be positive!!
            th = np.arctan(np.sqrt(np.abs((B * C).real / (A * D).real)))
        else:
            th = np.arcsin(np.sqrt(np.abs((B * C).real)))

        default_number = 1  # set ph1 or ph2 to a default value
        if np.cos(th) == 0 and np.sin(th) != 0:
            phi1 = default_number
            phi2 = np.angle(B / np.sin(th))
        elif np.cos(th) != 0 and np.sin(th) == 0:
            phi1 = np.angle(A / np.cos(th))
            phi2 = default_number
        else:
            phi1 = np.angle(A / np.cos(th))
            phi2 = np.angle(B / np.sin(th))

        # Not returning global phase phi due to fidelity metric
        return th, phi1, phi2

    @staticmethod
    def rand_unitary(min_target, max_target, base_gates_target, norm, epsilon, dim_space):
        """

        Args:
            min_target:
            max_target:
            base_gates_target:
            norm:
            epsilon:
            dim_space:

        Returns:

        """
        is_valid_target = False
        n_qubits = int(np.log2(dim_space))

        while not is_valid_target:
            # continue generating a random unitary matrix U until d(U, identity)>epsilon

            if min_target == 0:
                # Return a Haar unitary
                unitary = qt.rand_unitary_haar(dim_space).full()

            else:
                # Generate the target as composition of target_base gates
                unitary = np.identity(dim_space)

                # np.random.triangular(MIN, PEAK, MAX) with [MIN, MAX]
                length = int(np.random.triangular(min_target, max_target, max_target))

                idx = np.random.choice(len(base_gates_target), length)

                for i in idx:
                    gate = base_gates_target[i]

                    if gate.is_parametric:
                        qt_gate = gate.get_random_unitary(qt_object=True)
                    else:
                        qt_gate = gate.full(qt_object=True)

                    if gate.N == 2:  # if gate is a single-qubit gate
                        if gate.is_controlled:
                            control_idx, target_idx = np.random.choice(np.arange(0, n_qubits), replace=False, size=2)
                            gate = controlled_gate(qt_gate, N=n_qubits, control=control_idx,
                                                   target=target_idx).full()
                        else:
                            target_idx = np.random.randint(0, n_qubits)
                            gate = gate_expand_1toN(qt_gate, n_qubits, target=target_idx).full()
                    else:
                        control_idx, target_idx = np.random.choice(range(n_qubits), replace=False, size=2)
                        gate = gate_expand_2toN(qt_gate, N=n_qubits, targets=[control_idx, target_idx]).full()

                    unitary = np.matmul(gate, unitary)

            is_valid_target = Compiler.is_valid_target(unitary, norm, epsilon, dim_space)

        return unitary

    def step(self, actions, normalized_params=True):
        """
        actions is a list of actions to perform for each qubit in the circuit.
        Each action has (n, m, q) where n is a set of n real numbers equal to the maximum number of parameters, m is the
        gate index to apply and q is the control qubit index.
        Args:
            actions: list of (n, m, q)

        Returns:

        """

        # is_available = np.ones(self.n_qubits, dtype=bool)
        # single_qubit_gates = np.zeros(self.n_qubits, dtype=object)

        for target_qubit_idx, action in enumerate(actions):

            gate_idx = int(action[-2])
            control_qubit_idx = int(action[-1])
            parameters = action[:-2]

            gate = self.base_gates[gate_idx]  # Select the only gate in the action space (azero continuous only)
            if gate.is_parametric:
                qt_gate = gate.full(parameters[:gate.n_parameters], qt_object=True, normalized_params=normalized_params)
            else:
                qt_gate = gate.full(qt_object=True)

            # Handle non controlled gates
            if gate.N == 2:  # if gate is a single-qubit gate
                # Handle controlled single qubit gates
                if gate.is_controlled:
                    if target_qubit_idx == control_qubit_idx: continue
                    qt_gate = controlled_gate(qt_gate, N=self.n_qubits, control=control_qubit_idx,
                                              target=target_qubit_idx)
                else:
                    qt_gate = gate_expand_1toN(qt_gate, N=self.n_qubits, target=target_qubit_idx)
            elif gate.N == 2:  # if gate is a two-qubit gate
                if target_qubit_idx == control_qubit_idx: continue
                qt_gate = gate_expand_2toN(qt_gate, N=self.n_qubits, targets=[target_qubit_idx, control_qubit_idx])

            # Apply gate to the circuit
            self.current_state = qt_gate.full() @ self.current_state

        reward, info = self.get_reward()

        # Add chosen action to sequence
        self.actions_sequence.append(actions)

        self.timestep += 1

        info["solved"] = self.solved
        info["length"] = len(self.actions_sequence)
        info["goal"] = self.target_unitary
        info["final_state"] = self.current_state

        return self.get_S(), reward, self.done, info

    @staticmethod
    def is_valid_target(target, norm, epsilon, dim_space) -> bool:
        """

        Args:
            target:
            norm:
            epsilon:
            dim_space:

        Returns:

        """
        # Check if the target is valid i.e. the target is not a solution
        distance = Compiler.distance(target, qt.identity(dim_space), norm)
        if norm == "fidelity":
            is_valid_target = distance < epsilon
        else:
            is_valid_target = distance > epsilon

        return is_valid_target

    @staticmethod
    def distance(A, B, norm):
        """

        Args:
            A:
            B:
            norm:

        Returns:

        """

        if not isinstance(A, qt.Qobj): A = qt.Qobj(A)
        if not isinstance(B, qt.Qobj): B = qt.Qobj(B)

        if norm == "diamond":
            return qt.dnorm(A, B, solver="SCS")
        if norm == "trace":
            return (A - B).norm()
        if norm == "fidelity":
            return qt.average_gate_fidelity(A, B)


class CompilerFullDiscrete(Compiler):
    def __init__(self, n_qubits, base_gates, target_gates, min_target, max_target, max_length_circuit, epsilon,
                 reward_type, norm, concatenate_input, target, gym_standard, no_glob_phase):
        super().__init__(n_qubits, base_gates, target_gates, min_target, max_target, max_length_circuit, epsilon,
                         reward_type, norm, concatenate_input, target, gym_standard, no_glob_phase)

        if self.gym_standard:
            self.action_space = gym.spaces.Discrete(len(self.base_gates))

        _, self.n_actions, self.n_targets = self.init_action_info()

    def init_action_info(self):
        """
        Return the number of possible actions and the number of possible targets

        Remember:
            - 0 <= m < n_action
            - 0 <= q < n_target

        """

        logger.debug("this is the Discrete init")

        n_action = len(self.base_gates)
        n_target = self.n_qubits

        return None, n_action, n_target

    def sample_action(self):
        """
        Returns a random action

        Returns:

        """

        return self.action_space.sample()

    def step(self, action, normalized_params=None):
        """

        Args:
            action:

        Returns:

        """
        gate = self.base_gates[action]  # Select the only gate in the action space (azero continuous only)

        # Evolve
        self.current_state = gate.full() @ self.current_state

        reward, info = self.get_reward()

        # Add chosen action to sequence
        self.actions_sequence.append(action)

        self.timestep += 1

        info["solved"] = self.solved
        info["length"] = len(self.actions_sequence)
        info["goal"] = self.target_unitary
        info["final_state"] = self.current_state

        return self.get_S(), reward, self.done, info


class CompilerFullContinuous(Compiler):
    def __init__(self, n_qubits, base_gates, target_gates, min_target, max_target, max_length_circuit, epsilon,
                 reward_type, norm, concatenate_input, target, gym_standard, no_glob_phase):
        super().__init__(n_qubits, base_gates, target_gates, min_target, max_target, max_length_circuit, epsilon,
                         reward_type, norm, concatenate_input, target, gym_standard, no_glob_phase)

        # Define the action space action. Action = (n, m, q)
        self.n_actions, self.n_gates, self.n_targets = self.init_action_info()

        # if self.gym_standard :
        # Every parameters is returned in the range [-1,1] and then expanded between [min, max] in the step method
        high = np.ones(self.n_actions, dtype=np.float32)
        self.action_space = gym.spaces.Box(-high, high)

    def init_action_info(self):

        logger.debug("this is the Continuous init")

        n_parameters = 0
        for gate in self.base_gates:
            n_parameters += gate.n_parameters

        n_action = len(self.base_gates)
        n_target = self.n_qubits

        return n_parameters, n_action, n_target

    def sample_action(self):
        """
        Returns a random action

        Returns:

        """

        return self.action_space.sample()

    def step(self, actions, normalized_params=True):
        """

        Args:
            actions:

        Returns:

        """

        if normalized_params:
            assert np.less_equal(np.abs(actions), 1).all(), "inputs must be in the range [-1, 1] {}".format(actions)

        gate = self.base_gates[0]  # Select the only gate in the action space (azero continuous only)

        # Evolve
        self.current_state = gate.full(actions, normalized_params=normalized_params) @ self.current_state

        reward, info = self.get_reward()

        # Add chosen action to sequence. The parameters are saved NON normalized
        self.actions_sequence.append(gate.normalized_to_params(actions))

        self.timestep += 1

        info["solved"] = self.solved
        info["length"] = len(self.actions_sequence)
        info["goal"] = self.target_unitary
        info["final_state"] = self.current_state

        return self.get_S(), reward, self.done, info
