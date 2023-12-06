import math
from functools import partial

import numpy as np
from qutip import Qobj
from qutip.qip.operations.gates import gate_expand_1toN, cnot, sigmax


class ParametricGate:
    def __init__(self, U, range_parameters, is_controlled=False):
        """
        Class that implements parametric gates

        Args:
            U: a function that returns the gate matrix if some parameters are passed
            range_parameters: [(min1, max1), (min2, max2), ..., (min, max)]. The parameters range
        """

        self.U = U

        self.is_parametric = False if range_parameters is None else True
        self.is_controlled = is_controlled

        if self.is_parametric:
            self.n_parameters = len(range_parameters)
            self.range_parameters = range_parameters  # [(min1, max1), (min2, max2), ..., (min, max)]
            min_parameters, max_parameters = zip(*range_parameters)
            self.min_parameters = np.asarray(min_parameters)  # [min1, min2, ..., min]
            self.max_parameters = np.asarray(max_parameters)  # [max1, max2, ..., max]

            # It defines the normalization range for the parameters, i.e. [-1, 1]
            self.normalized_min_parameters = -np.ones(self.n_parameters)
            self.normalized_max_parameters = np.ones(self.n_parameters)

            self.N = self.get_random_unitary().shape[0]  # 2
            n = self.N // 2
            self.dims = [[2] * n, [2] * n]

    def qutip_gate(self, normalized_params):
        """
        Return a function to build a qutip gate
        :param normalized_params:
        :return:
        """

        qt_full = partial(self.full, normalized_params=normalized_params)
        return qt_full

    def full(self, parameters=None, normalized_params=True, qt_object=False):
        """
        Returns the matrix that defines the gate

        Args:
            normalized_params: if False parameters are not normalized
            parameters: The parameters
            qt_object: if True return a Qutip Qobj

        Returns: numpy/Qobj array that defines the matrix

        """

        if parameters is None:
            parameters = []

        if normalized_params:
            # convert normalized parameters to true parameters
            parameters = self.normalized_to_params(parameters)

        assert len(parameters) == self.n_parameters, "Too few/many parameters"
        return self.U(*parameters) if not qt_object else Qobj(self.U(*parameters), dims=self.dims)

    @staticmethod
    def rescale_parameters(parameters, r_min, r_max, t_min, t_max):
        """
        https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        Args:
            parameters:
            r_min:
            r_max:
            t_min:
            t_max:

        Returns:

        """
        return (parameters - r_min) / (r_max - r_min) * (t_max - t_min) + t_min

    def params_to_normalized(self, parameters):
        """
        Normalize the parameters in the range [-1,1]

        Args:
            parameters: the parameters to normalize

        Returns:

        """

        if isinstance(parameters, list): parameters = np.asarray(parameters)

        '''
        if np.less_equal(parameters, self.max_parameters).all() and np.greater_equal(parameters, self.min_parameters).all():
            # WARNING it works with angles only!!!
            parameters = np.mod(parameters, self.max_parameters)
        '''

        t_min = self.normalized_min_parameters
        t_max = self.normalized_max_parameters
        r_min = self.min_parameters
        r_max = self.max_parameters

        norm_params = ParametricGate.rescale_parameters(parameters, r_min, r_max, t_min, t_max)

        return norm_params

    def normalized_to_params(self, normalized):
        """
        Inverse function of params_to_normalized.

        Args:
            normalized: the normalized parameters

        Returns:

        """

        if isinstance(normalized, list): normalized = np.asarray(normalized)
        assert np.less_equal(np.abs(normalized), 1).all(), "inputs must be in the range [-1, 1] {}".format(normalized)

        t_min = self.min_parameters
        t_max = self.max_parameters
        r_min = self.normalized_min_parameters
        r_max = self.normalized_max_parameters

        params = ParametricGate.rescale_parameters(normalized, r_min, r_max, t_min, t_max)

        return params

    def get_random_unitary(self, qt_object=False):
        """
        Returns a random matrix by sampling the space of parameters randomly

        Args:
            qt_object: if True return a Qutip Qobj

        Returns: numpy/Qobj array that defines the random gate

        """

        a, b = zip(*self.range_parameters)
        a = np.asarray(a)
        b = np.asarray(b)

        parameters = np.random.uniform(low=a, high=b, size=self.n_parameters)

        return self.full(parameters, qt_object=qt_object, normalized_params=False)


class Gate(ParametricGate):
    def __init__(self, U, is_controlled=False):
        """
        The class that implements non-parametric gates

        Args:
            U: the numpy matrix that defines the gate
        """

        super().__init__(U, None, is_controlled)

        self.N = self.U.shape[0]
        n = self.N // 2
        self.dims = [[2] * n, [2] * n]
        self.qt_U = Qobj(U, dims=self.dims)  # Store here the Qobj instead of converting it on demand

    def full(self, qt_object=False, normalized_params=None):
        """
        Get the matrix

        Args:
            qt_object: if True return a Qutip Qobj

        Returns: numpy/Qutip array that defines the matrix

        """

        return self.U if not qt_object else self.qt_U

    def __mul__(self, other):
        self.U *= other.U
        self.qt_U = Qobj(self.U, dims=self.dims)
        return self


def YY(a):
    return np.array([[math.cos(a), 0, 0, 1j * math.sin(a)],
                     [0, math.cos(a), -1j * math.sin(a), 0],
                     [0, -1j * math.sin(a), math.cos(a), 0],
                     [1j * math.sin(a), 0, 0, math.cos(a)]], dims=[[2, 2], [2, 2]])


def XX(a):
    return np.array([[math.cos(a), 0, 0, -1j * math.sin(a)],
                     [0, math.cos(a), -1j * math.sin(a), 0],
                     [0, -1j * math.sin(a), math.cos(a), 0],
                     [-1j * math.sin(a), 0, 0, math.cos(a)]], dims=[[2, 2], [2, 2]])


# Pauli sigmax gate
SX = np.array([[0, 1],
               [1, 0]], dtype=complex)
# Pauli sigmay gate
SY = np.array([[0, -1.j],
               [1.j, 0]], dtype=complex)
# Pauli sigmaz gate
SZ = np.array([[1, 0],
               [0, -1]], dtype=complex)

# Optimal set of gates
V1 = 1 / np.sqrt(5) * np.array([[1, 2j],
                                [2j, 1]], dtype=complex)
# Optimal set of gates
V2 = 1 / np.sqrt(5) * np.asarray([[1, 2],
                                  [-2, 1]], dtype=complex)
# Optimal set of gates
V3 = 1 / np.sqrt(5) * np.asarray([[1 + 2j, 0],
                                  [0, 1 - 2j]], dtype=complex)


def global_phase(phi):
    return np.exp(phi * 1j)


def RX(phi):
    """
    Rotation around x axis of phi
    Args:
        phi:

    Returns:

    """
    return math.cos(phi / 2) * np.identity(2) - 1j * SX * math.sin(phi / 2)


def RY(phi):
    """
    Rotation around y axis of phi
    Args:
        phi:

    Returns:

    """
    return math.cos(phi / 2) * np.identity(2) - 1j * SY * math.sin(phi / 2)


def RZ(phi):
    """
    Rotation around Z axis of phi
    Args:
        phi:

    Returns:

    """
    return math.cos(phi / 2) * np.identity(2) - 1j * SZ * math.sin(phi / 2)


def RXRY(phi, gamma):
    """
    Dummy gate used as test for continuous azero
    Args:
        gamma:
        phi:

    Returns:

    """

    return (math.cos(phi / 2) * np.identity(2) - 1j * SX * math.sin(phi / 2)) @ (
                math.cos(gamma / 2) * np.identity(2) - 1j * SY * math.sin(gamma / 2))


def U3(a, b, c):
    """
    Unitary matrix U3 from qiskit
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html

    Args:
        a:
        b:
        c:

    Returns:

    """

    U = np.asarray([[math.cos(a / 2), -np.exp(1j * c) * math.sin(a / 2)],
                    [np.exp(1j * b) * math.sin(a / 2), np.exp(1j * (b + c)) * math.cos(a / 2)]])
    return U


def cnot10():
    """

    :return:
    """
    return cnot(2, 0, 1).full()


def not_gate():
    return sigmax()


def identity():
    """
    Single qubit identity
    :return:
    """
    return np.identity(2)


GATES = {"rotationsPI128": [Gate(RX(np.pi / 128)), Gate(RY(np.pi / 128)), Gate(RZ(np.pi / 128)),
                            Gate(RX(-np.pi / 128)), Gate(RY(-np.pi / 128)), Gate(RZ(-np.pi / 128))],
         "rotations": [ParametricGate(RX, [(0, 2 * np.pi)]), ParametricGate(RY, [(0, 2 * np.pi)]),
                       ParametricGate(RZ, [(0, 2 * np.pi)])],
         "RXRY": [ParametricGate(RXRY, [(0, 2 * np.pi), (0, 2 * np.pi)])],
         "RXRYPI128": [ParametricGate(RXRY, [(-np.pi / 128, np.pi / 128), (-np.pi / 128, np.pi / 128)])],
         "RXRYPI128F": [ParametricGate(RXRY, [(-3*np.pi / 128, np.pi / 128), (-3*np.pi / 128, np.pi / 128)])],
         "RXRYPI3128": [ParametricGate(RXRY, [(-3*np.pi / 128, 3*np.pi / 128), (-3*np.pi / 128, 3*np.pi / 128)])],
         "RXRYPI20": [ParametricGate(RXRY, [(-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20)])],
         "RXPI20": [ParametricGate(RX, [(-np.pi / 20, np.pi / 20)])],
         "optimal": [Gate(V1), Gate(V2), Gate(V3)],
         "mixed": [Gate(V1), ParametricGate(RX, [(0, 2 * np.pi)]),
                   ParametricGate(U3, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])],
         "qiskit": [ParametricGate(U3, [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]),
                    Gate(not_gate(), is_controlled=True), Gate(identity())],
         "test_U3": [ParametricGate(RX, [(-np.pi, np.pi)]), ParametricGate(RY, [(-np.pi, np.pi)]), Gate(identity())]}

# Temporary code, used to add a 2qubit gate set swiftly. To remove when the compiler will be able to manage more qubits
gate_extension = []

for gate in GATES["optimal"]:
    gate = gate.full(qt_object=True)
    gate_extension.append(Gate(gate_expand_1toN(gate, 2, 0).full()))
    gate_extension.append(Gate(gate_expand_1toN(gate, 2, 1).full()))

gate_extension.append(Gate(cnot(2, 0, 1).full()))
gate_extension.append(Gate(cnot(2, 1, 0).full()))

# Change dims. I think it has no effects
for gate in gate_extension:
    gate.dims = [[4], [4]]

GATES["optimal2"] = gate_extension

"""

GATES = {"QT_RXD": qt.Qobj(RXD(THETA)),
         "QT_RYD": qt.Qobj(RYD(THETA)),
         "QT_RZD": qt.Qobj(RZD(THETA)),
         "QT_RXS": qt.Qobj(RXS(THETA)),
         "QT_RYS": qt.Qobj(RYS(THETA)),
         "QT_RZS": qt.Qobj(RZS(THETA)),
         "RXD": RXD,
         "RYD": RYD,
         "RZD": RZD,
         "RXS": RXS,
         "RYS": RYS,
         "RZS": RZS,
         "R_ion": R_ion,
         "X": sigmax(),
         "Y": qt.sigmay(),
         "Z": qt.sigmaz(),
         "H": snot(),
         "S": snot()*phasegate(np.pi/4)*snot(),
         "V": phasegate(np.pi/4)*(snot()*phasegate(np.pi/4)*snot()),
         "T": phasegate(np.pi/4),
         "F": qt.Qobj([[ -0.40194 -0.43507j, -0.36803-0.71674j],
                       [ 0.36803-0.71674j,  -0.40194+0.43507j ]]),
         "QT_V1": 1/np.sqrt(5) * qt.Qobj([[ 1, 2j],
                                       [ 2j, 1]]),
         "QT_V2": 1/np.sqrt(5) * qt.Qobj([[ 1, 2],
                                       [ -2, 1]]),
         "QT_V3": 1/np.sqrt(5) * qt.Qobj([[ 1+2j, 0],
                                       [ 0, 1-2j]]),
         "V1": 1/np.sqrt(5) * np.asarray([[ 1, 2j],
                                       [ 2j, 1]]),
         "V2": 1/np.sqrt(5) * np.asarray([[ 1, 2],
                                       [ -2, 1]]),
         "V3": 1/np.sqrt(5) * np.asarray([[ 1+2j, 0],
                                       [ 0, 1-2j]]),
         "fixed": qt.Qobj([[0.76749896-0.43959894*1.j, -0.09607122+0.45658344*1.j],
                                    [0.09607122+0.45658344*1.j, 0.76749896+0.43959894*1.j]]),
         "fixed2x2_10": qt.Qobj([[0.51876777 + 0.50087923j, -0.35777088 + 0.17888544j, -0.39354796 - 0.07155418j, 0.17888544 + 0.35777088j],
                                 [-0.35777088 + 0.17888544j, -0.62609903 + 0.35777088j, 0.17888544 + 0.35777088j, 0.17888544 - 0.35777088j],
                                 [0.39354796 + 0.07155418j, -0.17888544 - 0.35777088j, 0.07155418 - 0.39354796j, 0.08944272 - 0.71554175j],
                                 [-0.17888544 - 0.35777088j, -0.17888544 + 0.35777088j, 0.08944272 - 0.71554175j, 0.35777088 + 0.17888544j]])}

# "fixed2x2_10" -> composition of [5 0 3 6 6 6 6 6 5 0] optimal

BASES_GATES = {"rotations": (GATES["QT_RXD"], GATES["QT_RYD"], GATES["QT_RZD"], GATES["QT_RXS"], GATES["QT_RYS"], GATES["QT_RZS"]),
               "rotationstarget": (GATES["RXD"], GATES["RYD"], GATES["RZD"], GATES["RXS"], GATES["RYS"], GATES["RZS"]),
               "halfrotationstarget": (GATES["RXD"], GATES["RYD"], GATES["RZD"]),
               "rotations_H": (GATES["QT_RXD"], GATES["QT_RYD"], GATES["QT_RZD"], GATES["QT_RXS"], GATES["QT_RYS"], GATES["QT_RZS"], GATES["H"]),
               "rotations2": (GATES["QT_RXD"], GATES["QT_RYD"], GATES["QT_RXS"], GATES["QT_RYS"], GATES["H"], GATES["T"]),
               "Rion": [GATES["R_ion"]],
               "diffusives": (GATES["H"]*GATES["F"],GATES["T"]*GATES["F"]),
               "optimal": (GATES["QT_V2"], GATES["QT_V1"], GATES["QT_V3"]),
               "polak": (GATES["S"], GATES["V"]),
               "discrete": (GATES["X"], GATES["T"], GATES["H"]), #std base. Ho aggiunto x ma non dovrebbe essere necessario
               "IBM": (GATES["X"], GATES["Y"], GATES["Z"], GATES["H"], GATES["T"]),
               "fixed": (GATES["fixed"]),
               "fixed2": (GATES["fixed2x2_10"])
               } #dovrebbe essere T15 e ci dovrebbe essere S

# gate_expand_1toN(U, N, target)
# http://qutip.org/docs/3.1.0/modules/qutip/qip/gates.html

gate_extension = []

for gate in BASES_GATES["optimal"]:
    gate_extension.append(gate_expand_1toN(gate, 2, 0))
    gate_extension.append(gate_expand_1toN(gate, 2, 1))

gate_extension.append(cnot(2, 0, 1))
gate_extension.append(cnot(2, 1, 0))

# Change dims. I think it has no effects
for gate in gate_extension:
    gate.dims = [[4],[4]]

BASES_GATES["optimal2"] = gate_extension

"""
