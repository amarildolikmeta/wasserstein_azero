import numpy as np
import qiskit as qc
import qutip as qt
from qiskit import Aer


from stable_baselines import PPO2

class Solver:
    def __init__(self, n_action=4, random_policy=False):

        self.model = None
        self.random_policy = random_policy
        self.n_action = n_action

    def load_model(self, path):
        self.model = PPO2.load(path)

    def predict_one(self, state):
        assert self.model is not None, "Load PPO model first!"

        if self.random_policy:
            action = np.random.randint(self.n_action)
        else:
            action, _ = self.model.predict(state, deterministic=True)

        pi = np.zeros(self.n_action)
        pi[action] = 1.

        return pi, 1

    def get_experiences(self, env):
        """
        Compile a unitary_target using base_gates
        Args:
            base_gates:
            unitary_target:

        Returns:

        """
        assert self.model is not None, "Load PPO model first!"

        solved = False
        target = None
        actions = None
        info = None

        while not solved:

            done = False
            S = env.reset()
            actions = []
            while not done:
                action, _ = self.model.predict(S, deterministic=True)
                S, reward, done, info = env.step(action)
                actions.append(action)

            solved = info["solved"]
            target = env.target_unitary

        return target, actions

class AzeroSolver():
    def __init__(self, env, tree_maker, brain_producer, brain_params: dict, network_type, pv_loss_ratio):

        self.tree_maker = tree_maker
        self.brain_producer = brain_producer

        self.env = env
        self.tree = tree_maker(self.env)
        self.brain = None

        self.network_type = network_type
        self.lr = 0.01
        self.pv_loss_ratio = pv_loss_ratio
        self.brain_params = brain_params

        self.init_solver()

    def init_solver(self):
        nn_state, _ = self.env.reset()
        obs_dim = nn_state["nn_input"].shape
        act_dim = self.env.n_actions

        self.brain = self.brain_producer(obs_dim, act_dim, self.network_type, self.lr,
                              pv_loss_ratio=self.pv_loss_ratio, use_gpu=False, **self.brain_params)

    def load_model(self, path):
        self.brain.load_weights(path)
        self.tree.set_brain(self.brain)

    def get_experiences(self, depth):
        """
        Compile a unitary_target using base_gates
        Args:
            base_gates:
            unitary_target:

        Returns:

        """

        solved = False
        target = None
        actions = None
        info = None

        while not solved:

            self.tree.reset()
            target = self.env.target_unitary
            done = False
            actions = []
            while not done:
                action, index = self.tree.get_best_action(depth)
                S_, reward, done, info = self.env.step(action)
                self.tree.set_new_root(index, S_)
                actions.append(action)

            solved = info["solved"]
            solved = True

        return target, actions


class QiskitCompiler():
    # https://stackoverflow.com/questions/61790974/qiskit-transpiler-for-quantum-circuits
    def __init__(self, n_qubits, n_parameters, basis_gates):
        self.n_qubits = n_qubits
        self.n_parameters = n_parameters
        self.basis_gates = basis_gates
        self.identity_idx = 2  # the index of the identity action in the GATES list

    def is_qubit_free_u3_cnot(self, action, qubit_idx, target_idx=None):

        '''
        # Check if the target qubit is available
        if action[qubit_idx][-2] != self.identity_idx:
            return False
        '''

        is_available = np.ones(len(action), dtype=bool)
        for idx, qbit in enumerate(action):
            # Check if the qbit is available
            if qbit[-2] != self.identity_idx: is_available[idx] = False
            # Check if the qbit is not a control for a cnot
            if qbit[-2] == 1: is_available[int(qbit[-1])] = False

        if target_idx is None:
            return is_available[qubit_idx]
        else:
            return is_available[qubit_idx] and is_available[target_idx]

    def get_reversed_index(self, idx):

        reversed_index = np.arange(self.n_qubits)[::-1]

        return reversed_index[idx]

    def get_actions_rx_ry(self, actions, transpiled_circuit):

        for depth_idx, instruction in enumerate(transpiled_circuit):

            gate = instruction[0].name

            # qubit_idx è dove agisce il qubit di qiskit
            qubit_idx = self.get_reversed_index(instruction[1][0].index)
            actions[depth_idx][qubit_idx][:self.n_parameters] = instruction[0].params
            # No need to change actions[depth_idx][qubit_idx][-1] since is already set randomly

            if gate == "rx":
                actions[depth_idx][qubit_idx][-2] = 0  # "rx" gate is the first element in GATES
            elif gate == "ry":
                actions[depth_idx][qubit_idx][-2] = 1  # "ry" gate is the second element in GATES
            else:
                print("ERROR: {} not defined".format(gate))

        return actions

    def get_actions_u3_cnot(self, actions, transpiled_circuit):

        depth_idx = 0
        for n, instruction in enumerate(transpiled_circuit):

            gate = instruction[0].name

            if gate == "u3":
                # qubit_idx è dove agisce il qubit di qiskit
                qubit_idx = self.get_reversed_index(instruction[1][0].index)

                if not self.is_qubit_free_u3_cnot(actions[depth_idx], qubit_idx):
                    depth_idx += 1

                actions[depth_idx][qubit_idx][:self.n_parameters] = instruction[0].params
                actions[depth_idx][qubit_idx][-2] = 0  # "U3" gate is the first element in GATES
                # No need to change actions[depth_idx][qubit_idx][-1] since is already set randomly

            elif gate == "cx":
                # In the qc env, the target of the cnot is the index of the action
                # qiskit: instruction[1][0] is the control; instruction[1][1] is the target
                target_idx = self.get_reversed_index(instruction[1][0].index)
                qubit_idx = self.get_reversed_index(instruction[1][1].index)

                if not self.is_qubit_free_u3_cnot(actions[depth_idx], qubit_idx, target_idx):
                    depth_idx += 1

                actions[depth_idx][qubit_idx][-2] = 1  # "cx" gate is the second element in GATES
                actions[depth_idx][qubit_idx][-1] = target_idx

            else:
                print("ERROR: {} not defined".format(gate))

        return actions

    def get_experiences(self, base_gates, unitary_target=None):
        if unitary_target is None:
            unitary_target = qt.rand_unitary_haar(2 ** self.n_qubits)

        # Create the circuit
        circuit = qc.QuantumCircuit(self.n_qubits)
        circuit.unitary(unitary_target, range(self.n_qubits))

        # print(circuit.draw())
        # Compile the circuit
        transpiled_circuit = qc.transpile(circuit, basis_gates=self.basis_gates, optimization_level=2)
        # print(transpiled_circuit.draw())

        # Transform circuit instructions to experiences

        # TODO: extend this procedure to any set of gates
        '''
        IDEA: https://medium.com/arnaldo-gunzi-quantum/how-to-calculate-the-depth-of-a-quantum-circuit-in-qiskit-868505abc104
        guarda come calcolano la profondità del circuito. Puoi tranquillamente conoscere a priori quella che sono i numeri
        di layer del circuito chiamando .depth. Poi ti basta tracciare la profondità come fanno nell'articolo. l'indice
        relativo al layer sarà il max([0,1,2,0,0,2]). 
        '''

        circuit_depth = transpiled_circuit.depth() + 1

        # Create a dummy actions filled with identity gate
        transposed_action = np.random.uniform(low=-1, high=1,
                                              size=(self.n_parameters + 2, self.n_qubits, circuit_depth))
        transposed_action[-2] = np.ones(
            shape=(self.n_qubits, circuit_depth)) * self.identity_idx  # Quale gate usare: identità
        # transposed_action[-1] = np.transpose(np.tile(np.arange(self.n_qubits), (circuit_depth, 1))) # Dove applicarlo: chissenefotte
        transposed_action[-1] = np.random.randint(self.n_qubits, size=(
        self.n_qubits, circuit_depth))  # Dove applicarlo: chissene fotte tanto usi l'indice della azione
        actions = np.transpose(transposed_action)

        # The quantum env generate the circuit in the reversed order, i.e. F @ E @ D @ C @ B @ A @ |psi>, where A is generated before B and so on
        # \reversed_circuit = transpiled_circuit.data[::-1]

        if base_gates == "qiskit":
            actions = self.get_actions_u3_cnot(actions, transpiled_circuit)
        elif base_gates == "test_U3":
            actions = self.get_actions_rx_ry(actions, transpiled_circuit)
        else:
            print("ERROR: {} not implemented yet".format(base_gates))
            return -1

        backend = Aer.get_backend('unitary_simulator')
        job = qc.execute(transpiled_circuit, backend)
        result = job.result()
        B = result.get_unitary(transpiled_circuit, decimals=5)
        B = qt.Qobj(B.data)

        distance = qt.average_gate_fidelity(qt.Qobj(unitary_target.full(), dims=B.dims), B)
        assert np.isclose(distance, 1), distance

        B.full()

        """
        Somewhere, somehow qiskit return reversed order qubits.
        https://quantumcomputing.stackexchange.com/questions/15491/qubit-ordering-in-qiskit
        
        reversed_qubits = []
        for action in actions:
            reversed_qubits.append(action[::-1])
        """

        return B, actions
