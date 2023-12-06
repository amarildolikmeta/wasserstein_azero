import os
import sys
from collections import deque

import matplotlib
import matplotlib.pyplot as plt

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
import torch
from scipy.signal import lfilter
import pickle

from qc_azher import initialize_args
from qc_solver import Solver, QiskitCompiler, AzeroSolver

torch.multiprocessing.set_sharing_strategy('file_system')
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from baseline.Algorithms.Hybrid.AlphaZero.azero_memory import AzeroMemory
from baseline.Algorithms.Hybrid.AlphaZero.azero_tree_continuous import AzeroTreeContinuous as AzeroTree
from baseline.Algorithms.Hybrid.AlphaZero.azero_network_pytorch_cont import AzeroContinuousBrain
from baseline.Algorithms.Hybrid.AlphaZero.azero_network_pytorch_mixed import AzeroMixedBrain
from baseline.Algorithms.Hybrid.AlphaZero.parallel_sampler import ParallelSampler
from baseline.Environments.QuantumCompiling import quantumCompiling
from baseline.Environments.QuantumCompiling.gates import GATES, ParametricGate, Gate
from qc_azher import argument_parser, fix_randomicity, return_env_from_base

"""
Il file è un marasma di funzioni, alcune datate. Funzionano test_model_once() e benchmark_policy().

Benchmark_policy si occupa sia di generare e salvare su file le esperienze sia di addestrare la politica. Ad esempio

python test_model_matrici.py --depth 30 --min_target 0 --max_target 100 --max_lenght_sequence 300 --n_qubits 1 --network_type QC --base_gates RXRYPI128 --pv_loss_ratio 0 --concatenate False

se vuoi cambiare solver devi modificare la flag "solver" definita in benchmark_policy()
"""


def run_episode_given_actions(env, actions, from_qiskit_actions, gym_standard, gamma=0.99):
    """
        Gioca una lista di azioni e restituisce le esperienze da aggiungere alla memoria
        """

    if from_qiskit_actions:
        # normalize actions (qiskit returns parameters not normalized)
        normalized_actions = []
        for action in actions:
            normalized_action = []
            for qbit_action in action:
                gate_idx = int(qbit_action[-2])
                parameters = qbit_action[:-2]
                if env.base_gates[gate_idx].is_parametric:
                    normalized_parameters = env.base_gates[gate_idx].params_to_normalized(parameters).tolist()
                else:
                    normalized_parameters = np.random.uniform(low=-1, high=1, size=len(parameters)).tolist()
                normalized_parameters.extend(qbit_action[len(parameters):])

                normalized_action.append(normalized_parameters)
            normalized_actions.append(normalized_action)
    else:
        normalized_actions = actions

    if gym_standard:
        S = env.reset()
    else:
        S, _ = env.reset()

    memories = deque()
    experiences = []
    ep_rewards = []
    info = {}
    for action in normalized_actions:
        S = env.get_S()
        S_, reward, done, info = env.step(action, normalized_params=True)
        P = 1.
        if gym_standard:
            experiences.append((S, P, action))
        else:
            experiences.append((S["nn_input"], P, action))
        ep_rewards.append(reward)

    r = ep_rewards[::-1]
    a = [1, -gamma]
    b = [1]
    y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
    values = y[::-1]

    memories.extend([(S, P, a, v) for (S, P, a), v in zip(experiences, values)])
    return memories, info


def run_episode(env, pi, deterministic=True, continuous_policy=True, tree_policy=False, args=None):
    S, _ = env.reset()
    done = False
    info = {}
    while not done:
        # print("INIZIO:")
        if tree_policy:
            action, index = pi.get_best_action(depth=args["depth"])
        elif continuous_policy:
            action, _ = pi.predict_one(S['nn_input'])
        else:
            P, _ = pi.predict_one(S['nn_input'])
            if deterministic:
                action = np.argmax(P)
            else:
                P = np.array(P)
                P = P / np.sum(P)
                action = np.random.choice(len(P), p=P)
        S_, reward, done, info = env.step(action, normalized_params=True)
        if tree_policy:
            pi.set_new_root(index, S_)
        if done: info = info

    return info


def run_episode_random(env, args, gamma=0.99):
    """
    Gioca a caso e restituisce quello che l'azione (randomica) che è stata scelta come soluzione
    """
    S, _ = env.reset()
    np.random.seed()

    max_lenght = args["max_lenght_sequence"]
    n_action = args["act_dim"]

    new_goal = None
    actions_performed = []

    for _ in range(np.random.randint(5, max_lenght)):

        action = np.random.randint(n_action)
        actions_performed.append(action)

        new_goal, reward, done, info = env.step(action)

        if done: break

    new_goal = env.current_state.copy()
    S, _ = env.reset()
    env.target_unitary = new_goal

    memories = deque()
    experiences = []
    ep_rewards = []
    info = {}
    for action in actions_performed:

        P = np.zeros(n_action)
        P[action] = 1.

        S_, reward, done, info = env.step(action)
        S = S["nn_input"].copy()
        experiences.append((S, P))
        ep_rewards.append(reward)
        S = S_
        if done: info = info

    r = ep_rewards[::-1]
    a = [1, -gamma]
    b = [1]
    y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
    values = y[::-1]

    memories.extend([(S, P, v) for (S, P), v in zip(experiences, values)])
    return memories, info


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("loss_test_model_matrici.png")
    plt.close()


def plot_loss_and_val(histories, validations):
    loss = []
    for history in histories:
        loss += history.history['loss']
    plt.plot(loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("loss_test_model_matrici.png")

    print_target_gate = False
    if len(validations[0]) > 2:
        print_target_gate = True

    # plot r2
    r2s = []
    acc_gates = []
    acc_targets = []
    r2_value = []
    for validation in validations:
        if print_target_gate:
            r2s.append(validation[0])
            r2_value.append(validation[3])
            acc_gates.append(validation[1])
            acc_targets.append(validation[2])
        else:
            r2s.append(validation[0])
            r2_value.append(validation[1])

    r2s = np.array(r2s)
    r2_value = np.array(r2_value)
    acc_gates = np.array(acc_gates)
    acc_targets = np.array(acc_targets)

    for q in range(len(r2s[0])):
        plt.close()
        plt.plot(r2s[:, q])
        plt.title('r2')
        plt.xlabel('epoch')
        plt.savefig("r2_q" + str(q) + ".png")
        plt.close()
        if print_target_gate:
            plt.plot(acc_gates[:, q])
            plt.plot(acc_targets[:, q])
            plt.title('Accuracy')
            plt.xlabel('Acc')
            plt.xlabel('epoch')
            plt.legend(["acc_gates", "acc_targets"], loc='upper left')
            plt.savefig("accuracy_q" + str(q) + ".png")
    plt.close()
    plt.plot(r2_value)
    plt.title('r2 value')
    plt.xlabel('epoch')
    plt.savefig("r2_value.png")


def print_results(info_results):
    for key, value in info_results.items():
        print("{}: {}".format(key, value))


def test_solver_policy_azero():
    num_episodes = 100
    validation_episodes = 100
    training_epochs = 100

    info_results = {}

    args = initialize_args(30)

    args["min_target"] = 5
    args["max_target"] = 100
    args["max_lenght_sequence"] = 100
    args["network_type"] = "QC"

    def env_maker():
        env = quantumCompiling.Environment(args["base_gates"], args["reward"], args["tollerance"], args["target_gates"],
                                           args["max_lenght_sequence"],
                                           args["min_target"], args["max_target"], norm=args["norm"],
                                           target=args["target"])
        return env

    env = env_maker()
    nn_state, _ = env.reset()
    args["obs_dim"] = nn_state["nn_input"].shape
    args["act_dim"] = env.n_actions

    def tree_maker(env):
        return AzeroTree(env, args, verbose=args["verbose"])

    sampler_params = dict(
        n_workers=min(args['max_processes'], validation_episodes),
        seed=args['random_seed'] + 4000,
    )

    sampler = ParallelSampler(make_env=env_maker,
                              make_tree=tree_maker,
                              **sampler_params)

    solver = Solver(n_action=env.n_actions, random_policy=False)
    solver.load_model("PPO_Model/PPO_best_model.pkl")

    memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    # Fill the memory and benchmarch the Solver
    results, solver_ep_solved, solver_ep_length = [], [], []
    for i in range(num_episodes):
        # res, info = run_episode(env, solver, gamma=0.99)
        res, info = run_episode_random(env, args, gamma=0.99)
        results.append(res)
        solver_ep_solved.append(info["solved"])
        solver_ep_length.append(info["lenght"])

    info_results["solver_ep_solved"] = np.mean(solver_ep_solved)
    info_results["solver_ep_length"] = np.mean(solver_ep_length)
    info_results["solver_ep_std_length"] = np.std(solver_ep_length)
    info_results["solver_ep_max_length"] = np.max(solver_ep_length)

    experiences = [item for sublist in results for item in sublist]
    memory.add_batch(experiences)

    nn_state, _ = env.reset()
    obs_dim = nn_state["nn_input"].shape
    act_dim = env.n_actions

    # Initialize the AZ policy
    current_brain = AzeroBrain(obs_dim, act_dim, args["network_type"], args["lr"])

    # Train the AZ policy
    with tf.device(tf.test.gpu_device_name()):  # device_lib.list_local_devices()[-1].name  "/GPU:0"
        history = current_brain.train(memory.old_sample(1, uniform_sampling=True), training_epochs, args['batch_size'],
                                      stopping=True,
                                      verbose=2)
    # Plot the training loss
    plot_loss(history)

    # Benchmarch the AZ policy
    evals, policy_ep_solved, policy_ep_length = [], [], []
    for j in range(validation_episodes):
        res, info = run_episode(env, current_brain, gamma=0.99)
        evals.append(res[0][-1])
        policy_ep_solved.append(info["solved"])
        policy_ep_length.append(info["lenght"])

    info_results["policy_ep_solved"] = np.mean(policy_ep_solved)
    info_results["policy_ep_length"] = np.mean(policy_ep_length)
    info_results["policy_ep_std_length"] = np.std(policy_ep_length)
    info_results["policy_ep_max_length"] = np.max(policy_ep_length)

    # Benchmark AZher (policy + search)
    infos = sampler.collect(current_brain.get_weights().copy(), n_episodes=validation_episodes, eval=True)

    if sampler is not None: sampler.close()

    AZ_ep_solved = deque()
    AZ_ep_length = deque()

    for info in infos:
        AZ_ep_solved.append(info["solved"])
        AZ_ep_length.append(info["length"])

    info_results["AZ_ep_solved"] = np.mean(AZ_ep_solved)
    info_results["AZ_ep_length"] = np.mean(AZ_ep_length)
    info_results["AZ_ep_std_length"] = np.std(AZ_ep_length)
    info_results["AZ_ep_max_length"] = np.max(AZ_ep_length)

    print_results(info_results)

    current_brain.save("saved_model.h5")


def test_solver_azero():
    num_episodes = 100
    validation_episodes = 5
    training_epochs = 300

    info_results = {}

    args = initialize_args(30)
    args["min_target"] = 5
    args["max_target"] = 10
    args["max_lenght_sequence"] = 10
    args["depth"] = 40
    args["reward"] = 3
    args["n_stochastic_depth"] = 0
    args["base_gates"] = "optimal2"
    args["target_gates"] = "optimal2"
    args["network_type"] = "QC2"
    args["batch_size"] = 128

    def env_maker():
        env = quantumCompiling.Environment(args["base_gates"], args["reward"], args["tollerance"], args["target_gates"],
                                           args["max_lenght_sequence"],
                                           args["min_target"], args["max_target"], norm=args["norm"],
                                           target=args["target"])
        return env

    env = env_maker()
    nn_state, _ = env.reset()
    args["obs_dim"] = nn_state["nn_input"].shape
    args["act_dim"] = env.n_actions

    def tree_maker(env):
        return AzeroTree(env, args, verbose=args["verbose"])

    sampler_params = dict(
        n_workers=min(args['max_processes'], validation_episodes),
        seed=args['random_seed'] + 4000,
    )

    sampler = ParallelSampler(make_env=env_maker,
                              make_tree=tree_maker,
                              **sampler_params)

    memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    # Fill the memory and benchmarch the Solver
    results, solver_ep_solved, solver_ep_length = [], [], []
    for i in range(num_episodes):
        res, info = run_episode_random(env, args, gamma=0.99)
        results.append(res)
        solver_ep_solved.append(info["solved"])
        solver_ep_length.append(info["lenght"])

    info_results["solver_ep_solved"] = np.mean(solver_ep_solved)
    info_results["solver_ep_length"] = np.mean(solver_ep_length)
    info_results["solver_ep_std_length"] = np.std(solver_ep_length)
    info_results["solver_ep_max_length"] = np.max(solver_ep_length)

    experiences = [item for sublist in results for item in sublist]
    memory.add_batch(experiences)

    nn_state, _ = env.reset()
    obs_dim = nn_state["nn_input"].shape
    act_dim = env.n_actions

    # Initialize the AZ policy
    current_brain = AzeroBrain(obs_dim, act_dim, args["network_type"], args["lr"])

    # Train the AZ policy
    with tf.device(tf.test.gpu_device_name()):  # device_lib.list_local_devices()[-1].name  "/GPU:0"
        history = current_brain.train(memory.old_sample(1, uniform_sampling=True), training_epochs, args['batch_size'],
                                      stopping=True,
                                      verbose=2)
    # Plot the training loss
    plot_loss(history)

    # Benchmarch the AZ policy
    evals, policy_ep_solved, policy_ep_length = [], [], []
    for j in range(validation_episodes):
        res, info = run_episode(env, current_brain, gamma=0.99)
        evals.append(res[0][-1])
        policy_ep_solved.append(info["solved"])
        policy_ep_length.append(info["lenght"])

    info_results["policy_ep_solved"] = np.mean(policy_ep_solved)
    info_results["policy_ep_length"] = np.mean(policy_ep_length)
    info_results["policy_ep_std_length"] = np.std(policy_ep_length)
    info_results["policy_ep_max_length"] = np.max(policy_ep_length)

    # Benchmark AZher (policy + search)
    infos = sampler.collect(current_brain.get_weights().copy(), n_episodes=validation_episodes, eval=True)

    if sampler is not None: sampler.close()

    AZ_ep_solved = deque()
    AZ_ep_length = deque()

    for info in infos:
        AZ_ep_solved.append(info["solved"])
        AZ_ep_length.append(info["length"])

    info_results["AZ_ep_solved"] = np.mean(AZ_ep_solved)
    info_results["AZ_ep_length"] = np.mean(AZ_ep_length)
    info_results["AZ_ep_std_length"] = np.std(AZ_ep_length)
    info_results["AZ_ep_max_length"] = np.max(AZ_ep_length)

    print_results(info_results)

    current_brain.save("saved_model.h5")


def initialize_args():
    CLI = argument_parser()

    args = CLI.parse_args()
    args = vars(args)

    # Brutta soluzione ma funziona
    if args["target"] == "True":
        args["target"] = True
    elif args["target"] == "None":
        args["target"] = None

    args["note"] = "Da qui in poi questi paramentri NON sono modificabili da linea di comando"
    args["single_player"] = True

    return args

def test_model_once():
    """
    Load policy weights and test the model once.
    Returns:

    """

    args = initialize_args()
    args = fix_randomicity(args)

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    def env_maker():
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=args["target"],
                          concatenate_input=args["concatenate"])
        return env

    def tree_maker(env):
        return AzeroTree(env, args, verbose=args["verbose"])

    args["brain_params"] = {
        "std": args["std"],
        "entropy_coef": args["entropy_coef"]
    }

    env = env_maker()

    azSolver = AzeroSolver(env=env, tree_maker=tree_maker, brain_params=args["brain_params"],
                           brain_producer=args["brain_producer"], network_type=args["network_type"],
                           pv_loss_ratio=args["pv_loss_ratio"])

    azSolver.load_model("0_current_model.h5")

    done = False
    idx = 0
    while not done:
        action, index = azSolver.tree.get_best_action(args["depth"])
        S_, reward, done, info = env.step(action)
        azSolver.tree.print_infos()
        # tree.render("timestep_{}.pdf".format(idx))
        azSolver.tree.set_new_root(index, S_)
        idx += 1
    print("solved: ", info["solved"])

def generate_examples_qiskit(args, tmp_path, num_episodes = 100, validation_episodes = 50):
    """
    Collect experiences by using qiskit compiler or a trained azero model or a PPO trained model.
    """

    save_path = "Test_supervised/Experiences/Qiskit/"
    os.makedirs(save_path, exist_ok=True)

    n_parameters = None
    basis_gates = None
    # Qiskit Solver
    if args["base_gates"] == "test_U3":
        n_parameters = 1
        basis_gates = ['rx', 'ry']
    if args["base_gates"] == "qiskit":
        n_parameters = 3
        basis_gates = ['u3', 'cx']

    # target = qt.Qobj(U3Gate(0, 0, -np.pi/2).to_matrix())
    # not_gate = qt.qip.operations.gates.x_gate(N=2, target=2)
    # target = gate_expand_1toN(target, 2, target=0)

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                      min_target=args["min_target"], max_target=args["max_target"],
                      max_length_circuit=args["max_lenght_sequence"],
                      epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                      gym_standard=args["gym_standard"],
                      concatenate_input=args["concatenate"],
                      target=None)

    qk_compiler = QiskitCompiler(args["n_qubits"], n_parameters=n_parameters, basis_gates=basis_gates)

    memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    for episode in range(num_episodes):
        print("episode {}".format(episode))

        target, actions = qk_compiler.get_experiences(base_gates=args["base_gates"])

        if not isinstance(target, np.ndarray):
            target = target.full()

        """
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=target,
                          concatenate_input=args["concatenate"])
        """

        # Fix the unitary target
        env.target_unitary = target
        env.fixed_target = True

        # run episode following action
        experiences, info = run_episode_given_actions(env, actions, from_qiskit_actions=True,
                                                      gym_standard=False, gamma=args["gamma"])

        # Load the memory
        memory.add_batch(experiences)

    # Save memory to file
    with open('{}{}_memory.obj'.format(save_path, tmp_path), 'wb') as fp:
        pickle.dump(memory, fp)

    # 2) collect validation episodes
    val_memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    for episode in range(validation_episodes):
        print("validation episode {}".format(episode))

        target, actions = qk_compiler.get_experiences(base_gates=args["base_gates"])

        if not isinstance(target, np.ndarray):
            target = target.full()

        """
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=target,
                          concatenate_input=args["concatenate"])
        """

        # Fix the unitary target
        env.target_unitary = target
        env.fixed_target = True

        # run episode following action
        experiences, info = run_episode_given_actions(env, actions, gamma=args["gamma"],
                                                      gym_standard=False,
                                                      from_qiskit_actions=True)

        # Load the memory
        val_memory.add_batch(experiences)

    with open('{}{}_validation_memory.obj'.format(save_path, tmp_path), 'wb') as fp:
        pickle.dump(val_memory, fp)

def generate_examples_azero(args, tmp_path, num_episodes, validation_episodes, depth, path_model):
    """
    Collect experiences by using qiskit compiler or a trained azero model or a PPO trained model.
    """

    save_path = "Test_supervised/Experiences/Azero/"
    os.makedirs(save_path, exist_ok=True)

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                      min_target=args["min_target"], max_target=args["max_target"],
                      max_length_circuit=args["max_lenght_sequence"],
                      epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                      gym_standard=args["gym_standard"],
                      concatenate_input=args["concatenate"],
                      target=None)

    def tree_maker(env):
        return AzeroTree(env, args, verbose=args["verbose"])

    args["brain_params"] = {
        "std": args["std"],
        "entropy_coef": args["entropy_coef"]
    }

    azSolver = AzeroSolver(env=env, tree_maker=tree_maker, brain_params=args["brain_params"],
                           brain_producer=args["brain_producer"], network_type=args["network_type"],
                           pv_loss_ratio=args["pv_loss_ratio"])

    # Load the model
    azSolver.load_model(path_model)

    memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    for episode in range(num_episodes):
        print("episode {}".format(episode))

        target, actions = azSolver.get_experiences(depth=depth)

        if not isinstance(target, np.ndarray):
            target = target.full()

        """
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=target,
                          concatenate_input=args["concatenate"])
        """

        # Fix the unitary target
        env.target_unitary = target
        env.fixed_target = True

        # run episode following action
        experiences, info = run_episode_given_actions(env, actions, from_qiskit_actions=False,
                                                      gym_standard=False, gamma=args["gamma"])

        # Load the memory
        memory.add_batch(experiences)

    # Save memory to file
    with open('{}{}_memory.obj'.format(save_path, tmp_path), 'wb') as fp:
        pickle.dump(memory, fp)

    # 2) collect validation episodes
    val_memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    for episode in range(validation_episodes):
        print("validation episode {}".format(episode))

        target, actions = azSolver.get_experiences(depth=depth)

        if not isinstance(target, np.ndarray):
            target = target.full()

        """
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=target,
                          concatenate_input=args["concatenate"])
        """

        # Fix the unitary target
        env.target_unitary = target
        env.fixed_target = True

        # run episode following action
        experiences, info = run_episode_given_actions(env, actions, gamma=args["gamma"],
                                                      gym_standard=False,
                                                      from_qiskit_actions=False)

        # Load the memory
        val_memory.add_batch(experiences)

    with open('{}{}_validation_memory.obj'.format(save_path, tmp_path), 'wb') as fp:
        pickle.dump(val_memory, fp)

def generate_examples_PPO(args, tmp_path, num_episodes, validation_episodes, path_model):
    """
    Collect experiences by using qiskit compiler or a trained azero model or a PPO trained model.
    """

    save_path = "Test_supervised/Experiences/PPO/"
    os.makedirs(save_path, exist_ok=True)

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                      min_target=args["min_target"], max_target=args["max_target"],
                      max_length_circuit=args["max_lenght_sequence"],
                      epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                      gym_standard=True,
                      no_glob_phase=False,
                      concatenate_input=args["concatenate"],
                      target=None)

    solver = Solver(n_action=env.n_actions, random_policy=False)
    solver.load_model(path_model)

    memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    for episode in range(num_episodes):
        print("episode {}".format(episode))

        target, actions = solver.get_experiences(env)

        if not isinstance(target, np.ndarray):
            target = target.full()

        """
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=target,
                          concatenate_input=args["concatenate"])
        """

        # Fix the unitary target
        env.target_unitary = target
        env.fixed_target = True

        # run episode following action
        experiences, info = run_episode_given_actions(env, actions, from_qiskit_actions=False,
                                                      gym_standard=True, gamma=args["gamma"])

        # Load the memory
        memory.add_batch(experiences)

    # Save memory to file
    with open('{}{}_memory.obj'.format(save_path, tmp_path), 'wb') as fp:
        pickle.dump(memory, fp)

    # 2) collect validation episodes
    val_memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

    for episode in range(validation_episodes):
        print("validation episode {}".format(episode))

        target, actions = solver.get_experiences(env)

        if not isinstance(target, np.ndarray):
            target = target.full()

        """
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=target,
                          concatenate_input=args["concatenate"])
        """

        # Fix the unitary target
        env.target_unitary = target
        env.fixed_target = True

        # run episode following action
        experiences, info = run_episode_given_actions(env, actions, gamma=args["gamma"],
                                                      gym_standard=True,
                                                      from_qiskit_actions=False)

        # Load the memory
        val_memory.add_batch(experiences)

    with open('{}{}_validation_memory.obj'.format(save_path, tmp_path), 'wb') as fp:
        pickle.dump(val_memory, fp)

def train_policy(args, env, training_epochs, val_memory, memory):

    if args["gym_standard"]:
        nn_state = env.reset()
        args["obs_dim"] = nn_state.shape
    else:
        nn_state, _ = env.reset()
        args["obs_dim"] = nn_state["nn_input"].shape
    args["act_dim"] = env.n_actions

    args["brain_params"] = {
        "std": args["std"],
        "entropy_coef": args["entropy_coef"]
    }

    # Initialize the AZ policy
    current_brain = args['brain_producer'](args["obs_dim"], args["act_dim"], args["network_type"], args["lr"],
                                           pv_loss_ratio=args["pv_loss_ratio"], use_gpu=not args['no_gpu']
                                           , **args["brain_params"])

    # Train the AZ policy
    histories = []
    validations = []
    for i in range(training_epochs):
        print("Epoch:", i)
        validation = current_brain.validate(val_memory.old_sample(1), deterministic=True)
        validations.append(validation)
        if len(validation) > 2:
            print("R2s:", validation[0])
            print("Accuracy Gates:", validation[1])
            print("Accuracy Targets:", validation[2])
            print("R2 value:", validation[3])
        else:
            print("R2s:", validation[0])
            print("R2 value:", validation[1])
        history = current_brain.train(memory.old_sample(1), 1, args["batch_size"], stopping=False)
        histories.append(history)

    validation = current_brain.validate(val_memory.old_sample(1), deterministic=True)
    validations.append(validation)
    if len(validation) > 2:
        print("VALIDATION R2s:", validation[0])
        print("VALIDATION Accuracy Gates:", validation[1])
        print("VALIDATION Accuracy Targets:", validation[2])
        print("VALIDATION R2 value:", validation[3])
    else:
        print("VALIDATION R2s:", validation[0])
        print("VALIDATION R2 value:", validation[1])
    # Plot the training loss
    plot_loss_and_val(histories, validations)

    return current_brain

def train_policy_old():
    """
    Collect experiences by using qiskit compiler and train azero policy. quantumCompiling.Compiler and AzeroMixedBrain must
    be used here. The reason is that you have to choose between more than one gate to apply.
    Returns:

    """

    training_epochs = 100
    num_episodes = 100
    validation_episodes = 50

    # Select the solver (it generates samples)
    solver = "qiskit"  # or "PPO" or "azero"
    solver = "azero"  # or "PPO" or "azero"
    solver = "PPO"  # or "PPO" or "azero"

    args = initialize_args()
    args = fix_randomicity(args)

    if args["base_gates"] == "test_U3":
        n_parameters = 1
        basis_gates = ['rx', 'ry']
        assert solver != "PPO", "cannot solve {} using PPO".format(basis_gates)
    if args["base_gates"] == "qiskit":
        n_parameters = 3
        basis_gates = ['u3', 'cx']
        assert solver != "PPO", "cannot solve {} using PPO".format(basis_gates)

    if solver == "PPO":
        args["gym_standard"] = True

    # target = qt.Qobj(U3Gate(0, 0, -np.pi/2).to_matrix())
    # not_gate = qt.qip.operations.gates.x_gate(N=2, target=2)
    # target = gate_expand_1toN(target, 2, target=0)

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                      min_target=args["min_target"], max_target=args["max_target"],
                      max_length_circuit=args["max_lenght_sequence"],
                      epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                      gym_standard=args["gym_standard"],
                      concatenate_input=args["concatenate"],
                      target=None)

    if solver == "qiskit":
        qk_compiler = QiskitCompiler(args["n_qubits"], n_parameters=n_parameters, basis_gates=basis_gates)
        from_qiskit_actions = True
    elif solver == "azero":

        def tree_maker(env):
            return AzeroTree(env, args, verbose=args["verbose"])

        args["brain_params"] = {
            "std": args["std"],
            "entropy_coef": args["entropy_coef"]
        }

        azSolver = AzeroSolver(env=env, tree_maker=tree_maker, brain_params=args["brain_params"],
                               brain_producer=args["brain_producer"], network_type=args["network_type"],
                               pv_loss_ratio=args["pv_loss_ratio"])

        azSolver.load_model("0_current_model.h5")
        from_qiskit_actions = False

    else:
        solver = Solver(n_action=env.n_actions, random_policy=False)
        solver.load_model("Modelli/PPO/best_RXRYPI128_concatFalse.zip")
        from_qiskit_actions = False

    try:
        with open('memory.obj', 'rb') as fp:
            memory = pickle.load(fp)
    except:
        memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

        for episode in range(num_episodes):
            print("episode {}".format(episode))

            if solver == "qiskit":
                target, actions = qk_compiler.get_experiences(base_gates=args["base_gates"])
            elif solver == "azero":
                target, actions = azSolver.get_experiences(depth=10)
            else:
                target, actions = solver.get_experiences(env)

            if not isinstance(target, np.ndarray):
                target = target.full()

            env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                              target_gates=args["target_gates"],
                              min_target=args["min_target"], max_target=args["max_target"],
                              max_length_circuit=args["max_lenght_sequence"],
                              epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                              gym_standard=args["gym_standard"],
                              target=target,
                              concatenate_input=args["concatenate"])

            # run episode following action
            experiences, info = run_episode_given_actions(env, actions, from_qiskit_actions=from_qiskit_actions, gym_standard=args["gym_standard"], gamma=args["gamma"])

            # Load the memory
            memory.add_batch(experiences)

        with open('memory.obj', 'wb') as fp:
            pickle.dump(memory, fp)

    # 2) collect validation episodes
    try:
        with open('val_memory.obj', 'rb') as fp:
            val_memory = pickle.load(fp)
    except:
        val_memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)

        for episode in range(validation_episodes):
            print("validation episode {}".format(episode))

            if solver == "qiskit":
                target, actions = qk_compiler.get_experiences(base_gates=args["base_gates"])
            elif solver == "azero":
                target, actions = azSolver.get_experiences(depth=10)
            else:
                target, actions = solver.get_experiences(env)

            if not isinstance(target, np.ndarray):
                target = target.full()

            env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                              target_gates=args["target_gates"],
                              min_target=args["min_target"], max_target=args["max_target"],
                              max_length_circuit=args["max_lenght_sequence"],
                              epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                              gym_standard=args["gym_standard"],
                              target=target,
                              concatenate_input=args["concatenate"])

            # run episode following action
            experiences, info = run_episode_given_actions(env, actions, gamma=args["gamma"], gym_standard=args["gym_standard"], from_qiskit_actions=from_qiskit_actions)

            # Load the memory
            val_memory.add_batch(experiences)
        with open('val_memory.obj', 'wb') as fp:
            pickle.dump(val_memory, fp)


    # 3) train azero policy
    args["brain_params"] = {
        "std": args["std"],
        "entropy_coef": args["entropy_coef"]
    }

    if args["gym_standard"]:
        nn_state = env.reset()
        args["obs_dim"] = nn_state.shape
    else:
        nn_state, _ = env.reset()
        args["obs_dim"] = nn_state["nn_input"].shape
    args["act_dim"] = env.n_actions

    # Initialize the AZ policy
    current_brain = args['brain_producer'](args["obs_dim"], args["act_dim"], args["network_type"], args["lr"],
                                           pv_loss_ratio=args["pv_loss_ratio"], use_gpu=not args['no_gpu']
                                           , **args["brain_params"])

    # Train the AZ policy
    histories = []
    validations = []
    for i in range(training_epochs):
        print("Epoch:", i)
        validation = current_brain.validate(val_memory.old_sample(1), deterministic=True)
        validations.append(validation)
        if len(validation) > 2:
            print("R2s:", validation[0])
            print("Accuracy Gates:", validation[1])
            print("Accuracy Targets:", validation[2])
            print("R2 value:", validation[3])
        else:
            print("R2s:", validation[0])
            print("R2 value:", validation[1])
        history = current_brain.train(memory.old_sample(1), 1, args["batch_size"], stopping=False)
        histories.append(history)

    validation = current_brain.validate(val_memory.old_sample(1), deterministic=True)
    validations.append(validation)
    if len(validation) > 2:
        print("VALIDATION R2s:", validation[0])
        print("VALIDATION Accuracy Gates:", validation[1])
        print("VALIDATION Accuracy Targets:", validation[2])
        print("VALIDATION R2 value:", validation[3])
    else:
        print("VALIDATION R2s:", validation[0])
        print("VALIDATION R2 value:", validation[1])
    # Plot the training loss
    plot_loss_and_val(histories, validations)

    # 4) Benchmarch the AZ policy

    if args["gym_standard"]:
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"],
                          target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=False,
                          target=args["target"],
                          no_glob_phase=False,
                          concatenate_input=args["concatenate"])

        args["gym_standard"] = False

    info_results = {}
    policy_ep_solved, policy_ep_length = [], []
    for j in range(validation_episodes):
        info = run_episode(env, current_brain, continuous_policy=True)
        policy_ep_solved.append(info["solved"])
        policy_ep_length.append(info["length"])

    info_results["policy_ep_solved"] = np.mean(policy_ep_solved)
    info_results["policy_ep_length"] = np.mean(policy_ep_length)
    info_results["policy_ep_std_length"] = np.std(policy_ep_length)
    info_results["policy_ep_max_length"] = np.max(policy_ep_length)

    print_results(info_results)

    if args["base_gates"] == "test_U3": return 0

    # 5) Benchmarch the AZ-tree policy
    def tree_maker(env):
        return AzeroTree(env, args, verbose=args["verbose"])

    env.reset()
    tree = tree_maker(env)
    tree.set_brain(current_brain)
    info_results = {}
    policy_ep_solved, policy_ep_length = [], []

    for j in range(validation_episodes):
        tree.reset()
        info = run_episode(env, tree, args=args, tree_policy=True)
        policy_ep_solved.append(info["solved"])
        policy_ep_length.append(info["length"])

    info_results["policy_ep_solved"] = np.mean(policy_ep_solved)
    info_results["policy_ep_length"] = np.mean(policy_ep_length)
    info_results["policy_ep_std_length"] = np.std(policy_ep_length)
    info_results["policy_ep_max_length"] = np.max(policy_ep_length)

    print_results(info_results)

def benchmark_policy():

    args = initialize_args()
    args = fix_randomicity(args)

    training_epochs = 200
    num_episodes = 100
    validation_episodes = 50

    # If you want to use specific experiences
    memory_path = None
    val_path = None

    # Select the solver (it generates samples)
    solver = "qiskit"  # or "PPO" or "azero"
    solver = "azero"  # or "PPO" or "azero"
    solver = "PPO"  # or "PPO" or "azero"

    # tmp string used to load memories automatically
    tmp_path = "{}_{}_{}".format(num_episodes, validation_episodes, args["concatenate"])

    if solver == "qiskit":
        if memory_path is None and val_path is None:
            save_path = "Test_supervised/Experiences/Qiskit/"
            memory_path = '{}{}_memory.obj'.format(save_path, tmp_path)
            val_path = '{}{}_validation_memory.obj'.format(save_path, tmp_path)

        if not (os.path.exists(memory_path) and os.path.exists(val_path)):
            generate_examples_qiskit(args=args, tmp_path=tmp_path, num_episodes=num_episodes, validation_episodes=validation_episodes)

    if solver == "azero":
        if memory_path is None and val_path is None:
            save_path = "Test_supervised/Experiences/Azero/"
            memory_path = '{}{}_memory.obj'.format(save_path, tmp_path)
            val_path = '{}{}_validation_memory.obj'.format(save_path, tmp_path)

        if not (os.path.exists(memory_path) and os.path.exists(val_path)):
            generate_examples_azero(args=args, tmp_path=tmp_path, num_episodes=num_episodes, validation_episodes=validation_episodes, depth=10, path_model="0_current_model.h5")
    if solver == "PPO":
        if memory_path is None and val_path is None:
            save_path = "Test_supervised/Experiences/PPO/"
            memory_path = '{}{}_memory.obj'.format(save_path, tmp_path)
            val_path = '{}{}_validation_memory.obj'.format(save_path, tmp_path)

        if not (os.path.exists(memory_path) and os.path.exists(val_path)):
            generate_examples_PPO(args=args, tmp_path=tmp_path, num_episodes=num_episodes, validation_episodes=validation_episodes,
                              path_model="Modelli/PPO/best_RXRYPI3128_concat{}".format(args["concatenate"]))

    with open(val_path, 'rb') as f:
        val_memory = pickle.load(f)
    with open(memory_path, 'rb') as f:
        memory = pickle.load(f)

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                      min_target=args["min_target"], max_target=args["max_target"],
                      max_length_circuit=args["max_lenght_sequence"],
                      epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                      gym_standard=args["gym_standard"],
                      concatenate_input=args["concatenate"],
                      no_glob_phase=False,
                      target=None)

    trained_policy = train_policy(args, env=env, training_epochs=training_epochs, val_memory=val_memory, memory=memory)

    env.reset()
    info_results = {}
    policy_ep_solved, policy_ep_length = [], []

    for j in range(validation_episodes):
        info = run_episode(env, trained_policy, args=args, tree_policy=False)
        policy_ep_solved.append(info["solved"])
        policy_ep_length.append(info["length"])

    info_results["policy_ep_solved"] = np.mean(policy_ep_solved)
    info_results["policy_ep_length"] = np.mean(policy_ep_length)
    info_results["policy_ep_std_length"] = np.std(policy_ep_length)
    info_results["policy_ep_max_length"] = np.max(policy_ep_length)

    print_results(info_results)


# if __name__ == '__main__': test_solver_policy_azero()
# if __name__ == '__main__': test_solver_azero()
#if __name__ == '__main__': test_model_once()
if __name__ == '__main__': benchmark_policy()
