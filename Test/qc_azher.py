import argparse
import os
import sys
import time
import logging
from os import makedirs
from os.path import exists
from random import seed as random_seed

import torch.multiprocessing
from numpy.random import seed as np_seed
from torch import manual_seed
from torch.multiprocessing import cpu_count
# torch.multiprocessing.set_start_method('spawn', force=True)
# torch.multiprocessing.set_sharing_strategy('file_system')
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from baseline.Algorithms.Hybrid.AlphaZero.alphazero import Azero
from baseline.Algorithms.Hybrid.AlphaZero.azero_tree_continuous import AzeroTreeContinuous as AzeroTree
from baseline.Algorithms.Hybrid.AlphaZero.parallel_sampler import ParallelSampler
from baseline.Environments.QuantumCompiling import quantumCompiling
from baseline.Environments.QuantumCompiling.gates import GATES, ParametricGate, Gate
from baseline.Algorithms.Hybrid.AlphaZero.callbacks import SaveInfo, CheckpointModel, TestModel
from baseline.Algorithms.Hybrid.AlphaZero.azero_network_pytorch_cont import AzeroContinuousBrain
from baseline.Algorithms.Hybrid.AlphaZero.azero_network_pytorch_mixed import AzeroMixedBrain

"""
Ricorda che i parametri:

    --min_target: se impostato a 0 ignora i parametri "max_target" e "target_gates".

> Parametri di esempio per AzeroHER con environment full continuous

    python qc_azher.py --base_gates RXRYPI128 --target_gates RXRYPI128 --max_lenght_sequence 300  --n_qubits 1 
                       --depth 120 --lr 5e-05 --min_target 0
                       
> Parametri di esempio per AzeroHER con environment misto (azioni continue e discrete)

    Per questo environment, le azioni sono una lista (n, m, q), dove n sono n numeri reali che corrispondono ai parametri,
    m è l'indice del gate della base da applicare, mentre q è l'indice del gate target. Vedi environment.step() per 
    ulteriori dettagli.

    python qc_azher.py --base_gates qiskit --target_gates qiskit --max_lenght_sequence 300 --n_qubits 2 
                       
    base_gates "qiskit" usa un set di gate che sono "nativi" sui QC IBM. Corrisponde a un gate U3(a,b,c) con a,b,c angoli,
    e una porta logica CNOT.
"""


def strtobool(v):
    if v.lower() in ("yes", "true", "True", "y", "1"):
        return True
    else:
        return False


def argument_parser():
    CLI = argparse.ArgumentParser()

    # ENV Parameters
    CLI.add_argument(
        "--base_gates",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="?",
        type=str,
        default="rotations",
        choices=["optimal", "rotations", "RXRYPI128", "RXRYPI3128", "rotationsPI128", "optimal2", "qiskit", "test_U3"],
        help="base of gates to build solution")
    CLI.add_argument(
        "--target_gates",
        nargs="?",
        type=str,
        default="optimal",
        choices=["optimal", "rotations", "optimal2", "RXRYPI128", "RXRYPI3128", "qiskit", "test_U3"],
        help="base of gates to build targets")
    CLI.add_argument(
        "--reward",
        nargs="?",
        type=int,
        default=0,
        help="the reward used. 0 is sparse reward, 1 is informative.")
    CLI.add_argument(
        "--tolerance",
        nargs="?",
        type=float,
        default=0.99,
        help="the distance between solution and target")
    CLI.add_argument(
        "--max_lenght_sequence",
        nargs="?",
        type=int,
        default=100,
        help="Episode length")
    CLI.add_argument(
        "--min_target",
        nargs="?",
        type=int,
        default=0,
        help="Targets are build as a random composition of N gates. N € [min_target, max_target]")
    CLI.add_argument(
        "--max_target",
        nargs="?",
        type=int,
        default=100,
        help="Targets are build as a random composition of N gates. N € [min_target, max_target]")
    CLI.add_argument(
        "--norm",
        nargs="?",
        type=str,
        default="fidelity",
        help="The distance metric")
    CLI.add_argument(
        "--target",
        nargs="?",
        type=str,
        default="None",
        choices=["True", "GATE_2", "None"],
        help="If != None it set a fixed target")
    CLI.add_argument(
        "--n_qubits",
        nargs="?",
        type=int,
        default=1,
        help="The number of qubits in the circuit. It corresponds to the width of the circuit.")
    CLI.add_argument(
        "--concatenate",
        nargs="?",
        type=strtobool,
        default=False,
        help="# If True it concatenates unitary and target [U,T] as nn_input instead of [X] where T = X*U")

    # AZEROHER Parameters
    CLI.add_argument(
        "--depth",
        nargs="?",
        type=int,
        default=10,
        help="number of searches")
    CLI.add_argument(
        "--n_epochs",
        nargs="?",
        type=int,
        default=1050,
        help="number of training epochs")
    CLI.add_argument(
        "--n",
        nargs="?",
        type=int,
        default=0,
        help="number used to label identical run")
    CLI.add_argument(
        "--HER",
        nargs="?",
        type=str,
        default="Posterior",
        choices=["None", "Posterior", "PosteriorFuture", "PosteriorFutureP",
                 "PosteriorFutureAllP", "PosteriorFutureNoisyP"],
        help="HER strategy")
    CLI.add_argument(
        "--k",
        nargs="?",
        type=int,
        default=4,
        help="HER parameter")
    CLI.add_argument(
        "--lr",
        nargs="?",
        type=float,
        default=0.001,
        help="optimizer learning rate")
    CLI.add_argument(
        "--network_type",
        nargs="?",
        type=str,
        default="QC",
        choices=["FC", "QC", "QC2", "FCPPO"],
        help="network type")
    CLI.add_argument(
        "--mem_max_capacity",
        nargs="?",
        type=int,
        default=200000,
        help="Max memory capacity")
    CLI.add_argument(
        "--max_processes",
        nargs="?",
        type=int,
        default=cpu_count(),
        help="number of process")
    CLI.add_argument(
        "--sample_size",
        nargs="?",
        type=int,
        default=64,
        help="number of episodes to play")
    CLI.add_argument(
        "--test_size",
        nargs="?",
        type=int,
        default=32,
        help="number of episodes to test")
    CLI.add_argument(
        "--frequency_test",
        nargs="?",
        type=int,
        default=5,
        help="number of epochs between tests")
    CLI.add_argument(
        "--init_episodes",
        nargs="?",
        type=int,
        default=0,
        help="episodes used to initially populate replay buffer")
    CLI.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=256,
        help="training batch size")
    CLI.add_argument(
        "--enable_pitting",
        nargs="?",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="Enable the pitting phase between the old and the new model")
    CLI.add_argument(
        "--n_batches",
        nargs="?",
        type=int,
        default=100,
        help="number of batches (of batch_size elements) to train at each train step")
    CLI.add_argument(
        "--fraction_new_experiences",
        nargs="?",
        type=float,
        default=0.3,
        help="number of batches (of batch_size elements) to train at each train step")
    CLI.add_argument(
        "--HER_decay_mode",
        nargs="?",
        type=str,
        default="constant",
        choices=["linear", "exp", "constant"],
        help="Scheduler decay mode for HER probability")
    CLI.add_argument(
        "--random_seed",
        nargs="?",
        type=int,
        default=0,
        help="Set the randomness. If 0 a random seed will be used")
    CLI.add_argument(
        "--dirichlet_noise_ratio",
        nargs="?",
        type=float,
        default=0.25,
        help="Set the ratio of Dirichlet noise to add to the probability of root node. Used during training only")
    CLI.add_argument(
        "--n_stochastic_depth",
        nargs="?",
        type=int,
        default=5,
        help="Number of episode before selecting the action deterministically (max number of visit)")
    CLI.add_argument(
        "--new_training",
        nargs="?",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="Use faster training")
    CLI.add_argument(
        "--c_utc",
        nargs="?",
        type=float,
        default=2,
        help="UTC constant")
    CLI.add_argument(
        "--pv_loss_ratio",
        nargs="?",
        type=float,
        default=1,
        help="ratio between policy and value loss. Policy * pv_loss_ratio + value")
    CLI.add_argument(
        "--no_gpu",
        nargs="?",
        type=strtobool,
        choices=[True, False],
        default=True,
        help="Use gpu")
    CLI.add_argument(
        "--gym_standard",
        nargs="?",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="Use stable baseline formalism")
    CLI.add_argument(
        "--verbose",
        nargs="?",
        type=int,
        default=0,
        help="Set the verbosity level. 0: ERROR; 1: INFO; 2: DEBUG.")
    CLI.add_argument(
        "--gamma",
        nargs="?",
        type=float,
        default=0.99,
        help="discount rate")

    # PW Parameters
    CLI.add_argument(
        "--alpha",
        nargs="?",
        type=float,
        default=0.81,
        help="PW exponent")
    CLI.add_argument(
        "--c_pw",
        nargs="?",
        type=float,
        default=0.15,
        help="PW slope")
    CLI.add_argument(
        "--random_action_frac",
        nargs="?",
        type=float,
        default=0.04,
        help="PW fraction of random actions")
    CLI.add_argument(
        "--std",
        nargs="?",
        type=float,
        default=None,
        help="standard deviation of the policy")
    CLI.add_argument(
        "--entropy_coef",
        nargs="?",
        type=float,
        default=0.01,
        help="entropy loss coefficient")

    return CLI


def set_logger(verbose_level=0, log_path=None):
    """
    Configure the root logger

    A stream logger will print messages based on the verbosity level.
    A file logger will print to file every INFO message. The file is created into the dir logs/{current_time}/
    Returns:

    """

    if log_path is None:
        log_path = "logs/{}/".format(time.time())
        makedirs(log_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_logger = logging.getLogger("train")

    # Saves generic information
    file_handler = logging.FileHandler(log_path + "simulation.log")
    # Saves trining history
    training_history_handler = logging.FileHandler(log_path + "training_history.log")

    stream_formatter = logging.Formatter("[%(levelname)s] - %(message)s")
    file_formatter = logging.Formatter("[%(levelname)s] [%(asctime)s::%(module)s::%(lineno)d] - %(message)s")

    stream_handler = logging.StreamHandler()
    if verbose_level == 0:
        stream_handler.setLevel(logging.ERROR)
    elif verbose_level == 1:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(stream_formatter)

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    training_history_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    train_logger.addHandler(training_history_handler)

    logger.info("Saving log file in {}".format(log_path))

    return logger


def create_dir(args):
    # Create path dir if not exists
    for key, value in args.items():

        if key == "path_results" or key == "n" or key == "note: " or key == "random_seed" or key == "--gym_standard" or \
                key == "verbose": continue  # Skip over unwanted items

        args["path_results"] += str(value) + "_"

    args["path_results"] = args["path_results"][:-1]
    args["path_results"] += "/"

    if not exists(args["path_results"]):
        print("Directory {} created".format(args["path_results"]))
        try:
            makedirs(args["path_results"])
        except:
            pass

    args["path_results"] += str(args["n"]) + "_"


def initialize_args():
    CLI = argument_parser()
    args = CLI.parse_args()
    args = vars(args)

    # Brutta soluzione ma funziona
    if args["target"] == "True":
        args["target"] = True
    elif args["target"] == "None":
        args["target"] = None
    args["note: "] = "Da qui in poi questi paramentri NON sono modificabili da linea di comando"

    args["path_results"] = "results/"

    args["single_player"] = True

    return args


def fix_randomicity(args):
    if args["random_seed"] == 0:
        args["random_seed"] = int.from_bytes(os.urandom(4), byteorder="big")  # Use a number to fix the randomness

    # Set randomicity
    np_seed(args["random_seed"])
    random_seed(args["random_seed"])
    manual_seed(args["random_seed"])

    return args


def return_env_from_base(args):
    """
    Return the correct environment (fullDiscrete, fullContinuous, Mixed) based on the base of gates
    Args:
        base_gates:

    Returns:

    """

    base_gates = GATES[args["base_gates"]]

    is_full_discrete = True
    is_full_continuos = True
    is_mixed = False

    for gate in base_gates:
        if type(gate) != ParametricGate:
            is_full_continuos = False
        if type(gate) != Gate:
            is_full_discrete = False

    if is_full_continuos and len(base_gates) > 1:
        is_full_discrete = False
        is_full_continuos = False

    if not is_full_discrete and not is_full_continuos:
        is_mixed = True

    assert is_mixed or (is_full_continuos is not is_full_discrete), "Environment cannot be both full continuous and full discrete"

    if is_full_continuos:
        assert args["n_qubits"] == 1, "Can use AzeroContinuousBrain with single-qubit circuits"
        args['brain_producer'] = AzeroContinuousBrain
        return quantumCompiling.CompilerFullContinuous
    elif is_full_discrete:
        assert args["n_qubits"] == 1, "Can use CompilerFullDiscrete with single-qubit circuits"
        return quantumCompiling.CompilerFullDiscrete
    else:
        args['brain_producer'] = AzeroMixedBrain
        return quantumCompiling.Compiler


def main():
    args = initialize_args()
    create_dir(args)
    args = fix_randomicity(args)

    logger = set_logger(args["verbose"], args["path_results"])

    # Select the correct environment based on the base of gates
    Environment = return_env_from_base(args)

    def env_maker():
        env = Environment(n_qubits=args["n_qubits"], base_gates=args["base_gates"], target_gates=args["target_gates"],
                          min_target=args["min_target"], max_target=args["max_target"],
                          max_length_circuit=args["max_lenght_sequence"],
                          epsilon=args["tolerance"], reward_type=args["reward"], norm=args["norm"],
                          gym_standard=args["gym_standard"],
                          target=args["target"], no_glob_phase=False,
                          concatenate_input=args["concatenate"])
        return env

    def tree_maker(env):
        return AzeroTree(env, args, verbose=args["verbose"])

    env = env_maker()

    nn_state, _ = env.reset()
    args["obs_dim"] = nn_state["nn_input"].shape
    args["act_dim"] = env.n_actions

    args["brain_params"] = {
        "std": args["std"],
        "entropy_coef": args["entropy_coef"]
    }
    sampler = None
    if args['max_processes'] > 1:
        sampler_params = dict(
            n_workers=min(args['max_processes'], max(args['test_size'], args['sample_size'])),
            seed=args['random_seed'] + 4000
        )
        sampler = ParallelSampler(make_env=env_maker, make_tree=tree_maker,
                                  **sampler_params)
    args["tree_maker"] = tree_maker
    amcts_model = Azero(env_maker, args, sampler=sampler)
    saveinfo_callback = SaveInfo(check_freq=1, save_path=args["path_results"], verbose=args["verbose"])
    checkpoint_callback = CheckpointModel(check_freq=1, save_path=args["path_results"], verbose=args["verbose"])
    test_model_callback = TestModel(check_freq=args["frequency_test"], save_path=args["path_results"],
                                    test_size=args['test_size'],
                                    verbose=args["verbose"], sampler=sampler, tree_maker=tree_maker)

    logger.info("Initialization complete. Starting the learning")

    start = time.perf_counter()
    amcts_model.learn(callbacks=[checkpoint_callback, test_model_callback, saveinfo_callback], verbose=args["verbose"])
    end = time.perf_counter()

    logger.info("Finished in {} seconds!".format(end - start))

    if sampler is not None:
        sampler.close()


if __name__ == '__main__': main()
