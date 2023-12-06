import os
import pickle
import sys
import time
from functools import partial
from os import makedirs
from os.path import exists

import optuna
from optuna.samplers import TPESampler

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from baseline.Algorithms.Hybrid.AlphaZero.alphazero import Azero
from baseline.Algorithms.Hybrid.AlphaZero.azero_tree_continuous import AzeroTreeContinuous as AzeroTree
from baseline.Algorithms.Hybrid.AlphaZero.parallel_sampler import ParallelSampler
from baseline.Algorithms.Hybrid.AlphaZero.callbacks import TestOptimization
from qc_azher import argument_parser, fix_randomicity, return_env_from_base


def opt_argument_parser(CLI):
    """
    Extend qc_azher argument_parser
    Args:
        CLI:

    Returns:

    """

    # OPTIMIZER Parameters
    CLI.add_argument(
        "--max_evals",
        nargs="?",
        type=int,
        default=5,
        help="Number of optimization steps")
    CLI.add_argument(
        "--n_test",
        nargs="?",
        type=int,
        default=2,
        help="Number of azero simulations to run at each optimization step")
    CLI.add_argument(
        "--episode_to_test",
        nargs="?",
        type=int,
        default=1,
        help="Testing last episode_to_test episodes to estimate the cost")

    return CLI


def initialize_args():
    CLI = argument_parser()
    CLI = opt_argument_parser(CLI)

    args = CLI.parse_args()
    args = vars(args)

    # Brutta soluzione ma funziona
    if args["target"] == "True":
        args["target"] = True
    elif args["target"] == "None":
        args["target"] = None

    args["note"] = "Da qui in poi questi paramentri NON sono modificabili da linea di comando"

    args["simulation_dir"] = "optimization_results/"

    args["single_player"] = True

    return args


def create_dir(args):
    """

    Returns:
        object: 
    """
    # Create path dir if not exists
    for key, value in args.items():

        if key == "simulation_dir" or key == "n" or key == "note" or key == "random_seed" \
                or key == "brain_producer" or key == "brain_params" or key == "obs_dim" \
                or key == "act_dim": continue  # Skip over unwanted items

        args["simulation_dir"] += str(value) + "_"

    args["simulation_dir"] = args["simulation_dir"][:-1]
    args["simulation_dir"] += "/"

    if not exists(args["simulation_dir"]):
        print("Directory {} created".format(args["simulation_dir"]))
        try:
            makedirs(args["simulation_dir"])
        except:
            pass

    return args["simulation_dir"]


def objective(trial, args, azero, tree_maker, sampler):
    # Define the search space
    trial.suggest_float('lr', 0.0001, 0.01, log=True)
    trial.suggest_float('alpha', 0.1, 0.99, step=0.01)
    # trial.suggest_float('random_action_frac', 0., 0.25, step=0.01)
    trial.suggest_float("c_utc", 0.1, 2)
    trial.suggest_categorical("entropy_coef", [0.0001, 0.01, 1, 10])

    # path_results: cartella dove è contenuta la singola simulazione
    # simulation_dir: è la cartella che contiene le simulazioni (path_results)
    args["path_results"] = args['simulation_dir'] + str(time.time()) + '/'

    if not os.path.exists(args['path_results']):
        os.makedirs(args['path_results'])

    print("Evaluating with params:", trial.params)
    print("Saving results in:" + args['path_results'])

    args.update(trial.params)
    sampler.reset(args=args)
    azero.reset(args)

    test_optimization_callback = TestOptimization(n_epochs=args["episode_to_test"], n_test=args["n_test"],
                                                  sampler=sampler, tree_maker=tree_maker, verbose=args["verbose"])

    azero.learn(callbacks=[test_optimization_callback])

    optimization_info = azero.optimization_info

    print("Mean return: {}".format(optimization_info["score"]))
    print("Mean solved: {}".format(optimization_info["solved"]))
    print("Average  Length: {}".format(optimization_info["length"]))

    # Update results to file
    with open(args['simulation_dir'] + "/results.txt", "a") as f:
        f.write("##### Optimization step #####\n")
        for keys, values in optimization_info.items():
            f.write("{}: {} \n".format(keys, values))
        f.write("using parameters\n")
        for keys, values in trial.params.items():
            f.write("{}: {} \n".format(keys, values))
        f.write("\n\n")

    # opt score is the return
    return optimization_info["score"]


def main():
    args = initialize_args()
    path_results = create_dir(args)
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

    def tree_maker(env, arg=None):
        if arg is None:
            arg = args
        return AzeroTree(env, arg, verbose=args["verbose"])

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

    amcts_model = Azero(env_maker, args, sampler=sampler)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=args["random_seed"]))
    study.optimize(partial(objective, args=args, tree_maker=tree_maker,
                           azero=amcts_model, sampler=sampler), n_trials=args["max_evals"])

    with open(path_results + "trials.pickle", "wb") as dumpfile:
        pickle.dump(study, dumpfile)
        dumpfile.close()

    if sampler is not None:
        sampler.close()


if __name__ == '__main__': main()
