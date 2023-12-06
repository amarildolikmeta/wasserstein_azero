import sys, os
import numpy as np
from os import makedirs
from os.path import exists
import random
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # AMARILDO SEI UN EROE
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse
import time
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from baseline.Algorithms.Hybrid.AlphaZero.alphazero import Azero
from baseline.Algorithms.Hybrid.AlphaZero.azero_tree import AzeroTree
from baseline.Algorithms.Hybrid.AlphaZero.parallel_sampler import ParallelSampler
from baseline.Environments.Maze.maze import Maze, Render
from baseline.Algorithms.Hybrid.AlphaZero.callbacks import SaveInfo, CheckpointModel, TestModel


# TODO: fix randomness and seeds
# TODO: split logs

def strtobool(v):
    if v.lower() in ("yes", "true", "True", "y", "1"):
        return True
    else:
        return False


def argument_parser():
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--depth",
        type=int,
        default=30,
        help="number of searches")
    CLI.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed")
    CLI.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="number of training epochs")
    CLI.add_argument(
        "--n",
        type=int,
        default=1,
        help="number used to label identical run")
    CLI.add_argument(
        "--HER",
        type=str,
        default="Posterior",
        choices=["None", "Naive", "Naive_posterior", "Posterior", "PosteriorFuture", "Loss"],
        help="HER strategy")
    CLI.add_argument(
        "--state_mode",
        type=str,
        default="2d",
        choices=["2d", "features"],
        help="state space")
    CLI.add_argument(
        "--k",
        type=int,
        default=1,
        help="HER parameter")
    CLI.add_argument(
        "--retrospective",
        type=strtobool,
        choices=[True, False],
        default=True,
        help="Using a retrospective strategy for value targets")
    CLI.add_argument(
        "--mem_max_capacity",
        type=int,
        default=20000,
        help="Max memory capacity")
    CLI.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="maximum episode length")
    CLI.add_argument(
        "--max_processes",
        type=int,
        default=10,
        help="number of process")
    CLI.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="number of episodes to play")
    CLI.add_argument(
        "--test_size",
        type=int,
        default=5,
        help="number of episodes to test")
    CLI.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="training batch size")
    CLI.add_argument(
        "--iterative",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="Using an iterative re-weighting for HER")
    CLI.add_argument(
        "--maze_width",
        type=int,
        default=10,
        help="Width of the maze")
    CLI.add_argument(
        "--maze_height",
        type=int,
        default=10,
        help="height of the maze")
    CLI.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="optimizer learning rate")
    CLI.add_argument(
        "--init_episodes",
        type=int,
        default=100,
        help="episodes used to initially populate replay buffer")
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
        "--frequency_test",
        nargs="?",
        type=int,
        default=20,
        help="number of epochs between tests")
    CLI.add_argument(
        "--HER_decay_mode",
        nargs="?",
        type=str,
        default="constant",
        choices=["linear", "exp", "constant"],
        help="Scheduler decay mode for HER probability")
    CLI.add_argument(
        "--zero_reward",
        nargs="?",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="Use sparsest reward")
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
        "--min_path_length",
        nargs="?",
        type=int,
        default=15,
        help="UTC constant")

    return CLI


def create_dir(args):
    # Create path dir if not exists
    for key, value in args.items():

        if key == "path_results" or key == "n" or key == "note: " or key == "random_seed": continue  # Skip over unwanted items

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

    print(args["path_results"])


def initialize_args(bit_length, search_depth):
    CLI = argument_parser()
    args = CLI.parse_args()
    args = vars(args)

    args["note: "] = "Da qui in poi questi paramentri NON sono modificabili da linea di comando"

    args["path_results"] = "results/maze/"

    args["single_player"] = True
    # args["multiprocess"] = True
    args["value_strategy"] = "mean"  # "mean" or "max"
    seed = args['seed']
    if seed == 0:
        np.random.seed()
        seed = np.random.randint(0, 1000000)
    args["random_seed"] = seed
    # args["random_seed"] = 120329792
    np.random.seed(args["random_seed"])
    random.seed(args["random_seed"])
    # tf.random.random_seed(args["random_seed"])
    args["gamma"] = 0.999
    args["boltzmann"] = False
    if args["boltzmann"]: args["temp"] = 0.1
    if args['state_mode'] == "2d":
        args["network_type"] = "CNN"  # CNN or FC
    elif args['state_mode'] == "features":
        args["network_type"] = "FC"
    else:
        raise ValueError("Specify Correct State Mode!")
    create_dir(args)

    return args


def main():
    args = initialize_args(15, 30)

    def env_maker():
        env = Maze(width=args['maze_width'], height=args['maze_height'], gamma=args['gamma'],
                   max_len_episode=args['horizon'], state_mode=args['state_mode'], zero_reward=args['zero_reward'],
                   min_path_length=args['min_path_length'])
        return env

    def tree_maker(env):
        return AzeroTree(env, args, verbose=False)
    env = env_maker()
    # env.render_mode = Render.TRAINING
    nn_state, _ = env.reset()
    obs_dim = nn_state["nn_input"].shape
    act_dim = env.n_actions
    args["obs_dim"] = obs_dim
    args["act_dim"] = act_dim
    sampler = None
    if args['max_processes'] > 1:
        sampler_params = dict(
            n_workers=min(args['max_processes'], args['test_size']),
            seed=args['seed'] + 4000,
            sample_size=args['test_size'],
            eval=True
        )
        sampler = ParallelSampler(make_env=env_maker, make_tree=tree_maker,
                                  **sampler_params)

    amcts_model = Azero(env_maker, args)
    saveinfo_callback = SaveInfo(check_freq=1, save_path=args["path_results"], verbose=False)
    checkpoint_callback = CheckpointModel(check_freq=1, save_path=args["path_results"], verbose=False)
    test_model_callback = TestModel(check_freq=args["frequency_test"], save_path=args["path_results"],
                                    test_size=args['test_size'], verbose=False, sampler=sampler, tree_maker=tree_maker)
    start = time.time()
    amcts_model.learn(callbacks=[checkpoint_callback, test_model_callback, saveinfo_callback])
    end = time.time()
    print("Finished in ", end-start, " seconds!")
    if sampler is not None:
        sampler.close()


if __name__ == '__main__': main()
