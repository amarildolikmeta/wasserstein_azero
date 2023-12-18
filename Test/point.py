import sys, os
import numpy as np
from os import makedirs
from os.path import exists
from random import seed as random_seed
from numpy.random import seed as np_seed
from torch import manual_seed
from torch.multiprocessing import cpu_count
import argparse
import time
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn', force=True)
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from baseline.Algorithms.Hybrid.AlphaZero.alphazero import Azero
from baseline.Algorithms.Hybrid.AlphaZero.azero_tree import AzeroTree
from baseline.Algorithms.Hybrid.AlphaZero.azero_wasserstein_tree import AzeroWassersteinTree
from baseline.Algorithms.Hybrid.AlphaZero.parallel_sampler import ParallelSampler
from baseline.Environments.Point.point import PointEnv
from baseline.Algorithms.Hybrid.AlphaZero.callbacks import SaveInfo, CheckpointModel, TestModel
from baseline.Algorithms.Hybrid.AlphaZero.azero_network_pytorch import AzeroBrain
from baseline.Algorithms.Hybrid.AlphaZero.azero_wasserstein_network_pytorch import AzeroWassersteinBrain


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
        "--difficulty",
        type=str,
        default="double_L",
        choices=["easy", "medium", "hard", "maze", "double_L"],
        help="configuration of the point environment")
    CLI.add_argument(
        "--depth",
        type=int,
        default=70,
        help="number of searches")
    CLI.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed")
    CLI.add_argument(
        "--n_epochs",
        type=int,
        default=200,
        help="number of training epochs")
    CLI.add_argument(
        "--n",
        type=int,
        default=0,
        help="number used to label identical run")
    CLI.add_argument(
        "--HER",
        type=str,
        default="None",
        choices=["None", "Naive", "Naive_posterior", "Posterior", "Loss", "PosteriorFuture"],
        help="HER strategy")
    CLI.add_argument(
        "--k",
        type=int,
        default=5,
        help="HER parameter. If k=-1 then it samples all the (future) experiences")
    CLI.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="optimizer learning rate")
    CLI.add_argument(
        "--network_type",
        type=str,
        default="FC",
        choices=["FC", "QC"],
        help="network type")
    CLI.add_argument(
        "--init_episodes",
        type=int,
        default=100,
        help="episodes used to initially populate replay buffer")
    CLI.add_argument(
        "--retrospective",
        type=strtobool,
        choices=[True, False],
        default=True,
        help="Using a retrospective strategy for value targets")
    CLI.add_argument(
        "--mem_max_capacity",
        type=int,
        default=1000000,
        help="Max memory capacity")
    CLI.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="maximum episode length")
    CLI.add_argument(
        "--max_processes",
        type=int,
        default=10,
        help="number of process")
    CLI.add_argument(
        "--sample_size",
        type=int,
        default=64,
        help="number of episodes to play")
    CLI.add_argument(
        "--test_size",
        type=int,
        default=20,
        help="number of episodes to test")
    CLI.add_argument(
        "--frequency_test",
        nargs="?",
        type=int,
        default=5,
        help="number of epochs between tests")
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
        "--c_utc",
        nargs="?",
        type=float,
        default=2,
        help="UTC constant")
    CLI.add_argument(
        "--radius",
        type=float,
        default=1.,
        help="radius of the action circle in the point env")
    CLI.add_argument(
        "--distance_reward",
        nargs="?",
        type=strtobool,
        choices=[True, False],
        default=True,
        help="Use a reward function based on distance from goal")

    CLI.add_argument("--pv_loss_ratio", nargs="?", type=float, default=1, help="ratio between policy and value loss")
    CLI.add_argument("--path_results", type=str, default="results/", help="directory to save results")

    CLI.add_argument('--delta', type=float, default=0.95)
    CLI.add_argument('--r_min', type=float, default=0.)
    CLI.add_argument('--r_max', type=float, default=1.)
    CLI.add_argument('--prv_std_qty', type=float, default=0.)
    CLI.add_argument('--prv_std_weight', type=float, default=1.)
    CLI.add_argument('--num_layers', type=int, default=2)
    CLI.add_argument('--num_hidden', type=int, default=64)
    CLI.add_argument('--gamma', type=float, default=0.99)
    CLI.add_argument("--tree_selection", type=str, default="optimistic", choices=["optimistic", "counts", "mean"],
                     help="directory to save results")
    CLI.add_argument(
        "--optimistic",
        type=strtobool,
        choices=[True, False],
        default=True,
        help="run wasserstein azero")
    CLI.add_argument(
        "--tree_samples_ratio",
        type=int,
        default=0.,
        help="Fraction of samples from the tree")
    CLI.add_argument(
        "--action_selection",
        type=str,
        choices=["optimistic", "mean", "counts"],
        default="optimistic",
        help="behavioral policy after search")
    CLI.add_argument(
        "--tree_policy",
        type=str,
        choices=["optimistic", "mean"],
        default="optimistic",
        help="behavioral policy during search")
    CLI.add_argument(
        "--backpropagation",
        type=str,
        choices=["mc", "optimistic", "mean", "wass"],
        default="mc",
        help="target policy during tree updates")
    CLI.add_argument(
        "--mc_targets",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="Use MC-targets for the value function")
    CLI.add_argument(
        "--use_gpu",
        type=strtobool,
        choices=[True, False],
        default=False,
        help="run wasserstein azero")

    CLI.add_argument(
        "--suffix",
        type=str,
        default="",
        help="last directory of logs")
    return CLI


def create_dir(args):
    # Create path dir if not exists
    args["path_results"] += "point/" + args["difficulty"] + "/"

    if args["optimistic"]:
        args["path_results"] += "wasserstein/"
    else:
        args["path_results"] += "azero/"

    if args["optimistic"]:
        for key, value in args.items():
            if key == "single_player" or key == "path_results" or key == "n" or key == "note: " \
                    or key == "random_seed": continue
            if key not in ["tree_selection", "action_selection", "backpropagation", "mc_targets"]:
                continue
            args["path_results"] += str(key) + "_" + str(value) + "/"

    if args["suffix"] != "":
        args["path_results"] += args["suffix"] + "/"
    args["path_results"] += "s" + str(args["random_seed"]) + "/"
    if not exists(args["path_results"]):
        print("Directory {} created".format(args["path_results"]))
        makedirs(args["path_results"])

    print(args["path_results"])


def initialize_args():
    CLI = argument_parser()
    args = CLI.parse_args()
    args = vars(args)
    args["single_player"] = True
    fix_randomicity(args)
    create_dir(args)
    return args


def fix_randomicity(args):
    if args["random_seed"] == 0:
        np_seed()
        args["random_seed"] = np.random.randint(100000)
    # Set randomicity
    np_seed(args["random_seed"])
    random_seed(args["random_seed"])
    manual_seed(args["random_seed"])

    return args


def main():
    args = initialize_args()

    def env_maker():
        env = PointEnv(difficulty=args["difficulty"], horizon=args['horizon'], distance_reward=args['distance_reward'])
        return env

    env = env_maker()
    nn_state, _ = env.reset()
    obs_dim = nn_state["nn_input"].shape
    act_dim = env.n_actions
    args["obs_dim"] = obs_dim
    args["act_dim"] = act_dim
    if args["optimistic"]:
        args["brain_producer"] = AzeroWassersteinBrain
        q_max = args["r_max"] / (1 - args["gamma"])
        q_min = args["r_min"] / (1 - args["gamma"])
        mean = (q_max + q_min) / 2
        std = (q_max - q_min) / np.sqrt(12)
        log_std = np.log(std)
        args["brain_params"] = {
            "num_layers": args["num_layers"],
            "num_hidden": args["num_hidden"],
            "init_mean": mean,
            "init_std": log_std,
            "prv_std_qty": args["prv_std_qty"],
            "prv_std_weight": args["prv_std_weight"],
        }
        args["tree_params"] = {
            "backpropagation": args["backpropagation"],
            "tree_policy": args["tree_policy"]
        }
        args["search_params"] = {
            "selection": args["action_selection"]
        }
        tree_producer = AzeroWassersteinTree
    else:
        args["brain_producer"] = AzeroBrain
        args["brain_params"] = {
            "num_layers": args["num_layers"],
            "num_hidden": args["num_hidden"],
            "network_type": args["network_type"],
            "pv_loss_ratio": args["pv_loss_ratio"],
        }
        args["tree_params"] = {

        }
        args["search_params"] = {

        }
        tree_producer = AzeroTree
    args["brain_params"]["use_gpu"] = args["use_gpu"]

    def tree_maker(env):
        return tree_producer(env, args, verbose=False)
    args["tree_maker"] = tree_maker
    sampler = None
    if args['max_processes'] > 1:
        sampler_params = dict(
            n_workers=min(args['max_processes'], max(args['test_size'], args['sample_size'])),
            seed=args['random_seed'] + 4000
        )
        sampler = ParallelSampler(make_env=env_maker, make_tree=tree_maker,
                                  **sampler_params)

    amcts_model = Azero(env_maker, args,)
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


if __name__ == '__main__':
    main()
