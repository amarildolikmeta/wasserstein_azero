from csv import writer
from numbers import Number
from os import stat
from os.path import exists
from time import time

import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.figure import figaspect
from pandas import read_csv, Series
import pickle

from .azero_tree import AzeroTree
from .utils import test_model

# Use default (root) logger
logger = logging.getLogger()

class CheckpointModel:
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    """

    def __init__(self, check_freq: int, save_path: str, verbose=0):
        """
        Args:
            check_freq:
            save_path: Path to the folder where the model will be saved.
            verbose:
        """

        self.check_freq = check_freq
        self.save_path = save_path
        self.n_calls = 0
        self.current_brain = None

        self.verbose = verbose

    def _init_callback(self, model) -> None:
        """

        Args:
            model:

        Returns:

        """
        self.current_brain = model.current_brain

        logger.info("****CheckpointModelCallback****")
        logger.info("save_path: {}".format(self.save_path))
        logger.info("check_freq: {}".format(self.check_freq))

    def _on_step(self) -> bool:
        """

        Returns:

        """
        if self.n_calls % self.check_freq == 0:
            self.current_brain.save("{}models/current_model/".format(self.save_path))
            logger.info("Checkpoint done at episode {}".format(self.n_calls))

        self.n_calls += 1

        return True


class PrintStats:
    """
    Callback to print imags of reward, solved episodes percentage and length of episodes.
    """

    def __init__(self, check_freq: int, dir_path: str, N: int, to_plot: list, verbose=True):
        """

        Args:
            check_freq:
            dir_path:
            N:
            to_plot:
            verbose:
        """

        self.check_freq = check_freq
        self.n_calls = 0
        self.dir_path = dir_path
        self.N = N  # Mean running averages
        self.to_plot = to_plot  # Labels to plot from info
        self.verbose = verbose

        self.c_blue = '#003f5c'
        self.C_ORANGE = '#c44900'
        self.C_GREEN = '#1b6d39'
        self.C_GRAY = "#EEEEEE"

    def _init_callback(self, model) -> None:
        """

        Args:
            model:

        Returns:

        """

        self.infos = model.infos

        logger.info("****PrintStatsCallback****")
        logger.info("check_freq: {}".format(self.check_freq))
        logger.info("dir_path: {}".format(self.dir_path))

    def print_moving_average(self, data, y_label_name):
        """

        Args:
            data:
            y_label_name:

        Returns:

        """

        gap = 1

        w, h = figaspect(1 / 2)
        plt.figure(dpi=400, figsize=(w, h))
        plt.xlabel("Episode")
        plt.ylabel(y_label_name)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.rc('grid', linestyle="-", color='gray')

        ts = Series(data)

        running_mean = np.array(ts.rolling(self.N).mean())[self.N - 1:]
        running_variance = np.array(ts.rolling(self.N).std())[self.N - 1:]

        len_running_mean = len(running_mean)

        running_mean = running_mean[::gap]
        running_variance = running_variance[::gap]

        plt.plot(np.arange(0, len_running_mean, gap), running_mean, color=self.c_blue)
        plt.fill_between(np.arange(0, len_running_mean, gap), running_mean - running_variance,
                         running_mean + running_variance, color=self.c_blue, alpha=0.2)

        plt.grid(True)

        plt.savefig("{}{}.pdf".format(self.dir_path, y_label_name), bbox_inches='tight')

        plt.close("all")

    def _on_step(self) -> bool:
        """

        Returns:

        """

        if self.n_calls % self.check_freq == 0 and self.n_calls != 0:

            # Check if file exists
            if exists(self.dir_path + "monitor.csv"):
                # load_csv
                datas = read_csv(self.dir_path + "monitor.csv", header=0)

                header = [key for key, data in datas.items()]

                # for key, data in datas.items():

                # Check if arguments exists and discard those that are not present
                for arg in self.to_plot:
                    if arg in header:
                        self.print_moving_average(datas[arg], arg)  # Print moving averages
                    else:
                        logger.debug(header)
                        logger.debug(self.to_plot)
                        logger.debug(arg)

            else:
                logger.error("file not found {}info.txt".format(self.dir_path))

        self.n_calls += 1

        return True


class TestOptimization:
    """
    Callback used during the optimization procedure. It test the model n_epochs before the ending of the training.

    :param n_epochs: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param validation_data: (list)
    :param lr: float32 learning rate
    :param training_type quantum or classic
    :param verbose: (int)
    """

    def __init__(self, n_epochs: int, n_test: int, tree_maker, sampler, verbose=True):

        self.n_epochs = n_epochs  # number of epochs to test
        self.n_calls = 0
        self.n_test = n_test
        self.tree_maker = tree_maker
        self.tree = None
        self.sampler = sampler
        self.verbose = verbose

        # where to save episodes scores.
        self.costs = []
        self.ep_solved = []
        self.ep_length = []

    def _init_callback(self, model) -> None:

        self.model = model
        self.infos = model.infos
        self.env = model.env
        self.current_brain = model.current_brain
        self.args = model.args

        self.training_epochs = self.args["n_epochs"]


        logger.info("****TestOtimization****")
        logger.info("Testing last {} epochs: ".format(self.n_epochs))

    def _on_step(self) -> bool:

        # Test last epochs only
        if self.n_calls >= (self.training_epochs - self.n_epochs):
            if self.args["max_processes"] == 1:
                if self.tree is None:
                    if self.tree_maker is None:
                        self.tree = AzeroTree(self.env, self.args)
                    else:
                        self.tree = self.tree_maker(self.env)
                    self.tree.set_brain(self.current_brain)
                results, logs = test_model(self.tree, self.current_brain.get_weights(), n_episodes=self.n_test)
            else:
                results, logs = self.sampler.collect(self.current_brain.get_weights(), eval=True, n_episodes=self.n_test)
            for k in results[0].keys():
                # cannot do the mean of list element
                if not hasattr(results[0][k], '__len__'):
                    if isinstance(results[0][k], Number):
                        self.infos[k] = np.mean(list(d[k] for d in results))

            self.costs.append(self.infos["return"])
            self.ep_solved.append(self.infos["solved"])
            self.ep_length.append(self.infos["length"])

            self.model.optimization_info["score"] = np.mean(self.costs)  # Update optimization score (to be optimized)
            self.model.optimization_info["solved"] = np.mean(self.ep_solved)  # Update solved episodes
            self.model.optimization_info["length"] = np.mean(self.ep_length)  # Update length episodes

        self.n_calls += 1

        return True


class TestModel:
    """
    Callback to test the model (reward, solved episodes percentage)
    """

    def __init__(self, check_freq: int, save_path: str, test_size: int, verbose=0, sampler=None,
                 tree_maker=None, start_from_episode=0):
        """

        Args:
            check_freq:
            save_path:
            test_size:
            verbose:
            sampler:
            tree_maker:
            start_from_episode:
        """

        self.check_freq = check_freq
        self.n_calls = 0
        self.save_path = save_path
        self.verbose = verbose
        self.n_test = test_size
        self.sampler = sampler
        self.infos = None
        self.env = None
        self.current_brain = None
        self.args = None
        self.tree = None
        self.tree_maker = tree_maker
        self.start_from_episode = start_from_episode  # Start to test all episodes starting from "start_from_episode"

    def _init_callback(self, model) -> None:
        """

        Args:
            model:

        Returns:

        """

        self.infos = model.infos
        self.env = model.env
        self.current_brain = model.current_brain
        self.args = model.args

        logger.info("****TestModelCallback****")
        logger.info("check_freq: {}".format(self.check_freq))

    def _on_step(self) -> bool:
        """

        Returns:

        """

        t0 = time()

        if self.start_from_episode == 0:
            testing = self.n_calls % self.check_freq == 0
        else:
            testing = self.n_calls > self.start_from_episode

        if testing and self.n_test > 0:  # Start at the beginning
            if self.args["max_processes"] == 1:
                if self.tree is None:
                    if self.tree_maker is None:
                        self.tree = AzeroTree(self.env, self.args)
                    else:
                        self.tree = self.tree_maker(self.env)
                    self.tree.set_brain(self.current_brain)
                results, logs = test_model(self.tree, self.current_brain.get_weights(), n_episodes=self.n_test)
            else:
                results, logs = self.sampler.collect(self.current_brain.get_weights(), eval=True, n_episodes=self.n_test)
            for k in results[0].keys():
                # cannot do the mean of list element
                if not hasattr(results[0][k], '__len__'):
                    if isinstance(results[0][k], Number):
                        self.infos[k] = np.mean(list(d[k] for d in results))
            if len(logs) > 0:
                save_path = self.save_path + "/logs_" + str(self.n_calls) + ".pkl"
                with open(save_path, 'wb') as outp:
                    pickle.dump(logs, outp)
        self.n_calls += 1
        self.infos["t-testing"] = time() - t0

        return True


class SaveInfo:
    """
    Callback for saving the stats to file.
    """

    def __init__(self, check_freq: int, save_path: str, verbose=0):
        """

        Args:
            check_freq:
            save_path: Path to the folder where the model will be saved.
            verbose:
        """

        self.check_freq = check_freq
        self.save_path = save_path
        self.n_calls = 0
        self.verbose = verbose

        self.infos = None
        self.history = None

        self.model = None

    def _init_callback(self, model) -> None:
        """

        Args:
            model:

        Returns:

        """
        self.infos = model.infos
        self.history = model.history
        self.model = model

        logger.info("****SaveInfoCallback****")
        logger.info("save_path: {}".format(self.save_path))
        logger.info("check_freq: {}".format(self.check_freq))

    @staticmethod
    def print_as_csv(data, path):
        """

        Args:
            data:
            path:

        Returns:

        """
        with open(path, 'a') as f:  # You will need 'wb' mode in Python 2.x

            w = writer(f)

            if stat(path).st_size == 0:
                w.writerow(data.keys())
                w.writerow(data.values())
            else:
                w.writerow(data.values())

    @staticmethod
    def print_to_file(data, path):
        """

        Args:
            data:
            path:

        Returns:

        """
        keys = list(data.keys())
        values = list(data.values())

        with open(path, "a") as f:

            if stat(path).st_size == 0:
                for k in keys:
                    f.write("{0:20}".format(k))
                f.write("\n")
            for v in values:
                f.write("{:<20.2E}".format(v))

            f.write("\n")

    def _on_step(self) -> bool:
        """

        Returns:

        """
        # Print stuff

        if self.n_calls % self.check_freq == 0 and self.n_calls != 0:
            self.print_as_csv(self.infos, "{}info.csv".format(self.save_path))
            self.print_as_csv(self.model.history, "{}history.csv".format(self.save_path))
            self.print_as_csv(self.infos, "{}monitor.csv".format(self.save_path))
        self.n_calls += 1

        return True
