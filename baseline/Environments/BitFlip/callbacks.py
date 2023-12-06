'''
stable baselines not compatible with TF 2.2
'''
import numpy as np
import csv

from time import time
from os import stat
from pickle import loads, dumps

class CheckpointModel():
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, save_path: str, n: int, verbose=True):

        self.check_freq = check_freq
        self.save_path = save_path
        self.n_calls = 0
        self.n = n
        self.agent = None

        self.verbose = verbose

    def _init_callback(self, model) -> None:

        self.agent = model.agent

        if self.verbose:
            print("****CheckpointModelCallback****")
            print("save_path: ", self.save_path)
            print("check_freq: ", self.check_freq)

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            self.agent.save_model("{}{}_last_model_".format(self.save_path, self.n))
            if self.verbose: print("Checkpoint done")

        self.n_calls += 1

        return True

class SaveInfo():
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, save_path: str, n: int, verbose=True):

        self.check_freq = check_freq
        self.save_path = save_path
        self.n_calls = 0
        self.n = n
        self.t0 = time()
        self.model = None

        self.csv_infos = {"current_episode": [],
                      "episode_reward": [],
                      "solved_test": [],
                      "current_lr": [],
                      "solved": [],
                      "actions_sequence": [],
                      "lenght_episode": [],
                      "lenght_solved": [],
                      "Q_target": []}

        self.verbose = verbose

    def _init_callback(self, model) -> None:

        self.model = model

        if self.verbose:
            print("****SaveInfoCallback****")
            print("save_path: ", self.save_path)
            print("check_freq: ", self.check_freq)

    def print_info_to_file(self, args, path):

        keys = list(args.keys())
        values = list(args.values())

        with open(path, "a") as f:

            if stat(path).st_size == 0:
                for k in keys:
                    f.write("{0:20}".format(k))
                f.write("\n")


            for v in values:
                f.write("{:<20.4f}".format(v))

            f.write("\n")

    def print_info_to_csv(self, path):

        args = self.csv_infos

        # Some envs doesn't support solved info
        if "solved" not in args: args["solved"] = np.zeros(len(args["episode_reward"]))

        infos = [args["episode_reward"], args["solved"], args["lenght_episode"], args["current_lr"]]
        infos = list(zip(*infos))

        with open(path, mode="a") as f:
            csv_writer = csv.writer(f, delimiter=',')

            if stat(path).st_size == 0:
                csv_writer.writerow(["Simulation"])
                csv_writer.writerow(["r","s","l","lr"])

            for row in infos:
                csv_writer.writerow(row)

    def _on_step(self) -> bool:

        # Save info
        self.csv_infos["current_episode"].append(self.n_calls) # self.n_calls true if called at every steps only
        self.csv_infos["episode_reward"].append(self.model.info["episode_reward"].copy())
        if "solved" in self.model.info: self.csv_infos["solved"].append(self.model.info["solved"]*100)
        self.csv_infos["actions_sequence"].append(self.model.info["actions_sequence"])
        self.csv_infos["lenght_episode"].append(len(self.model.info["actions_sequence"]))
        if "solved" in self.model.info and self.model.info["solved"]: self.csv_infos["lenght_solved"].append(self.csv_infos["lenght_episode"][-1])
        self.csv_infos["current_lr"].append(self.model.agent.brain.learning_rate)
        self.csv_infos["Q_target"].append(self.model.agent.Q_target)

        # Env specific

        if "distance" in self.model.info: self.csv_infos["distance"].append( self.model.info["distance"] )

        if self.n_calls % self.check_freq == 0:

            # Print monitor infos on csv file
            self.print_info_to_csv("{}{}_.monitor.csv".format(self.save_path, self.n) )
            if self.verbose: print("Monitor csv saved of lenght {}".format(len(self.csv_infos["current_episode"])))

            # Here save info to a file
            file_info = {}
            file_info["current_episode"] = self.n_calls
            file_info["episode_reward"] = np.mean(self.csv_infos["episode_reward"])
            if "distance" in self.model.info: file_info["distance"] = np.mean(self.csv_infos["distance"])
            if "solved" in self.csv_infos: file_info["solved"] = np.mean(self.csv_infos["solved"]*100)
            file_info["lenght_episode"] = np.mean(self.csv_infos["lenght_episode"])
            if "solved" in self.csv_infos: file_info["lenght_solved"] = np.mean(self.csv_infos["lenght_solved"])
            file_info["solved_test"] = np.mean(self.csv_infos["solved_test"])
            file_info["Q_target"] = np.mean(self.csv_infos["Q_target"])
            file_info["epsilon"] = self.model.agent.epsilon
            file_info["memory_size"] = len(self.model.agent.memory.samples)
            file_info["time_cycle"] = time() - self.t0
            self.t0 = time()

            self.print_info_to_file(file_info, "{}{}_info.txt".format(self.save_path, self.n))
            if self.verbose: print("Info saved")

            self.csv_infos = {key: [] for key in self.csv_infos} # Reset infos to empty list

        self.n_calls += 1

        return True

class SaveBestModel():
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, save_path: str, n: int, number_tests: int, verbose=True):

        self.check_freq = check_freq
        self.save_path = save_path
        self.n_calls = 0
        self.n = n
        self.model = None
        self.env = None
        self.agent = None
        self.best_episode_reward = -np.inf
        self.number_tests = number_tests

        self.verbose = verbose

    def _init_callback(self, model) -> None:

        self.model = model
        self.agent = self.model.agent
        self.env = loads(dumps(self.model.env, -1))

        if self.verbose:
            print("****SaveBestModelCallback****")
            print("save_path: ", self.save_path)
            print("check_freq: ", self.check_freq)

    def test_model(self):

        if self.number_tests < 1:
            return 0., -np.inf

        number_solved_episodes = 0
        cumulative_reward = 0

        for _ in range(self.number_tests):

            # Reset environment to default
            S = self.env.reset()
            done = False

            # Play episode
            while not done:

                a = self.agent.brain.predictOne(S)
                S_, r, done, info = self.env.step(a)

                if "solved" not in info:
                    info["solved"] = 0

                number_solved_episodes += info["solved"]
                cumulative_reward += r

                S = S_

        return (number_solved_episodes/self.number_tests)*100, cumulative_reward/self.number_tests

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

            # Test model and save it if better
            solved_test, cumulative_test = self.test_model()

            self.model.info["solved"] = solved_test

            if self.verbose: print("Model tested")

            if (self.best_episode_reward < cumulative_test):
                '''
                save info file for best episode
                and save best model
                '''

                self.best_episode_reward = cumulative_test

                self.agent.save_model("{}{}_best_model_".format(self.save_path, self.n))

                if self.verbose: print("Best model saved")

        self.n_calls += 1

        return True
