"""
This is based on alphaZero and has HER feature
"""
from collections import deque
from contextlib import closing
from os import environ
from os.path import exists
from time import time

import numpy as np
import logging
#from multiprocess import Pool
from .azero_memory import AzeroMemory
from .parallel_sampler import ParallelSampler
from .utils import run_episodes, run_random_policy

# Use default (root) logger
logger = logging.getLogger()

class Azero:
    def __init__(self, env_maker, args, sampler=None):
        """

        Args:
            env_maker: Function with no parameters that returns an instance of the environment
            args: Parameter dictionary used for the enviornment, approximators and tree-search
            sampler:
        """

        self.args = args
        self.optimization_info = {}  # Ignore if not optimizing (do not remove though)
        self.AzeroBrain = args['brain_producer']
        self.tree_maker = args["tree_maker"]
        self.sampler = None
        self.reset(args, env_maker)
        self.sampler = sampler
        if self.n_processes > 1 and sampler is None:
            sampler_params = dict(
                n_workers=self.n_processes,
                seed=args['random_seed'] + 5000
            )
            self.sampler = ParallelSampler(make_env=self.env_maker,
                                           make_tree=self.tree_maker,
                                           **sampler_params)

        # Set randomicity
        # np.random.seed(self.args["random_seed"])
        # seed(self.args["random_seed"])
        # manual_seed(self.args["random_seed"])

    def reset(self, args, env_maker=None):
        """
        Reset the tree
        Args:
            args:
            env_maker:

        Returns: None

        """
        self.args = args
        if env_maker is not None:
            self.env_maker = env_maker
            self.env = env_maker()
            nn_state, _ = self.env.reset()
            self.obs_dim = obs_dim = nn_state["nn_input"].shape
            self.act_dim = act_dim = self.env.n_actions
        self.random_seed = args["random_seed"]
        self.args["obs_dim"] = self.obs_dim
        self.args["act_dim"] = self.act_dim
        self.HER_prob = 1
        self.n_processes = args["max_processes"]
        self.sample_size = args["sample_size"]
        self.test_size = args["test_size"]
        self.batch_size = args["batch_size"]
        self.n_batches = args["n_batches"]

        self.infos = {"epoch": 0,
                      "t-expanded": 0.,
                      "t-training": 0.,
                      "t-pitting": 0.,
                      "ELO": 1000.,
                      "MCTS_ELO": 1000.,
                      "POLICY_ELO": 1000.,
                      "HER_prob": self.HER_prob}
        self.history = {}  # Training log
        self.memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True,
                                  fraction_new_experiences=args['fraction_new_experiences'])
        self.best_elo = -np.inf
        self.tree = None
        self.best_return = -np.inf
        self.last_returns = deque(maxlen=10)
        self.patience = 0  # Number of consecutive bad training
        self.current_brain = self.AzeroBrain(self.obs_dim, self.act_dim, lr=args["lr"], **args["brain_params"])
        self.best_weights = self.current_brain.get_weights().copy()
        if self.sampler is not None:
            self.sampler.reset(args=args, make_env=env_maker)

    def initialization(self):
        """
        Initialize the tree by setting
        Returns:

        """

        exists_current_model = exists("{}models/current_model/".format(self.args["path_results"])) and False

        # load old agent
        if exists_current_model:
            self.current_brain.load_weights("{}models/current_model/".format(self.args["path_results"]))
            self.best_weights = self.current_brain.get_weights().copy()

        # load memory and train new agent
        if exists("{}memory.npy".format(self.args["path_results"])):
            self.memory.load_form_file("{}memory.npy".format(self.args["path_results"]))
            if not exists_current_model: self.train_model(epochs=50, stopping=True)

        # Print args to file
        with open("{}parameters.txt".format(self.args["path_results"]), "w") as f:
            for keys, values in self.args.items():
                f.write("{}: {} \n".format(keys, values))

    def learn(self, callbacks=None, verbose=0):
        """

        Args:
            callbacks:

        Returns:

        """

        epochs = self.args["n_epochs"]
        self.initialization()
        if self.args["init_episodes"] > 0:
            self.initialize_memory(self.args["init_episodes"])

        # Initialize callbacks
        if callbacks is not None:
            if isinstance(callbacks, list):
                for callback in callbacks:
                    callback._init_callback(self)
            else:
                callbacks._init_callback(self)

        for epoch in range(epochs):
            logger.info("Epoch: {}".format(epoch))
            print("Epoch: {}".format(epoch))
            training_epochs = 1
            # Simulate games to increase experiences
            t0 = time()
            self.expand_memory()
            self.infos["t-expanded"] = time() - t0
            if epoch == 0:
                """
                Test the untrained model
                """
                # Run callbacks every time-step
                if callbacks is not None:
                    if isinstance(callbacks, list):
                        for i, callback in enumerate(callbacks):
                            callback._on_step()
                    else:
                        callbacks._on_step()
                self.infos["epoch"] = epoch

            else:
                """
                Train the model and then test it
                """
                # Train current_brain on experiences
                t0 = time()
                self.history = self.train_model(epochs=training_epochs)
                self.infos["t-training"] = time() - t0
                self.infos["memory size"] = len(self.memory.samples)

                # Run callbacks every time-step
                if callbacks is not None:
                    if isinstance(callbacks, list):
                        for i, callback in enumerate(callbacks):
                            callback._on_step()
                    else:
                        callbacks._on_step()
                self.infos["epoch"] = epoch

            # if self.args["enable_pitting"]:
            #     assert self.args["test_size"] > 0, "ERROR: cannot test a model using none test episode"
            #
            #     if epoch % self.args["frequency_test"] == 0 and epoch != 0:
            #         # Test model every frequency_test epochs
            #
            #         if self.infos["return"] >= self.best_return:
            #             '''
            #             Update if current model is better than old one
            #             (self.infos["return"] is updated by test_callback)
            #
            #             best_model: it is used to populate the memory
            #             current_model: it is the new trained policy
            #
            #             Update the best_model with the current_model if the latter achieve better performances
            #             otherwise discard the current_model and start again
            #             '''
            #             logger.info("Model updated at epoch {}".format(epoch))
            #             self.best_return = self.infos["return"]
            #             self.best_weights = self.current_brain.get_weights()
            #             self.current_brain.save("{}models/best_model/".format(self.args["path_results"]))
            #
            #             # self.patience = 0  # Reset counter
            #
            #         else:
            #             self.current_brain.set_weights(self.best_weights)
            #             # self.memory.reset()
            # else:
            self.best_weights = self.current_brain.get_weights()
            self.current_brain.save("{}models/best_model/".format(self.args["path_results"]))

            if self.args["test_size"] > 0 and epoch % self.args["frequency_test"] == 0:
                try:
                    self.last_returns.append(self.infos["return"])  # used by optimizer only
                except:
                    pass

            # HER scheduler
            self.decrease_HER_prob()

    def initialize_memory(self, initial_episodes=100):
        """

        Args:
            initial_episodes:

        Returns:

        """

        if self.args["HER"] != "Posterior":
            self.expand_memory(initial_episodes)
        else:
            if self.n_processes == 1:
                experiences = run_random_policy(self.env_maker, np.random.randint(100000000),
                                                (self.args["gamma"], self.args["k"]), initial_episodes,
                                                env=self.env)
            else:
                iterations = max(initial_episodes // self.n_processes, 1)
                remainder = initial_episodes % self.n_processes if self.n_processes < initial_episodes else 0
                episodes = [iterations for _ in range(self.n_processes)]
                for i in range(remainder):
                    episodes[i] += 1
                with closing(Pool(self.n_processes)) as pool:

                    envs = [self.env_maker for _ in range(self.n_processes)]
                    seeds = np.random.randint(100000000, size=self.n_processes)
                    arg = [(self.args["gamma"], self.args["k"]) for _ in range(self.n_processes)]

                    # Pool does not accept multiple arguments. You need to zip them
                    results = pool.starmap(run_random_policy, zip(envs, seeds, arg, episodes))
                    pool.close()
                    pool.join()
                experiences = [item for sublist in results for item in sublist]
            self.memory.add_batch(experiences)

    def expand_memory(self, episodes=-1):
        """
        Memory is expanded using best_weights

        Args:
            episodes:

        Returns:

        """

        # NOTE: puoi picklare più file alla volta...

        # weights_to_use = self.best_weights
        weights_to_use = self.current_brain.get_weights()

        # Only known method to avoid memory allocation on gpu
        environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if episodes == -1:
            episodes = self.sample_size
        if self.n_processes == 1:
            env = self.env
            if self.tree is None:
                self.tree = self.tree_maker(env)
                self.tree.set_brain(self.current_brain)
            tree = self.tree
            experiences = run_episodes(self.tree_maker, self.env_maker, weights_to_use, n_episodes=episodes, tree=tree,
                                       HER_prob=self.HER_prob)  # seed=np.random.randint(1000000),
        else:
            experiences, _ = self.sampler.collect(weights_to_use, HER_prob=self.HER_prob, n_episodes=episodes)
        self.memory.add_batch(experiences)

        # Restore GPU allocation
        # environ['CUDA_VISIBLE_DEVICES'] = '0'

    def train_model(self, epochs=2, stopping=False):
        """

        Args:
            epochs:
            stopping:

        Returns:

        """

        # self.memory.clean_samples(8)
        # from tensorflow.python.client import device_lib

        # with tf.device(tf.test.gpu_device_name()): #device_lib.list_local_devices()[-1].name  "/GPU:0"
        new_training = False  # self.args['new_training']
        if new_training:
            """
            Per ora questa strategia non funziona
            """
            experiences = self.memory.sample(self.n_batches, self.batch_size)

            # Forse funzionano meglio i generatori
            # Forse dovrei usare l'equivalente di train_on_batch
            experiences = sum(experiences, [])
            # for batch in experiences:
            #     history = self.current_brain.train(batch, 1, len(batch), stopping=stopping)
            history = self.current_brain.train(experiences, 1, self.batch_size, stopping=stopping)
        else:
            history = self.current_brain.train(self.memory.old_sample(1), epochs, self.batch_size,
                                               stopping=stopping)

        # Dovrei tenere conto delle history di tutti i batch non solo dell'ultimo
        hist = {"loss": np.average(history.history["loss"]),
                   "val_loss": np.average(history.history["val_loss"]),
                   # "value_loss": np.average(history.history["value_loss"]),
                   #"val_probabilities_loss": np.average(history.history["val_probabilities_loss"]),
                   #"val_value_loss": np.average(history.history["val_value_loss"])
                }
        additional_fields = ["q_loss", "std_loss", "old_std_loss", "value_loss", "mean_q", "mean_std",
                             "probabilities_loss", 'entropy', "val_q_loss", "val_std_loss", "val_value_loss",
                             'entropy_loss', 'std', 'policy_target_loss', 'policy_gate_loss', 'policy_lp_loss']
        for field in additional_fields:
            if field in history.history:
                hist[field] = np.average(history.history[field])
        return hist

    def save_model(self):
        """
        Save nn model to file

        Returns: None

        """
        self.current_brain.save("{}models/current_model/".format(self.args["path_results"]))

    def close(self):
        """

        Returns:

        """
        self.current_brain.close()

    def decrease_HER_prob(self):
        """
        HER scheduler
        Returns: None

        """

        min_HER_prob = 0.05

        if self.args["HER_decay_mode"] == "linear":
            decay_prob = 1 / self.args["n_epochs"]

            if self.HER_prob > min_HER_prob:
                self.HER_prob -= decay_prob
        elif self.args["HER_decay_mode"] == "exp":
            if self.HER_prob > min_HER_prob:
                self.HER_prob = np.power(self.HER_prob, -self.args["HER_decay_rate"])
        else:
            pass

        # Update the infos (to print it on file)
        self.infos["HER_prob"] = self.HER_prob
