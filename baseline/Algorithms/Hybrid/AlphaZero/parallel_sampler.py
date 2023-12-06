import os
import random
import time
import logging
from collections import deque
from torch.multiprocessing import Process, Queue, Event

import numpy as np

from .utils import run_episodes, test_model

# Use default (root) logger
logger = logging.getLogger()

def traj_segment_function(tree, env, weights, HER_prob, n_episodes, eval=False, env_maker=None):
    """
    Collects trajectories
    Args:
        tree:
        env:
        weights:
        HER_prob:
        n_episodes:
        eval:
        env_maker:

    Returns: the memories

    """
    if n_episodes == 0:
        return deque()
    if eval:
        memories = test_model(tree, weights, n_episodes)
    else:
        memories = run_episodes(mcts_maker=None, env_maker=env_maker, weights=weights, n_episodes=n_episodes,
                                tree=tree, HER_prob=HER_prob)
    return memories


class Worker(Process):
    """
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any tensorflow session!
    """

    def __init__(self, output, input, event, make_env, make_tree, traj_segment_generator, seed, index):
        child_env = os.environ.copy()
        child_env['CUDA_VISIBLE_DEVICES'] = ""
        super(Worker, self).__init__(kwargs={'env': child_env})
        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_tree = make_tree
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed
        self.index = index
        self.workerseed = self.seed + 10000 * (self.index + 1)
        self.env_seed = self.workerseed + 5
        self.n_reset = 0

    def close_env(self):
        """

        Returns:

        """
        self.env.close()

    def reset(self, args=None, make_env=None, make_tree=None):
        """

        Args:
            args:
            make_env:
            make_tree:

        Returns:

        """
        self.n_reset += 1
        if make_env is not None:
            self.make_env = make_env
            self.env = make_env()
            # self.env.seed(self.env_seed + self.n_reset)
        if args is not None:
            if make_tree is not None:
                self.make_tree = make_tree
            self.tree = self.make_tree(self.env, args)

    def run(self):
        """

        Returns:

        """
        workerseed = self.workerseed
        from torch import manual_seed
        # Set randomicity
        np.random.seed(workerseed)
        random.seed(workerseed)
        manual_seed(workerseed)
        env = self.make_env()
        self.env = env
        # self.env.seed(self.env_seed + self.n_reset)
        env.reset()
        self.tree = self.make_tree(env)
        while True:
            self.event.wait()
            self.event.clear()
            command, weights, n_episodes, HER_prob, eval = self.input.get()
            if command == 'collect':
                samples = self.traj_segment_generator(self.tree, self.env, weights, n_episodes, HER_prob, eval)
                self.output.put((os.getpid(), samples))
            elif command == 'reset':
                args = weights
                make_env = n_episodes
                make_tree = HER_prob
                self.reset(args, make_env, make_tree)
            elif command == 'exit':
                logger.debug('Worker {} - Exiting...'.format(os.getpid()))
                self.tree.brain.close()
                break


class ParallelSampler(object):
    """

    """

    def __init__(self, make_env, make_tree, n_workers, seed=0):
        """

        Args:
            make_env:
            make_tree:
            n_workers:
            seed:
        """
        self.n_workers = n_workers
        logger.info('Using {} CPUs'.format(self.n_workers))
        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        fun = []
        for i in range(n_workers):
            f = lambda tree, env, weights, n_episodes, HER_prob, evaluate: traj_segment_function(tree, env, weights,
                                                                                                 HER_prob, n_episodes,
                                                                                                 env_maker=make_env,
                                                                                                 eval=evaluate)
            fun.append(f)
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i],
                               make_env, make_tree, fun[i], seed + i, i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()

    def collect(self, actor_weights, n_episodes, HER_prob=1, eval=False):
        """

        Args:
            actor_weights:
            n_episodes:
            HER_prob:
            eval:

        Returns:

        """
        n_episodes_per_process = n_episodes // self.n_workers
        remainder = n_episodes % self.n_workers
        episodes = [n_episodes_per_process for _ in range(self.n_workers)]
        if remainder > 0:
            for i in range(remainder):
                episodes[i] += 1
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', actor_weights, episodes[i], HER_prob, eval))
        for e in self.events:
            e.set()
        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.extend(samples)
        return sample_batches

    def reset(self, args=None, make_env=None, make_tree=None):
        """

        Args:
            args:
            make_env:
            make_tree:

        Returns:

        """
        for i in range(self.n_workers):
            self.input_queues[i].put(('reset', args, make_env, make_tree, None))
        for e in self.events:
            e.set()

    def close(self):
        """

        Returns:

        """
        for i in range(self.n_workers):
            # command, weights, her_prob
            self.input_queues[i].put(('exit', None, None, None, None))
        for e in self.events:
            e.set()
        for w in self.workers:
            # w.terminate()
            w.join()
