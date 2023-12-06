###############################################################################
# The copyright of this code, including all portions, content, design, text,  #
# output and the selection and arrangement of the subroutines is owned by     #
# the Authors and by CNR, unless otherwise indicated, and is protected by the #
# provisions of the Italian Copyright law.                                    #
#                                                                             #
# All rights reserved. This software may not be reproduced or distributed, in #
# whole or in part, without the prior written permission of the Authors.      #
# However, reproduction and distribution, in whole or in part, by non-profit, #
# research or educational institutions for their own use is permitted if      #
# proper credit is given, with full citation, and copyright is acknowledged.  #
# Any other reproduction or distribution, in whatever form and by whatever    #
# media, is expressly prohibited without the prior written consent of the     #
# Authors. For further information, please contact CNR.                       #
# Contact person:           enrico.prati@cnr.it                               #
#                                                                             #
# Concept and development:  Lorenzo Moro, Enrico Prati                        #
# Year:                     2019                                              #
# Istituto di Fotonica e Nanotecnologie - Consiglio Nazionale delle Ricerche  #
###############################################################################

import collections
import random

import numpy as np


class AzeroMemory:  # stored as ( s, pi, z )

    def __init__(self, max_capacity=4000, delta_expand=1000, combined_q=True, fraction_new_experiences=0.1):
        """
        La memoria viene inizializzata ed è tutta disponibile. Per renderla dinamica
        chiamare set_adaptive_capacity e specificare l'incremento

        Args:
            max_capacity:
            delta_expand:
            combined_q:
            fraction_new_experiences:
        """

        self.max_capacity = max_capacity
        self.capacity = max_capacity
        self.delta_expand = delta_expand

        self.samples = collections.deque(maxlen=self.capacity)
        self.probabilities = collections.deque(maxlen=self.capacity)
        self.fraction_new_experiences = fraction_new_experiences
        self.combined_q = combined_q
        if self.combined_q:
            self.last_sample = None  # Combined-Q strategy replay

    def reset(self):
        """
        Reset the memory

        Returns: None

        """
        self.capacity = self.max_capacity
        self.samples = collections.deque(maxlen=self.capacity)
        self.probabilities = collections.deque(maxlen=self.capacity)

    def add(self, sample, probability=None):
        """
        Add a single experience to memory
        Args:
            sample:
            probability:

        Returns:

        """
        self.samples.append(sample)
        if probability is not None:
            self.probabilities.append(probability)

    def set_adaptive_capacity(self, delta_expand):
        """
        Initialize the memory

        Args:
            delta_expand: size (number of new experiences) to add

        Returns:

        """
        self.delta_expand = delta_expand
        self.capacity = len(self.samples) + delta_expand
        self.samples = collections.deque(self.samples, maxlen=self.capacity)
        self.probabilities = collections.deque(self.probabilities, maxlen=self.capacity)

    def extend_memory(self):
        """
        Resize the memory

        Returns: None

        """

        if self.capacity <= self.max_capacity:
            self.capacity += self.delta_expand
            self.samples = collections.deque(self.samples, maxlen=self.capacity)
            self.probabilities = collections.deque(self.probabilities, maxlen=self.capacity)

    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.

        Args:
            x:

        Returns:

        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def add_batch(self, sample_batch, probability_batch=None):
        """
        Add a batch of experiences to the memory
        Args:
            sample_batch: the batch of experiences to add
            probability_batch: tha associated probability (to remove)

        Returns:

        """
        # that's the reason why I need deque class
        # extend could be very slow with arrays

        self.samples.extend(sample_batch)

        samples = np.asarray(self.samples, dtype=object)
        np.random.shuffle(samples)
        self.samples.clear()
        self.samples.extend(samples)

        if self.combined_q: self.last_sample = sample_batch

        if probability_batch is not None:
            self.probabilities.extend(probability_batch)

    def old_sample(self, n, uniform_sampling=True):
        """
        """
        if n <= 1:
            n = int(len(self.samples) * n)
        if n > len(self.samples):
            n = len(self.samples)
        batch = collections.deque()

        if uniform_sampling:
            batch.extend(random.sample(self.samples, n))
        else:
            probabilities = AzeroMemory.softmax(self.probabilities)
            idx = np.random.choice(len(self.samples), n, p=probabilities, replace=False)
            batch.extend(np.asarray(self.samples, dtype=object)[idx])

        if self.combined_q:
            batch.extend(self.last_sample)

        return batch

    def sample(self, n_batches, batch_size):
        """
        Return a list of minibatches. Half of the data come from old memories, half from the most recent experiences.
        Args:
            n_batches: number of batches
            batch_size: number of experiences in a batch

        Returns:

        """
        n_samples = batch_size * n_batches
        n_new_exps = int(n_samples * self.fraction_new_experiences)
        n_old_exps = n_samples - n_new_exps

        # Convert deque to array so you can access by multiple indices
        np_samples = np.asarray(self.samples, dtype=object)
        np_last_sample = np.asarray(self.last_sample, dtype=object)

        # Sample from memory and from last samples
        old_exp_idx = np.random.choice(len(self.samples), size=n_old_exps)
        new_exp_idx = np.random.choice(len(self.last_sample), size=n_new_exps)

        # Join the memories
        a = np.concatenate((np_samples[old_exp_idx], np_last_sample[new_exp_idx]), axis=0)
        np.random.shuffle(a)
        a = a.tolist()
        # Create the batches
        batches = [a[i:i + batch_size] for i in range(0, len(a), batch_size)]

        return batches

    def clean_samples(self, n):
        """
        Remove duplicates samples from memory
        Args:
            n: depth

        Returns:

        """

        for _ in range(n):
            states = [e[0].flatten().tolist() for e in self.samples]
            probabilities = [e[1] for e in self.samples]
            winners = [e[2] for e in self.samples]

            out = collections.Counter([tuple(i) for i in states])

            most_common_states = list(out.most_common(1)[0][0])

            idxs = [i for i, el in enumerate(states) if el == most_common_states]
            new_samples = [el for i, el in enumerate(self.samples) if i not in idxs]

            mean_winner = np.mean(winners)
            mean_probabilities = np.asarray([sum(x) for x in zip(*probabilities)]) / (len(probabilities))

            new_samples.append((np.asarray(most_common_states), mean_probabilities, mean_winner))

            self.samples = collections.deque(maxlen=self.capacity)
            self.add_batch(new_samples)

    def get_all(self):
        """
        Get all the samples
        Returns: all the samples

        """
        return self.samples

    def load_form_file(self, path):
        """
        Load memories from file
        Args:
            path: path to the file

        Returns: None

        """
        samples, probabilities = np.load(path, allow_pickle=True)
        self.max_capacity = len(samples)
        self.capacity = self.max_capacity
        self.samples = collections.deque(maxlen=self.capacity)
        self.probabilities = collections.deque(maxlen=self.capacity)
        self.add_batch(samples, probabilities)

    def save_to_file(self, path):
        """
        Save memory to file
        Args:
            path: where to save the file

        Returns:

        """
        np.save(path, (self.samples, self.probabilities), allow_pickle=True, fix_imports=True)
