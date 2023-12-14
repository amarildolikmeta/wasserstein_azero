from collections import deque

import numpy as np
from scipy.signal import lfilter


def run_random_policy(env_maker, seed, args, n_episodes=100, env=None):
    """

    Args:
        env_maker:
        seed:
        args:
        n_episodes:
        env:

    Returns:

    """
    if env is None:
        env = env_maker()
    # Random seed per process
    # np.random.seed(seed)
    gamma, K_goals = args
    memories = deque()
    for i in range(n_episodes):
        S, done = env.reset()
        experiences = []
        states_buffer = []
        ep_rewards = []
        while not done:
            action = np.random.randint(env.n_actions)
            S_, reward, done, info = env.step(action)
            P = np.zeros(env.n_actions)
            P[action] = 1.
            experiences.append((S["nn_input"], P))
            ep_rewards.append(reward)
            states_buffer.append((S_, None))  # STD_HER
            S = S_

        r = ep_rewards[::-1]
        a = [1, -gamma]
        b = [1]
        y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
        values = y[::-1]
        memories.extend([(S, P, v) for (S, P), v in zip(experiences, values)])

        """
        Apply HER by sampling k targets from visited states. No Tree reweight
        """
        # TODO: é veramente necessario HER con politica randomica??
        HER_memories = her_posterior(env=env, k=K_goals, experiences=experiences, gamma=gamma,
                                     states_buffer=states_buffer)
        memories.extend(HER_memories)
    return memories


def her_posterior_future_P(env, k, states_buffer, experiences, gamma):
    """
    Apply HER by sampling k targets from future states. No Tree reweight.
    It also set the probabilities to 0,0,0,...,1,...,0 for every goal
    Args:
        env:
        k:
        states_buffer:
        experiences:
        gamma:

    Returns:

    """

    HER_memories = []

    for i, _ in enumerate(states_buffer[:-1]):
        # Qui ciclo su ogni timestep

        future_states = states_buffer[i:]
        oldS, P, action_ = experiences[i]
        len_future_states = len(future_states)

        n_goals = min(k, len_future_states)

        idxs = np.random.choice(len_future_states, size=n_goals, replace=False)
        future_states = np.asarray(future_states, dtype=object)
        new_goals = future_states[idxs]

        for new_goal, _ in new_goals:

            tmp_states = []
            tmp_rewards = []
            tmp_P = P.copy()

            for S, action in future_states:

                tmp_states.append(env.current_state_to_S(S["current_state"], new_goal["current_state"]))
                new_reward, info = env.get_reward(S["current_state"], new_goal["current_state"], evaluate=True)
                tmp_rewards.append(new_reward)

                if info['solved']: break  # Maybe it exists a solution before the end of the episode

            # Compute new values
            r = tmp_rewards[::-1]
            a = [1, -gamma]
            b = [1]
            y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
            new_values = y[::-1]

            '''
            Compute new probabilities!
            If the next visited state corresponds to the new goal, then set the P=[0,0,...,1,...,0]
            '''
            current_state, action = future_states[0]
            next_state, _ = future_states[1]
            if np.array_equal(new_goal["current_state"], next_state["current_state"]):
                tmp_P = np.zeros(len(P))
                tmp_P[action] = 1

            # Save just the first step
            HER_memories.append((tmp_states[0], tmp_P, action_, new_values[0]))
    return HER_memories


def her_posterior_future(env, k, states_buffer, experiences, gamma):
    """
    Apply HER by sampling k targets from future states. No Tree reweight
    Args:
        env:
        k:
        states_buffer:
        experiences:
        gamma:

    Returns:

    """

    HER_memories = []

    for i, _ in enumerate(states_buffer[:-1]):

        future_states = states_buffer[i:]
        oldS, P, action = experiences[i]
        len_future_states = len(future_states)

        n_goals = min(k, len_future_states)

        idxs = np.random.choice(len_future_states, size=n_goals, replace=False)
        future_states = np.asarray(future_states, dtype=object)
        new_goals = future_states[idxs]

        for new_goal, _ in new_goals:

            tmp_states = []
            tmp_rewards = []

            for S, _ in future_states:

                tmp_states.append(env.current_state_to_S(S["current_state"], new_goal["current_state"]))
                new_reward, info = env.get_reward(S["current_state"], new_goal["current_state"], evaluate=True)
                tmp_rewards.append(new_reward)

                if info['solved']:
                    break  # Maybe it exists a solution before the end of the episode

            # Compute new values
            r = tmp_rewards[::-1]
            a = [1, -gamma]
            b = [1]
            y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
            new_values = y[::-1]

            # Save just the first step
            HER_memories.append((tmp_states[0], P, action, new_values[0]))
    return HER_memories


def her_posterior(env, k, states_buffer, experiences, gamma):
    """
    Apply HER by sampling k targets from visited states. No Tree reweight
    Args:
        env:
        k:
        states_buffer:
        experiences:
        gamma:

    Returns:

    """
    n_goals = min(k, len(states_buffer))
    if n_goals == 1:
        new_goals = [states_buffer[-1]]
    else:
        idxs = np.random.choice(len(states_buffer), size=n_goals, replace=False)
        states_buffer = np.asarray(states_buffer, dtype=object)
        new_goals = states_buffer[idxs]
    HER_memories = []
    for new_goal, _ in new_goals:
        new_states = []
        new_rewards = []
        for S, _ in states_buffer:
            new_states.append(env.current_state_to_S(S["current_state"], new_goal["current_state"]))
            new_reward, info = env.get_reward(S["current_state"], new_goal["current_state"], evaluate=True)
            new_rewards.append(new_reward)
            if info['solved']:
                break  # Maybe it exists a solution before the end of the episode

        # Compute new values
        r = new_rewards[::-1]
        a = [1, -gamma]
        b = [1]
        y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
        new_values = y[::-1]
        HER_memories += [(S, P, a, v) for S, (oldS, P, a), v in zip(new_states, experiences, new_values)]
    return HER_memories


def run_episodes(mcts_maker, env_maker, weights, n_episodes=1, tree=None, HER_prob=1, seed=None):
    """
    Does not depend on the number of players (single-player == two-players)
    Args:
        mcts_maker:
        env_maker:
        weights:
        seed:
        n_episodes:
        tree:
        HER_prob:

    Returns:

    """

    if tree is None:
        env = env_maker()
        mcts = mcts_maker(env)
    else:
        mcts = tree
        env = mcts.env
    if seed is not None:
        np.random.seed(seed)
    memories = deque()
    training_tree = mcts
    training_tree.set_brain_weights(weights)
    depth_amcts = training_tree.args["depth"]
    optimistic = training_tree.args["optimistic"]
    mc_targets = training_tree.args["mc_targets"]
    search_params = training_tree.args["search_params"]
    tree_samples_ratio = training_tree.args["tree_samples_ratio"]
    HER_type = training_tree.args["HER"]
    if HER_type != "None":
        if np.random.rand() > HER_prob:
            HER_type = "None"
    for i in range(n_episodes):
        training_tree.reset()
        experiences = []
        ep_rewards = []

        if HER_type == "Posterior" or HER_type == "PosteriorFuture" \
                or HER_type == "PosteriorFutureP" or HER_type == "PosteriorFutureAllP" \
                or HER_type == "PosteriorFutureNoisyP": states_buffer = []
        #t = 0
        done = False
        while not done:
            action, index = training_tree.get_best_action(depth_amcts, mode="train", **search_params)
            S = training_tree.root.S["nn_input"].copy()
            if HER_type == "Posterior" or HER_type == "PosteriorFuture" \
                    or HER_type == "PosteriorFutureP" or HER_type == "PosteriorFutureAllP" \
                    or HER_type == "PosteriorFutureNoisyP":
                states_buffer.append((training_tree.root.S, action))  # STD_HER

            P = training_tree.get_probabilities(HER_type)
            S_, reward, done, info = env.step(action)
            #t += 1
            if optimistic:
                qs = training_tree.root.Qs
                sigmas = training_tree.root.sigmas
                if mc_targets:
                    experiences.append((S, action, qs[action], sigmas[action]))
                else:
                    for k, q in enumerate(qs):
                        experiences.append((S, k, q, sigmas[k]))
                    if tree_samples_ratio > 0:
                        nodes = training_tree.bfs(max=tree_samples_ratio)
                        for node in nodes:
                            qs = node.Qs
                            sigmas = node.sigmas
                            for k, q in enumerate(qs):
                                    experiences.append((node.S["nn_input"], k, q, sigmas[k]))
            else:
                experiences.append((S, P, action))
            ep_rewards.append(reward)
            training_tree.set_new_root(index, S_)
            if (HER_type == "Posterior" or HER_type == "PosteriorFuture" or
                HER_type == "PosteriorFutureP" or HER_type == "PosteriorFutureAllP"
                or HER_type == "PosteriorFutureNoisyP") and done:
                states_buffer.append((S_, None))  # STD_HER

        r = ep_rewards[::-1]
        a = [1, -training_tree.gamma]
        b = [1]
        y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
        values = y[::-1]
        if not optimistic:
            memories.extend([(S, P, a, v) for (S, P, a), v in zip(experiences, values)])
        else:
            if mc_targets:
                for k, (S, a, v, sigma) in enumerate(experiences):
                    experiences[k] = (S, a, values[k], sigma)
            memories.extend(experiences)
        if HER_type == "Posterior":
            HER_memories = her_posterior(env=env, states_buffer=states_buffer, k=training_tree.args["k"],
                                         gamma=training_tree.gamma, experiences=experiences)
            memories.extend(HER_memories)
        elif HER_type == "PosteriorFuture" or HER_type == "PosteriorFutureAllP" or \
                HER_type == "PosteriorFutureNoisyP":
            HER_memories = her_posterior_future(env=env, states_buffer=states_buffer, k=training_tree.args["k"],
                                                gamma=training_tree.gamma, experiences=experiences)
            memories.extend(HER_memories)
        elif HER_type == "PosteriorFutureP":
            HER_memories = her_posterior_future_P(env=env, states_buffer=states_buffer, k=training_tree.args["k"],
                                                  gamma=training_tree.gamma, experiences=experiences)
            memories.extend(HER_memories)
        else:
            pass

    return memories


def ELO(elo1, elo2, game_result):
    """

    Args:
        elo1:
        elo2:
        game_result:

    Returns:

    """
    if game_result == 0:
        S1 = 0.5
        S2 = 0.5
    elif game_result == -1:
        S1 = 0
        S2 = 1
    else:
        S1 = 1
        S2 = 0

    K = 32

    R1 = 10 ** (elo1 / 400)
    R2 = 10 ** (elo2 / 400)

    elo1 = elo1 + K * (S1 - R1 / (R1 + R2))
    elo2 = elo2 + K * (S2 - R2 / (R1 + R2))

    return elo1, elo2


def fill_memory(env_maker, args):
    """

    Args:
        env_maker:
        args:

    Returns:

    """
    # Random seed per process
    # np.random.seed(int.from_bytes(urandom(4), byteorder='little'))

    env = env_maker()

    memories = deque()

    S, done = env.reset()  # Ok, no tree here

    if not args["single_player"]: node_player = 1

    while not done:

        action = np.random.randint(0, env.n_actions)
        while env.is_valid_action(action):
            action = np.random.randint(0, env.n_actions)

        P = np.zeros(env.n_actions)
        P[action] = 1

        memories.append((S, P))

        S, reward, done, _ = env.step(action)

        z = reward if done else None

        if not args["single_player"]: node_player *= -1

    if args["single_player"]:
        memories = [(S, P, z) for S, P in memories]
    else:
        memories.reverse()

        v = z * node_player

        # in MCTS n=1,2,3,4
        memories = [(S, P, (-1) ** (n + 1) * v) for n, (S, P) in enumerate(memories)]

    return memories, z


def test_model(training_tree, weights, n_episodes=1, seed=None):
    """

        Args:
            training_tree:
            weights:
            n_episodes:

        Returns:

        """
    # Random seed per process
    if seed is not None:
        np.random.seed(seed)
    training_tree.set_brain_weights(weights)
    results = []
    env = training_tree.env
    for i in range(n_episodes):
        training_tree.reset()
        done = False
        info = {"return": 0,
                "length": 0}
        depth_amcts = training_tree.args["depth"]  # Non mi piace questa soluzione, ma va
        t = 0
        while not done:
            action, index = training_tree.get_best_action(depth_amcts)
            S_, reward, done, ep_info = env.step(action)
            training_tree.set_new_root(index, S_)
            t += 1
            info["length"] += 1
            info["return"] += reward
            if done:
                info["solved"] = ep_info["solved"]
            if "distance" in ep_info:
                info["distance"] = ep_info["distance"]
        results.append(info)
    return results
