from bitflip_solver import Solver
import sys, os
import numpy as np
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from baseline.Environments.BitFlip import bitFlipping
from bitflip import initialize_args
from scipy.signal import lfilter
from baseline.Algorithms.Hybrid.AlphaZero.azero_memory import AzeroMemory
from baseline.Algorithms.Hybrid.AlphaZero.azero_network_tf import AzeroBrain
from baseline.Algorithms.Hybrid.AlphaZero.azero_network import AzeroBrain as AzeroBrain_2
import matplotlib.pyplot as plt
import time


def run_episode(env, pi, gamma=0.99):
    """
    Does not depend on the number of players (single-player == two-players)
    NOTE: Non è importante la reward qui. Anzi, non conta nulla se non lo stato
          finale a cui arrivi. Per questo è "naturale" un env che restituisce None
          se non sono arrivato alla fine dell'episodio.
    """
    S, _ = env.reset()
    np.random.seed()
    memories = deque()
    experiences = []
    ep_rewards = []
    done = False
    while not done:
        # print("INIZIO:")
        P, _ = pi.predict_one(S['nn_input'])
        P = np.array(P)
        P = P / np.sum(P)
        action = np.random.choice(len(P), p=P)
        S_, reward, done, info = env.step(action)
        S = S["nn_input"].copy()
        experiences.append((S, P))
        ep_rewards.append(reward)
        S = S_

    r = ep_rewards[::-1]
    a = [1, -gamma]
    b = [1]
    y = lfilter(b, a, x=r)  # Discounted rewards (reversed)
    values = y[::-1]

    memories.extend([(S, P, v) for (S, P), v in zip(experiences, values)])
    return memories


def check_policy(mdp, pi):
    render = True
    n_episodes = 5
    for ep in range(n_episodes):
        s, _ = mdp.reset()
        ret = 0
        print("Episode ", ep+1)
        for i in range(1, 100):
            if render:
                mdp.render()
            # print(s['nn_input'])
            P, _ = pi.predict_one(s['nn_input'])
            P = np.array(P)
            a = np.random.choice(len(P), p=P)
            s, r, done, _ = mdp.step(a)
            ret += r
            if done:
                print("Return:", ret)
                break


def main():
    args = initialize_args(15, 30)
    solver = Solver()

    def env_maker():
        env = bitFlipping.BitFlip(args["bit_depth"], False, False)
        return env

    # check_policy(env_maker(), solver)
    env = env_maker()
    num_episodes = 1000
    validation_episodes = 100
    lr = args['lr']
    memory = AzeroMemory(max_capacity=args["mem_max_capacity"], combined_q=True)
    results = []
    for i in range(num_episodes):
        results.append(run_episode(env, solver, gamma=0.99))
    experiences = [item for sublist in results for item in sublist]
    memory.add_batch(experiences)
    nn_state, _ = env.reset()
    obs_dim = nn_state["nn_input"].shape
    act_dim = env.n_actions
    brain_2 = AzeroBrain_2(obs_dim, act_dim, args["network_type"], lr=lr)

    epochs = 200
    for scale in [1]:
        current_brain = AzeroBrain(obs_dim, act_dim, args["network_type"], lr=lr * scale)
        # with tf.device(tf.test.gpu_device_name()):  # device_lib.list_local_devices()[-1].name  "/GPU:0"
        start = time.time()
        history = current_brain.train(memory.sample(1, uniform_sampling=True), epochs, args['batch_size'],
                                      stopping=True,
                                      verbose=2)
        end = time.time()
        print("Tf model:", end - start, " seconds")
        plt.plot(history.history['loss'], label='loss_' + str(scale))
        plt.plot(history.history['val_loss'], label='val_loss_' + str(scale))
    start = time.time()
    # brain_2 = AzeroBrain(obs_dim, act_dim, args["network_type"])
    # brain_2.set_weights(current_brain.get_weights())
    history = brain_2.train(memory.sample(1, uniform_sampling=True), epochs, args['batch_size'], stopping=True,
                            verbose=2)
    end = time.time()
    print("Keras model:", end - start, " seconds")

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'], label='loss_old')
    plt.plot(history.history['val_loss'], label='val_loss_old')
    plt.legend(loc='upper left')
    plt.show()

    evals = []
    for j in range(validation_episodes):
        check_policy(env, current_brain)


if __name__ == '__main__': main()