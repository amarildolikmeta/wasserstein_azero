from stable_baselines import HER, DQN
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import numpy as np

model_class = DQN  # works also with SAC, DDPG and TD3
N_BITS = 10
env = BitFlippingEnv(N_BITS, continuous=False, max_steps=N_BITS+7)

# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
policy_kwargs = None  # {"layers": [20, 10]}#
# Wrap the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=5, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1, policy_kwargs=policy_kwargs)
# Train the model
model.learn(10000)


# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method

rets = []
for i in range(100):
    obs = env.reset()
    ret = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, r, done, _ = env.step(action)
        ret += r
    rets.append(ret)
print("Average return:", np.mean(rets))
print("Std returns:", np.std(rets))
