import numpy as np
import gym, gym.spaces
from gym.utils import seeding


class BitFlip(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_bits, fixed=True, gym_standard=True):
        super(BitFlip, self).__init__()

        self.gym_standard = gym_standard # are you using stable baselines?

        self.n_bits = n_bits
        self.fixed = fixed

        self.action_space = gym.spaces.Discrete(self.n_bits)
        self.observation_space = gym.spaces.Discrete(self.n_bits)
        
        self.actions_sequence_index = [] # Info...

        self.n_actions = self.n_bits

        if self.fixed:
            self.starting_state = np.ones(self.n_bits)
            self.goal = -np.ones(self.n_bits)
        else:
            self.change_states(random_goal=True, random_starting_state=True)
            
        self.current_state = self.starting_state.copy()

        self.max_len_episode = n_bits + 5
        self.timestep = 0

        self.done = False

    def current_state_to_S(self, current_state=None, goal=None):

        if current_state is None: current_state = self.current_state
        if goal is None: goal = self.goal

        S = np.asarray([1 if a == b else -1 for a,b in zip(current_state, goal)])
        return S

    def get_reward(self, current_state=None, goal=None, evaluate=False):

        info = {}
        if not evaluate: 
            if current_state is None: current_state = self.current_state
            if goal is None: goal = self.goal

        if np.array_equal(current_state, goal):
            reward = 1
            if not evaluate: self.done = True
            info["solved"] = True
        else:
            reward = -1
            if not evaluate: self.done = False
            info["solved"] = False

        return reward, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        """

        self.current_state[action] = -self.current_state[action]
        
        self.actions_sequence_index.append(action)

        reward, info = self.get_reward()

        if self.timestep > self.max_len_episode:
            self.done = True

        self.timestep += 1
        
        info["goal"] = self.goal
        info["actions_sequence_index"] = self.actions_sequence_index

        return self.get_S(), reward, self.done, info
    
    def change_states(self, random_goal=False, random_starting_state=False):
        
        assert random_goal or random_starting_state
        
        # Set the target bit and current_state randomly
        if random_starting_state: self.starting_state = np.random.choice([-1, 1], size=(self.n_bits,), p=[0.5, 0.5])
        if random_goal: self.goal = np.random.choice([-1, 1], size=(self.n_bits,), p=[0.5, 0.5])
        
        while (self.starting_state == self.goal).all():
            if random_starting_state: self.starting_state = np.random.choice([-1, 1], size=(self.n_bits,), p=[0.5, 0.5])
            if random_goal: self.goal = np.random.choice([-1, 1], size=(self.n_bits,), p=[0.5, 0.5])

    def reset(self, change_goal=True, change_starting_state=True):
        """
        """

        if not self.fixed and (change_goal or change_starting_state):
            self.change_states(change_goal, change_starting_state)
            
        self.current_state = self.starting_state.copy()
        self.timestep = 0
        self.actions_sequence_index = []
        self.done = False

        return self.get_S(), self.done

    def render(self, mode='human', close=False):
        """
        """
        print(self.current_state)
        print(self.goal)
        print("##############")

    def is_valid_action(self, action):
        return not self.done

    def get_S(self):

        #if self.gym_standard: return self.current_state_to_S()

        S = {"current_state": self.current_state,
             "goal": self.goal,
             "nn_input": self.current_state_to_S(),
             "done": self.done,
             "timestep": self.timestep}

        return S

    def set_S(self, node):

        self.current_state = node.S["current_state"].copy()
        self.goal = node.S["goal"].copy()
        self.done = node.is_terminal
        self.timestep = node.S["timestep"]

    def set_state(self, state):
        self.current_state = state["current_state"].copy()
        self.goal = state["goal"].copy()
        self.done = state["done"]
        self.timestep = state["timestep"]


if __name__ == '__main__':
    env = BitFlip(10)
    s, _ = env.reset()
    print(s['nn_input'].shape)
