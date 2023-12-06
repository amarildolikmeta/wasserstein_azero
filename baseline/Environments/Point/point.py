import numpy as np
import gym
from gym import utils
try:
    from mujoco_env import MujocoEnv
except:
    from .mujoco_env import MujocoEnv
import math


diff_to_path = {
    'easy': 'point.xml',
    'medium': 'point_medium.xml',
    'hard': 'point_hard.xml',
    'harder': 'point_harder.xml',
    'maze': 'maze.xml',
    'maze_easy': 'maze_easy.xml'
}


class PointEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, difficulty=None, max_state=25, clip_state=True, terminal=False,
                 goal=[25.0, 0.0], horizon=50, radius=1., distance_reward=False):
        if difficulty is None:
            difficulty = 'medium'
        model = diff_to_path[difficulty]
        self.max_state = max_state
        self.clip_state = clip_state
        self.max_len_episode = horizon
        self.bounds = [[0, -9.7, 0], [25, 9.7, 0]]
        self.terminal = terminal
        self.goal = goal
        self.n_actions = 2
        self.timestep = 0
        self.done = False
        self.radius = radius
        self.distance_reward = distance_reward
        MujocoEnv.__init__(self, model, 1)
        # self.action_space = gym.spaces.Discrete(self.n_actions)
        utils.EzPickle.__init__(self)
        self.reset()

    def current_state_to_S(self, current_state=None, goal=None):
        if current_state is None: current_state = self.get_current_state()
        if goal is None: goal = self.goal
        goal = goal[:2]
        S = np.concatenate([current_state, goal])
        return S

    def get_current_state(self):
        current_state = self._get_obs()
        return current_state

    def get_current_pos(self):
        current_state = self._get_obs()
        return current_state[:3]

    def get_reward(self, current_state=None, goal=None, evaluate=False):
        info = {}
        if not evaluate:
            if current_state is None: current_state = self.get_current_state()
            if goal is None: goal = self.goal
        qpos = current_state[:3]
        reward = -np.linalg.norm(goal[:2] - qpos[:2])
        done = False
        if reward >= -1.:
            if not evaluate: self.done = True
            info["solved"] = True
        else:
            if not evaluate: self.done = False
            info["solved"] = False
        if not self.distance_reward:
            reward = -1

        return reward, info

    def discrete_to_continuous_action(self, action):
        if action == 0:
            return np.array([0, 0])
        step_rad = math.radians(360 / (self.n_actions - 1))
        x = self.radius * math.cos((action - 1) * step_rad)
        y = self.radius * math.sin((action - 1) * step_rad)
        return np.array([x, y])

    def step(self, action):
        if self.timestep > self.max_len_episode or self.done:
            info = {}
            info["goal"] = self.goal
            info["done"] = True
            self.done = True
            return self.get_S(), 0, self.done, info
        # if not isinstance(action, np.ndarray):
        #     action = self.discrete_to_continuous_action(action)
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()

        qpos = next_obs[:3]

        if self.clip_state:
            qvel = next_obs[3:]
            qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            self._set_state(qpos_clipped, qvel)
            qpos = qpos_clipped
            next_obs = self._get_obs()
        reward, info = self.get_reward()
        self.timestep += 1
        if self.timestep > self.max_len_episode or info['solved']:
            info['done'] = True
            self.done = True

        info["goal"] = self.goal
        return self.get_S(), reward, self.done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self._set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent

    def is_valid_action(self, action):
        return not self.done

    def reset(self):
        self.goal = np.random.uniform(low=[15., 0], high=[25., 10])
        ob = super(PointEnv, self).reset()
        self.current_state = ob
        self.timestep = 0
        self.actions_sequence_index = []
        self.done = False
        return self.get_S(), self.done

    def get_S(self):

        #if self.gym_standard: return self.current_state_to_S()

        S = {"current_state": self.get_current_state(),
             "goal": self.goal,
             "nn_input": self.current_state_to_S(),
             "done": self.done,
             "timestep": self.timestep}

        return S

    def set_S(self, node):

        self.current_state = node.S["current_state"].copy()
        qpos = self.current_state[:3]
        qvel = self.current_state[3:]
        qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
        self._set_state(qpos_clipped, qvel)
        self.goal = node.S["goal"].copy()
        self.done = node.is_terminal
        self.timestep = node.S["timestep"]

    def set_state(self, state):
        self.current_state = state["current_state"].copy()
        qpos = self.current_state[:3]
        qvel = self.current_state[3:]
        qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
        self._set_state(qpos_clipped, qvel)
        self.goal = state["goal"].copy()
        self.done = state["done"]
        self.timestep = state["timestep"]


if __name__ == "__main__":
    env = PointEnv(difficulty='medium', horizon=120, radius=10.)
    rets = []
    for j in range(100):
        ob,_ = env.reset()
        root = env.get_S()
        done = False
        t = 0
        ret = 0
        while not done:
            # env.render()
            # command = input()
            # try:
            #     ac = int(command)
            # except:
            #     ac = np.random.choice(9)
            ac = np.random.choice(9)
            _, r, done, _ = env.step(ac)
            t += 1
            ret += r
            # if t== 30:
            #     node = lambda :None
            #     node.S = root
            #     node.is_terminal = False
            #     env.set_S(node)
        print("Return:", ret)
        rets.append(ret)
    print("Mean Return:", np.mean(rets))
    print("Std Return:", np.std(rets))
    # env.render()
