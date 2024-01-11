import numpy as np
from gym import utils
try:
    from mujoco_env import MujocoEnv
except:
    from .mujoco_env import MujocoEnv

import math

from gym import spaces

diff_to_path = {
    'empty': 'point_empty.xml',
    'easier': 'point.xml',
    'easy': 'point.xml',
    'medium': 'point_medium.xml',
    'hard': 'point_hard.xml',
    'harder': 'point_harder.xml',
    'maze': 'maze.xml',
    'maze_easy': 'maze_easy.xml',
    'maze_simple': 'maze_simple.xml',
    'maze_med': 'maze_med.xml',
    'maze_hard': 'maze_hard.xml',
    'double_L': 'double_L.xml',
    'double_I': 'double_I.xml',
    'para': 'point_para.xml'
}


class PointEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, difficulty=None, max_state=25, clip_state=True, terminal=False,
                 goal=[25.0, 0.0], horizon=50, radius=1., distance_reward=False, z_dim=False,
                 goal_dependent_state=False, reward_scale=True):
        if difficulty is None:
            difficulty = 'medium'
        elif difficulty == 'easier':
            goal = [10., 0.]
        model = diff_to_path[difficulty]
        self.max_state = max_state
        self.clip_state = clip_state
        self.max_len_episode = horizon
        self.bounds = [[0, -9.7, 0], [25, 9.7, 0]]
        self.vbounds = [[-50, -50, 0], [50, 50, 0]]
        self.terminal = terminal
        self.goal_dependent_state = goal_dependent_state
        self.z_dim = z_dim
        self.goal = goal
        self.n_actions = 9
        self.timestep = 0
        self.done = False
        self.radius = radius
        self.reward_scale = reward_scale
        self.distance_reward = distance_reward
        if self.reward_scale:
            self.abs_min_reward = np.linalg.norm(self.goal - np.array([0, 9.7]))
        low = np.array(self.bounds[0] + self.vbounds[0])
        high = np.array(self.bounds[1] + self.vbounds[1])
        if self.z_dim:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low[[0,1,3,4]], high[[0,1,3,4]], dtype=np.float32)
        MujocoEnv.__init__(self, model, 1)
        if self.z_dim:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low[[0,1,3,4]], high[[0,1,3,4]], dtype=np.float32)
        # self.action_space = gym.spaces.Discrete(self.n_actions)
        utils.EzPickle.__init__(self)

        self.reset()

    def current_state_to_S(self, current_state=None, goal=None):
        if current_state is None: current_state = self.get_current_state()
        if goal is None: goal = self.goal
        if self.goal_dependent_state:
            goal = goal[:2]
            S = np.concatenate([current_state, goal])
        else:
            S = current_state
        S = self.to_n1p1(S)
        return S

    def to_n1p1(self, state):
        v_min = self.observation_space.low
        v_max = self.observation_space.high
        if any(v_min == -np.inf) or any(v_max == np.inf):
            raise ValueError('unbounded state')
        new_min, new_max = -1, 1
        res = (state - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        res = np.nan_to_num(res, nan=0) # if we want to keep z at zero
        res = np.clip(res, new_min, new_max)
        return res

    def get_current_state(self):
        current_state = self._get_obs(z=self.z_dim)
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
        if reward >= -2.:
            if not evaluate: self.done = True
            info["solved"] = True
            self.solved = True
        else:
            if not evaluate: self.done = False
            info["solved"] = False
            self.solved = False
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
            info["solved"] = self.solved
            self.done = True
            return self.get_S(), 0, self.done, info
        if isinstance(action, int):
            action = self.discrete_to_continuous_action(action)
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs(z=True)

        qpos = next_obs[:3]
        if self.clip_state:
            qvel = next_obs[3:]
            qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            qvel_clipped = np.clip(qvel, a_min=self.vbounds[0], a_max=self.vbounds[1])
            self._set_state(qpos_clipped, qvel_clipped)
            qpos = qpos_clipped
        next_obs = self._get_obs(self.z_dim)
        reward, info = self.get_reward()
        self.timestep += 1
        if self.timestep > self.max_len_episode or info['solved']:
            info['done'] = True
            self.done = True

        info["goal"] = self.goal
        return self.get_S(), reward, self.done, info

    def _get_obs(self, z=True):
        if z:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat[:2],
                self.sim.data.qvel.flat[:2],
            ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        try:
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        except:
            qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        if self.clip_state:
            qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            qvel_clipped = np.clip(qvel, a_min=self.vbounds[0], a_max=self.vbounds[1])
            self._set_state(qpos_clipped, qvel_clipped)
        else:
            self._set_state(qpos, qvel)
        return self._get_obs(z=self.z_dim)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent

    def is_valid_action(self, action):
        return not self.done

    def reset(self):
        #self.goal = np.random.uniform(low=[15., 0], high=[25., 10])
        ob = super(PointEnv, self).reset()
        self.current_state = ob
        self.timestep = 0
        self.actions_sequence_index = []
        self.done = False
        self.solved = False
        #  self.goal = (self.current_state[:2] + np.array([2, 2])).tolist()
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
        qpos = np.zeros(3)
        qvel = np.zeros(3)
        index = 3 if self.z_dim else 2
        qpos[:index] = self.current_state[:index]
        qvel[:index] = self.current_state[index:]
        if self.clip_state:
            qpos = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            qvel = np.clip(qvel, a_min=self.vbounds[0], a_max=self.vbounds[1])
        self._set_state(qpos, qvel)
        self.goal = node.S["goal"].copy()
        self.done = node.done
        self.timestep = node.S["timestep"]

    def set_state(self, state):
        self.current_state = state["current_state"].copy()
        qpos = np.zeros(3)
        qvel = np.zeros(3)
        index = 3 if self.z_dim else 2
        qpos[:index] = self.current_state[:index]
        qvel[:index] = self.current_state[index:]
        if self.clip_state:
            qpos = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            qvel = np.clip(qvel, a_min=self.vbounds[0], a_max=self.vbounds[1])
        self._set_state(qpos, qvel)
        self.goal = state["goal"].copy()
        self.done = state["done"]
        self.timestep = state["timestep"]


if __name__ == "__main__":
    problem_state = {
        'current_state': np.array([8.56228667, 1.396712, 17.50119073, 3.32020388]),
        'goal': [10.0, 0.0],
        'nn_input': np.array([-0.31501707, 0.14399093, 0.35002381, 0.06640408]),
        'done': False,
        'timestep': 46}
    env = PointEnv(difficulty='easy', horizon=120, radius=1., goal=[10.0, 0.0])
    s, _ = env.reset()
    for i in range(10):
        env.step(np.random.choice(9))

    init_s = env.get_S()
    for i in range(10):
        env.step(np.random.choice(9))
    s = env.get_S()
    for i in range(25):
        actions = np.random.choice(9, 10)
        for a in actions:
            env.step(3)
        env.set_state(init_s)
        env.step(1)
        env.set_state(s)
        env.step(1)
        print(env.get_S())
        env.render()
        input()
        env.set_state(s)

    rets = []
    for j in range(100):
        ob, _ = env.reset()
        env.set_state(problem_state)
        root = env.get_S()
        done = False
        t = 0
        ret = 0
        while not done:
            env.render()
            command = input()
            try:
                ac = int(command)
            except:
                ac = np.random.choice(9)
            # ac = np.random.choice(9)
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
        input()
        env.set_state(root)
        env.render()
        input()
    print("Mean Return:", np.mean(rets))
    print("Std Return:", np.std(rets))
    # env.render()
