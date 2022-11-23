import numpy as np
from gym import spaces

from envs.SimpleATC_env_global_v2 import SimpleEnvV2
from envs.config import Config


class SimpleEnvLocalV2(SimpleEnvV2):
    def __init__(self):
        super().__init__()
        self.lines_draw = self.lines[:self.intruder_size]
        self.intruder_nearest = Config.n

        state_dimension = self.intruder_nearest * 4 + 8
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(state_dimension,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32
        )
