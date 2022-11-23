import random

import numpy as np

from envs.SimpleATC_env_global import Aircraft, Ownship, Goal
from envs.SimpleATC_env_local import SimpleEnvLocal


class SimpleEnvLocalX(SimpleEnvLocal):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.max_steps = 1000
        self.epochs += 1
        self.intruder_size = random.randint(self.intruder_nearest, 25)
        self.lines_draw = self.lines[:self.intruder_size]
        self.goal = Goal(position=self.random_pos())
        # self.goal = Goal(position=(self.window_width / 2, self.window_height / 2))

        reset_d = self.reset_drone()
        self.drone = Ownship(
            position=reset_d[0],
            speed=reset_d[1],
            heading=reset_d[2]
        )

        self.intruder_list = []
        for i in range(self.intruder_size):
            reset_i = self.reset_intruder()
            intruder = Aircraft(
                position=reset_i[i][0],
                speed=reset_i[i][1],
                heading=reset_i[i][2]
            )

            self.intruder_list.append(intruder)

        if self.goal_num:
            self.steps_num.append(self.max_steps_num)
            self.steps_num_mean.append(int(np.mean(self.steps_num)))
        else:
            self.steps_num_mean.append(0)
        self.no_conflict = 0
        self.conflict_num = 0
        self.collision_num = 0
        self.goal_num = 0
        self.wall_num = 0
        self.max_step = 0
        self.max_steps_num = 0

        return self._get_obs()
