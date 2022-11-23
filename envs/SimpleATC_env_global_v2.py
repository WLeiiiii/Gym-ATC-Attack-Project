import math

import numpy as np
from gym import spaces

from envs.SimpleATC_env_global import SimpleEnv, dist, Aircraft
from envs.config import Config


class SimpleEnvV2(SimpleEnv):
    def __init__(self):
        super().__init__()

    def _terminal_reward(self):
        conflict = False

        for idx in range(self.intruder_size):

            intruder = self.intruder_list[idx]
            intruder.position += intruder.velocity
            dist_intruder = dist(self.drone, intruder)
            # if this intruder out of map
            if not self.position_range.contains(intruder.position):
                reset_i = self.reset_intruder()
                intruder_i = Aircraft(
                    position=reset_i[idx][0],
                    speed=reset_i[idx][1],
                    heading=reset_i[idx][2]
                )

                self.intruder_list[idx] = intruder_i

            # if there is a conflict
            if dist_intruder < self.minimum_separation:
                conflict = True
                # if conflict status transition from False to True, increase number of conflicts by 1
                # if conflict status is True, monitor when this conflict status will be escaped
                if intruder.conflict == False:
                    self.no_conflict += 1
                    intruder.conflict = True
                else:
                    if not dist_intruder < self.minimum_separation:
                        intruder.conflict = False

                # if there is a near-mid-air-collision
                if dist_intruder < self.NMAC_dist:
                    self.collision_num += 1
                    self.collision_num_up += 1
                    return [0, Config.NMAC_penalty], True, 'n'  # NMAC

        if conflict:
            self.conflict_num += 1
            return [0, Config.conflict_penalty], False, 'c'  # conflict

        # if ownship out of map
        # if not self.position_range.contains(self.drone.position):
        #     self.wall_num += 1
        #     self.wall_num_up += 1
        #     return Config.wall_penalty, True, 'w'  # out-of-map

        if dist(self.drone, self.goal) < self.goal_radius:
            self.goal_num += 1
            self.goal_num_up += 1
            return [Config.goal_reward, 0], True, 'g'  # goal

        if Config.sparse_reward:
            return [Config.step_penalty, 0], False, ''
        else:
            return [-dist(self.drone, self.goal) / 2400 + Config.step_penalty, 0], False, ''