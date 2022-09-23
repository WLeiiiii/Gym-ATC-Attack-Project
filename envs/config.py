import math


class Config:
    # input dim
    window_width = 800
    window_height = 800
    diagonal = 800  # this one is used to normalize dist_to_intruder
    intruder_size = 5
    EPISODES = 1000
    G = 9.8
    tick = 30
    scale = 30

    # distance param
    minimum_separation = 1200 / scale
    NMAC_dist = 600 / scale
    horizon_dist = 4000 / scale
    initial_min_dist = 3000 / scale
    goal_radius = 600 / scale

    # speed
    min_speed = 50 / scale
    max_speed = 80 / scale
    d_speed = 5 / scale
    speed_sigma = 2 / scale
    position_sigma = 10 / scale

    # heading in rad TBD
    d_heading = math.radians(5)
    heading_sigma = math.radians(4)

    # maximum steps of one episode
    max_steps = 5000

    # reward setting
    NMAC_penalty = -10 / 10
    conflict_penalty = -5 / 10
    wall_penalty = -5 / 10
    step_penalty = -0.001 / 10
    goal_reward = 10 / 10
    sparse_reward = False
    conflict_coeff = 0.00025

    # Jonathan How R
    # NMAC_penalty = -1
    # conflict_penalty = -0.5
    # wall_penalty = -1
    # step_penalty = 0
    # goal_reward = 1
    # sparse_reward = True
    # conflict_coeff = 0.001

    # n nearest intruder
    n = 4
