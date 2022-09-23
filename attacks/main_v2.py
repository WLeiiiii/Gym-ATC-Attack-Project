import random

import numpy as np
import torch

from attacks.attack_v2 import Attack
from attacks.phys_attack_v2 import PhysAttack
from attacks.st_attack_v2 import STAttack
from attacks.uni_attack_v2 import UniAttack
from envs.SimpleATC_env_five_v2 import SimpleEnv
from models.dqn_model import QNetwork, ResBlock


def without_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes):
    without_atk = Attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=False, attack_p=False,
                         episodes=episodes)
    without_atk.run()
    without_atk.plot(atk_name="Without Attack")


def uniform_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, frq, attack_p):
    uni_atk = UniAttack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=True, attack_p=attack_p,
                        episodes=episodes, frq=frq)
    uni_atk.run()
    uni_atk.plot(atk_name="UniAttack with epsilon:{} & frq:{}".format(epsilon, frq))
    pass


def strategical_time_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, beta, attack_p):
    st_atk = STAttack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=True, attack_p=attack_p,
                      episodes=episodes, beta=beta)
    st_atk.run()
    st_atk.plot(atk_name="STAttack with epsilon:{} & beta:{}".format(epsilon, beta))
    pass


def physical_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, alpha, attack_p):
    phys_atk = PhysAttack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=True, attack_p=attack_p,
                          episodes=episodes, alpha=alpha)
    phys_atk.run()
    phys_atk.plot(atk_name="PhysAttack with epsilon:{} & alpha:{}".format(epsilon, alpha))
    pass


def main_v2():
    episodes = 500
    epsilon = 0.03
    beta = 0.05
    alpha = 100
    frq = 1
    attack_p = True

    seed = 9
    random.seed(seed)
    np.random.seed(seed)
    env = SimpleEnv()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_j = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    agent_p = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    load_path_j = "../save_model/dqn_random_goal_model_09_j.pth"
    load_path_p = "../save_model/dqn_random_goal_model_09_p.pth"

    without_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes)
    # uniform_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, frq, attack_p)
    # strategical_time_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, beta, attack_p)
    # physical_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, alpha, attack_p)


if __name__ == "__main__":
    main_v2()
