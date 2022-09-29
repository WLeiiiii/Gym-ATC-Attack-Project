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
    without_atk.plot(atk_name="Without Attack_safeDQN")


def uniform_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, frq, attack_p, method):
    uni_atk = UniAttack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=True, attack_p=attack_p,
                        episodes=episodes, frq=frq, method=method)
    uni_atk.run()
    uni_atk.plot(atk_name="UniAttack with epsilon:{} & frq:{}_safeDQN_{}".format(epsilon, frq, method))
    pass


def strategical_time_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, beta, attack_p,
                            method):
    st_atk = STAttack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=True, attack_p=attack_p,
                      episodes=episodes, beta=beta, method=method)
    st_atk.run()
    st_atk.plot(atk_name="STAttack with epsilon:{} & beta:{}_safeDQN_{}".format(epsilon, beta, method))
    pass


def physical_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, alpha, attack_p,
                    method):
    phys_atk = PhysAttack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk=True, attack_p=attack_p,
                          episodes=episodes, alpha=alpha, method=method)
    phys_atk.run()
    phys_atk.plot(atk_name="PhysAttack with epsilon:{} & alpha:{}_safeDQN_{}".format(epsilon, alpha, method))
    pass


def main_v2():
    episodes = 1000
    epsilon = 0.03
    beta = 0.05
    alpha = 100
    frq = 1
    attack_p = True  # True: attack DQN_p, False: attack DQN_j
    method = "G"

    seed = 9
    random.seed(seed)
    np.random.seed(seed)
    env = SimpleEnv()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_j = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    agent_p = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    load_path_j = "../save_model/dqn_random_goal_model_09_j.pth"
    load_path_p = "../save_model/dqn_random_goal_model_09_p.pth"

    # without_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes)
    uniform_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, frq, attack_p, method)
    # strategical_time_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, beta, attack_p,
    #                         method)
    # physical_attack(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, episodes, alpha, attack_p, method)


if __name__ == "__main__":
    main_v2()
