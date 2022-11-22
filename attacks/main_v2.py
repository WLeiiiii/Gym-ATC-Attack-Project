import random

import numpy as np
import torch

from attacks.attack_v2 import Attack
from attacks.st_attack_v2 import STAttack
from attacks.uni_attack_v2 import UniAttack
from envs.SimpleATC_env_flexible_v2 import SimpleEnv
from models.dqn_model import QNetwork, ResBlock


def without_attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, episodes):
    without_atk = Attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, atk=False, attack_g=False,
                         episodes=episodes)
    without_atk.run()
    without_atk.plot(atk_name="Without_Attack_safeDQN_X15")
    without_atk.save_data(atk_name="Without_Attack_safeDQN_X15")

def uniform_attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, episodes, frq, attack_g, method):
    uni_atk = UniAttack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, atk=True, attack_p=attack_g,
                        episodes=episodes, frq=frq, method=method)
    uni_atk.run()
    uni_atk.plot(atk_name="UniAttack with epsilon:{} & frq:{}_safeDQN_{}_X10".format(epsilon, frq, method))
    uni_atk.save_data(atk_name="UniAttack_with_epsilon_{}_frq_{}_safeDQN_{}_X10".format(epsilon, frq, method))
    pass


def strategical_time_attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, episodes, beta, attack_g,
                            method):
    st_atk = STAttack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, atk=True, attack_p=attack_g,
                      episodes=episodes, beta=beta, method=method)
    st_atk.run()
    st_atk.plot(atk_name="STAttack with epsilon:{} & beta:{}_safeDQN_{}_X10".format(epsilon, beta, method))
    st_atk.save_data(atk_name="STAttack_with_epsilon_{}_beta_{}_safeDQN_{}_X10".format(epsilon, beta, method))
    pass


def main_v2():
    episodes = 500
    epsilon = 0.03
    beta = 0.005
    alpha = 20
    frq = 0.25
    attack_g = True  # True: attack DQN_p, False: attack DQN_j
    method = "F"

    seed = 9
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SimpleEnv()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_c = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    agent_g = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    load_path_c = "../save_model_new/2dqn_Xlines_model_01_c.pth"
    load_path_g = "../save_model_new/2dqn_Xlines_model_01_g.pth"

    without_attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, episodes)
    # uniform_attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, episodes, frq, attack_g, method)
    # strategical_time_attack(env, device, agent_c, agent_g, load_path_c, load_path_g, epsilon, episodes, beta, attack_g,
    #                         method)


if __name__ == "__main__":
    main_v2()
