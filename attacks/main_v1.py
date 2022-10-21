import random

import numpy as np
import torch

from attacks.attack_v1 import Attack
from attacks.st_attack_v1 import STAttack
from attacks.uni_attack_v1 import UniAttack
from envs.SimpleATC_env_five import SimpleEnv
from models.dqn_model import QNetwork, ResBlock


def without_attack(env, device, agent, load_path, epsilon, episodes):
    without_atk = Attack(env, device, agent, load_path, epsilon, atk=False, episodes=episodes)
    without_atk.run()
    without_atk.plot(atk_name="Without Attack_convDQN_10")
    without_atk.save_data(atk_name="Without_Attack_convDQN_10")


def uniform_attack(env, device, agent, load_path, epsilon, episodes, frq, method):
    uni_atk = UniAttack(env, device, agent, load_path, epsilon, atk=True, episodes=episodes, frq=frq, method=method)
    uni_atk.run()
    uni_atk.plot(atk_name="UniAttack with epsilon:{} & frq:{}_convDQN_{}_5".format(epsilon, frq, method))
    uni_atk.save_data(atk_name="UniAttack with epsilon:{} & frq:{}_convDQN_{}_5".format(epsilon, frq, method))
    pass


def strategical_time_attack(env, device, agent, load_path, epsilon, episodes, beta, method):
    st_atk = STAttack(env, device, agent, load_path, epsilon, atk=True, episodes=episodes, beta=beta, method=method)
    st_atk.run()
    st_atk.plot(atk_name="STAttack with epsilon:{} & beta:{}_convDQN_{}".format(epsilon, beta, method))
    pass


def main():
    episodes = 500
    epsilon = 0.03
    beta = 0.025
    alpha = 100
    frq = 0.5
    method = "F"  # "F":fgsm, "G":gradient based attack

    seed = 9
    random.seed(seed)
    np.random.seed(seed)
    env = SimpleEnv()
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    load_path = "../save_model_new/dqn_10lines_model_01.pth"

    without_attack(env, device, agent, load_path, epsilon, episodes)
    # uniform_attack(env, device, agent, load_path, epsilon, episodes, frq, method)
    # strategical_time_attack(env, device, agent, load_path, epsilon, episodes, beta, method)


if __name__ == "__main__":
    main()
