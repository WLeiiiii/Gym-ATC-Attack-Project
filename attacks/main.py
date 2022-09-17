import random

import numpy as np
import torch

from attacks.attack import Attack
from attacks.phys_attack_v1 import PhysAttack
from attacks.st_attack_v1 import STAttack
from attacks.uni_attack_v1 import UniAttack
from envs.SimpleATC_env_five import SimpleEnv
from models.dqn_model import QNetwork, ResBlock


def without_attack(env, device, agent, load_path, epsilon, episodes):
    without_atk = Attack(env, device, agent, load_path, epsilon, atk=False, episodes=episodes)
    without_atk.run()
    without_atk.plot(atk_name="Without Attack")


def uniform_attack(env, device, agent, load_path, epsilon, episodes, frq):
    uni_atk = UniAttack(env, device, agent, load_path, epsilon, atk=True, episodes=episodes, frq=frq)
    uni_atk.run()
    uni_atk.plot(atk_name="UniAttack with epsilon:{} & frq:{}".format(epsilon, frq))
    pass


def strategical_time_attack(env, device, agent, load_path, epsilon, episodes, beta):
    st_atk = STAttack(env, device, agent, load_path, epsilon, atk=True, episodes=episodes, beta=beta)
    st_atk.run()
    st_atk.plot(atk_name="STAttack with epsilon:{} & beta:{}".format(epsilon, beta))
    pass


def physical_attack(env, device, agent, load_path, epsilon, episodes, alpha):
    phys_atk = PhysAttack(env, device, agent, load_path, epsilon, atk=True, episodes=episodes, alpha=alpha)
    phys_atk.run()
    phys_atk.plot(atk_name="PhysAttack with epsilon:{} & alpha:{}".format(epsilon, alpha))
    pass


def main():
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    env = SimpleEnv()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(device)
    load_path = "../save_model/dqn_random_goal_model_06.pth"

    episodes = 300
    epsilon = 0.03
    beta = 0.03
    alpha = 100
    frq = 0.5

    without_attack(env, device, agent, load_path, epsilon, episodes)
    uniform_attack(env, device, agent, load_path, epsilon, episodes, frq)
    strategical_time_attack(env, device, agent, load_path, epsilon, episodes, beta)
    physical_attack(env, device, agent, load_path, epsilon, episodes, alpha)


if __name__ == "__main__":
    main()
