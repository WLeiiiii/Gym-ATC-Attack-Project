import numpy as np
from matplotlib import pyplot as plt

from envs.SimpleATC_env import SimpleEnv
from envs.SimpleATC_env_img import SimpleImgEnv


def main():
    env = SimpleImgEnv()
    ob = env.reset()
    # obs = env.render()
    env.close()
    obs = np.copy(ob)
    # print(obs.shape)
    # print(obs)
    ori_obs = obs[0][0]/ 255.
    # print(ori_obs)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ori_obs)
    plt.show()
    # while True:
    # plt.imshow(ori_obs)
    perturb = np.random.random((200,200))*0.05
    adv_obs = ori_obs + perturb
    plt.xticks([])
    plt.yticks([])
    plt.imshow(perturb)
    plt.show()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(adv_obs)
    plt.show()
    pass


if __name__ == "__main__":
    main()
