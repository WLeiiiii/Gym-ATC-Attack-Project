import os
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import autograd, nn
import torch.nn.functional as F

from agents.two_dqn_agent_simple_env import Agent2
# from attacks.uni_attack import attack
from envs.SimpleATC_env_five_v2 import SimpleEnv
from models.dqn_model import QNetwork, ResBlock
from utils.display_plt import display_plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)
# env = SimpleEnv()

# agent = QNetwork(env.observation_space.shape[0], env.action_space.n, ResBlock, [6, 6, 6]).to(
#     "cuda:0")
# agent.load_state_dict(torch.load("../save_model/dqn_random_goal_model_03.pth"))
# agent.eval()


def attack(agent, obs_tensor):
    agent.local_p.eval()
    action = agent.local_p.act(obs_tensor)
    obs = Variable(obs_tensor.data, requires_grad=True)
    action = torch.from_numpy(action).to("cuda:0")
    logits = agent.local_p.forward(obs)
    softmax = nn.Softmax(dim=-1)
    prob = softmax(logits)
    loss = F.nll_loss(prob, action)
    agent.local_p.zero_grad()
    loss.backward()
    eta = 0.03 * obs.grad.data.sign()
    obs = Variable(obs.data + eta, requires_grad=True)
    obs.data = torch.clamp(obs.data, 0, 1)
    return obs.data


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path_j="save_model/dqn_model_01_j.pth", save_path_p="save_model/dqn_model_01_p.pth"):
    total_rewards_j = []
    reward_window_j = deque(maxlen=100)
    total_rewards_p = []
    reward_window_p = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf  # compare with reward_window to check save model or not
    for i in range(1, n_episodes + 1):
        episode_timestep = 0
        last_ob = env.reset()
        episode_experience = []  # experience in one episode
        total_reward = [0, 0]
        done = False
        observation = np.copy(last_ob)
        while not done and episode_timestep <= 200:
            env.render()
            ori_obs = torch.from_numpy(observation).float().unsqueeze(0).to("cuda:0")
            episode_timestep += 1
            # agent.local.eval()
            if random.random() < 0.9:
                obs = attack(agent, ori_obs)
            else:
                obs = ori_obs
            with torch.no_grad():
                action = agent.act(obs)
            for ___ in range(5):
                new_ob, reward, done, _ = env.step(action)
                if done:
                    break

            agent.step()

            episode_experience.append((observation, action, reward, new_ob, done))
            observation = new_ob
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]
            episode_timestep += 1

        agent.add(episode_experience, env)

        reward_window_j.append(total_reward[0])
        total_rewards_j.append(total_reward[0])
        reward_window_p.append(total_reward[1])
        total_rewards_p.append(total_reward[1])
        epsilon = max(eps_end, decay * epsilon)
        print('\rEpisode {}\tAverage Score_j: {:.2f}, Score_p: {:.2f}'.format(i, np.mean(reward_window_j),
                                                                              np.mean(reward_window_p)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score_j: {:.2f}, Score_p: {:.2f}'.format(i, np.mean(reward_window_j),
                                                                                  np.mean(reward_window_p)))
        if i >= 1000:
            if last_mean_reward < (np.mean(reward_window_j) + np.mean(reward_window_p)) or i % 100 == 0:
                torch.save(agent.local_j.state_dict(), save_path_j)
                torch.save(agent.local_p.state_dict(), save_path_p)
                print('\rEpisode {}\tAverage Score_j+p: {:.2f}\tPrevious Score_j+p: {:.2f}'.format(i, np.mean(
                    reward_window_j) + np.mean(reward_window_p),
                                                                                                   last_mean_reward))
                print('Model saved')
                last_mean_reward = np.mean(reward_window_j) + np.mean(reward_window_p)

    torch.save(agent.local_j.state_dict(), save_path_j)
    torch.save(agent.local_p.state_dict(), save_path_p)

    env.close()

    display_plt(total_rewards_j, 'Episode', 'Score', "DQN_j")
    display_plt(total_rewards_p, 'Episode', 'Score', "DQN_p")


def evaluate(env, agent, save_path="../save_model/dqn_adv_train_model_02.pth"):
    goal_num = 0
    collision_num = 0
    wall_num = 0
    conflict_num = 0
    total_timestep = 0
    total_rewards = []
    reward_window = deque(maxlen=500)
    agent.local.load_state_dict(torch.load(save_path))
    for i in range(500):
        last_ob = env.reset()
        done = False
        total_reward = 0
        # reward_window = deque(maxlen=100)
        episode_timestep = 0
        observation = np.copy(last_ob)
        while not done and episode_timestep < 500:
            # env.render()
            ori_obs = torch.from_numpy(observation).float().unsqueeze(0).to("cuda:0")
            episode_timestep += 1
            agent.local.eval()
            if random.random() < 0.5:
                obs = attack(agent, ori_obs)
            else:
                obs = ori_obs
            with torch.no_grad():
                action = agent.act(obs)
            for ___ in range(2):
                ob, reward, done, _ = env.step(action)
                if done:
                    break

            observation = ob
            total_reward += reward

            episode_timestep += 1
        # total_rewards.append(total_reward)
        results = env.terminal_info()
        total_timestep += episode_timestep
        goal_num += results[0]
        conflict_num += results[1]
        collision_num += results[2]
        wall_num += results[3]
        conflict_frq = conflict_num / total_timestep
        reward_window.append(total_reward)
        total_rewards.append(np.mean(reward_window))
        print(total_rewards[-1])

    print("测试结束")
    # env.close()
    print("----------Adv_training-----------")
    print("goal_num: {}/500".format(goal_num))
    print("collision_num: {}/500".format(collision_num))
    print("wall_num: {}/300".format(wall_num))
    print("conflict_frq: {}".format(conflict_frq))
    print("mean_score: {}".format(total_rewards[-1]))
    # print("attack_frq: {}".format(attack_frq[-1]))
    print("-----------------------------------")
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_rewards)), total_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=29999)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=9)
    # parser.add_argument('--load_path', type=str, default="save_model/dqn_model_01.pth")
    parser.add_argument('--save_path_j', type=str, default='../save_model/dqn_adv_training_model_02_j.pth')
    parser.add_argument('--save_path_p', type=str, default='../save_model/dqn_adv_training_model_02_p.pth')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = SimpleEnv()
    print('state dimension:', env.observation_space.shape)
    print('action dimension:', env.action_space.n)

    agent = Agent2(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    # agent.local.eval()
    if args.train:
        train(env, agent, n_episodes=args.episodes, save_path_j=args.save_path_j, save_path_p=args.save_path_p)

    # evaluate(env, agent, save_path_j=args.save_path_j, save_path_p=args.save_path_p)


if __name__ == "__main__":
    main()