import datetime
import os
import random
from collections import deque

import numpy as np
import torch

from agents.dqn_agent_simple_env_v2 import Agent2
from envs.SimpleATC_env_flexible_v2 import SimpleEnv
from utils.display_plt import display_plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path_c="save_model/dqn_model_01.pth", save_path_g="save_model/dqn_model_01.pth"):
    total_rewards_c = []
    reward_window_c = deque(maxlen=100)
    total_rewards_g = []
    reward_window_g = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf  # compare with reward_window to check save model or not
    for i in range(1, n_episodes + 1):
        episode_timestep = 0
        last_ob = env.reset()
        episode_experience = []  # experience in one episode
        total_reward = [0, 0]
        done = False

        while not done and episode_timestep < 200:
            env.render()
            observation = np.copy(last_ob)
            episode_timestep += 1
            action = agent.act(observation, epsilon)

            for ___ in range(5):
                new_ob, reward, done, _ = env.step(action)
                if done:
                    break

            new_observation = np.copy(new_ob)
            agent.step()
            episode_experience.append((observation, action, reward, new_observation, done))
            last_ob = new_ob
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]

        agent.add(episode_experience, env)

        reward_window_c.append(total_reward[0])
        total_rewards_c.append(total_reward[0])
        reward_window_g.append(total_reward[1])
        total_rewards_g.append(total_reward[1])
        epsilon = max(eps_end, decay * epsilon)
        print('\rEpisode {}\tAverage Score_c: {:.2f}, Score_g: {:.2f}'.format(i, np.mean(reward_window_c),
                                                                              np.mean(reward_window_g)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score_c: {:.2f}, Score_g: {:.2f}'.format(i, np.mean(reward_window_c),
                                                                                  np.mean(reward_window_g)))
        if i >= 1000:
            if last_mean_reward < (np.mean(reward_window_c) + np.mean(reward_window_g)) or i % 100 == 0:
                torch.save(agent.local_c.state_dict(), save_path_c)
                torch.save(agent.local_g.state_dict(), save_path_g)
                print('\rEpisode {}\tAverage Score_c+g: {:.2f}\tPrevious Score_c+g: {:.2f}'.format(i, np.mean(
                    reward_window_c) + np.mean(reward_window_g),
                                                                                                   last_mean_reward))
                print('Model saved')
                last_mean_reward = np.mean(reward_window_c) + np.mean(reward_window_g)

    torch.save(agent.local_c.state_dict(), save_path_c)
    torch.save(agent.local_g.state_dict(), save_path_g)

    env.close()

    display_plt(total_rewards_c, 'Episode', 'Score', "DQN_c")
    display_plt(total_rewards_g, 'Episode', 'Score', "DQN_g")


def evaluate(env, agent, load_path_c, load_path_g):
    goal_num = 0
    collision_num = 0
    wall_num = 0
    conflict_num = 0
    total_timestep = 0
    total_rewards_c = []
    reward_window_c = deque(maxlen=500)
    total_rewards_g = []
    reward_window_g = deque(maxlen=500)

    # frames = []
    agent.local_c.load_state_dict(torch.load(load_path_c))
    agent.local_g.load_state_dict(torch.load(load_path_g))
    for i in range(500):
        # frames = []
        last_ob = env.reset()
        done = False
        total_reward = [0, 0]
        episode_timestep = 0
        while not done and episode_timestep < 500:
            observation = np.copy(last_ob)
            action = agent.act(observation)
            # frames.append(env.render())
            env.render()
            for ___ in range(2):
                new_ob, reward, done, _ = env.step(action)
                if done:
                    break
            last_ob = new_ob
            total_reward[0] += reward[0]
            total_reward[1] += reward[1]
            episode_timestep += 1
        results = env.terminal_info()
        total_timestep += episode_timestep
        goal_num += results[0]
        conflict_num += results[1]
        collision_num += results[2]
        wall_num += results[3]
        conflict_frq = conflict_num / total_timestep
        reward_window_c.append(total_reward[0])
        total_rewards_c.append(np.mean(reward_window_c))
        reward_window_g.append(total_reward[1])
        total_rewards_g.append(np.mean(reward_window_g))
        # print(total_rewards_c[-1], total_rewards_g[-1])
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(frames[0])
        # plt.show()

    print("测试结束")
    print("----------DQN_without_attack-----------")
    print("goal_num: {}/500".format(goal_num))
    print("collision_num: {}/500".format(collision_num))
    print("wall_num: {}/500".format(wall_num))
    print("conflict_frq: {}".format(conflict_frq))
    print("mean_score: {}".format(total_rewards_c[-1] + total_rewards_g[-1]))
    print("---------------------------------------")
    # display_frames_as_gif(frames)
    env.close()
    # display_frames_as_gif(frames)

    display_plt(total_rewards_c, 'Episode', 'Score', "DQN_c")
    display_plt(total_rewards_g, 'Episode', 'Score', "DQN_g")
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=19999)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--save_path_c', type=str, default='../save_model/2dqn_Xlines_model_01_j.pth')
    parser.add_argument('--save_path_g', type=str, default='../save_model/2dqn_Xlines_model_01_p.pth')
    parser.add_argument('--load_path_c', type=str, default='../save_model/2dqn_Xlines_model_01_j.pth')
    parser.add_argument('--load_path_g', type=str, default='../save_model/2dqn_Xlines_model_01_p.pth')
    args = parser.parse_args()

    env = SimpleEnv()
    print('state dimension:', env.observation_space.shape)
    print('action dimension:', env.action_space.n)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = Agent2(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.train:
        train(env, agent, n_episodes=args.episodes)

    evaluate(env, agent, load_path_c=args.load_path_c, load_path_g=args.load_path_g)


if __name__ == "__main__":
    main()
