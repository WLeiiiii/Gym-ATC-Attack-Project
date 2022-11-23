import datetime
import os
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt

from agents.dqn_agent_simple_env import Agent
from envs.SimpleATC_env_global import SimpleEnv
from envs.SimpleATC_env_local import SimpleEnvLocal
from envs.SimpleATC_env_local_x import SimpleEnvLocalX

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path="save_model/dqn_model_01.pth"):
    total_rewards = []
    reward_window = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf  # compare with reward_window to check save model or not
    for i in range(1, n_episodes + 1):
        episode_timestep = 0
        last_ob = env.reset()
        episode_experience = []  # experience in one episode
        total_reward = 0
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
            total_reward += reward

        agent.add(episode_experience, env)

        reward_window.append(total_reward)
        total_rewards.append(total_reward)
        epsilon = max(eps_end, decay * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)))
        if i >= 1000:
            if last_mean_reward < np.mean(reward_window) or i % 100 == 0:
                torch.save(agent.local.state_dict(), save_path)
                print('\rEpisode {}\tAverage Score: {:.2f}\tPrevious Score: {:.2f}'.format(i, np.mean(reward_window),
                                                                                           last_mean_reward))
                print('Model saved')
                last_mean_reward = np.mean(reward_window)

    torch.save(agent.local.state_dict(), save_path)

    env.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_rewards)), total_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    pass


def evaluate(env, agent, save_path):
    goal_num = 0
    collision_num = 0
    wall_num = 0
    conflict_num = 0
    total_timestep = 0
    total_rewards = []
    results = []
    # frames = []
    reward_window = deque(maxlen=300)
    agent.local.load_state_dict(torch.load(save_path))
    for i in range(300):
        frames = []
        last_ob = env.reset()
        done = False
        total_reward = 0
        # reward_window = deque(maxlen=100)
        episode_timestep = 0
        while not done and episode_timestep < 500:
            observation = np.copy(last_ob)
            action = agent.act(observation)
            # frames.append(env.render())
            env.render()
            for ___ in range(2):
                new_ob, reward, done, _ = env.step(action)
                if done:
                    # print("{}".format(total_reward))
                    break
            # ob, reward, done, _ = env.step(action)
            last_ob = new_ob
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
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(frames[0])
        # plt.show()

    print("测试结束")
    print("----------DQN_without_attack-----------")
    print("goal_num: {}/300".format(goal_num))
    print("collision_num: {}/300".format(collision_num))
    print("wall_num: {}/300".format(wall_num))
    print("conflict_frq: {}".format(conflict_frq))
    print("mean_score: {}".format(total_rewards[-1]))
    print("---------------------------------------")
    # display_frames_as_gif(frames)
    env.close()
    # display_frames_as_gif(frames)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_rewards)), total_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title("DQN_without_attack")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=19999)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--save_path', type=str, default='../save_model/dqn_Xlines_model_01.pth')
    parser.add_argument('--load_path', type=str, default='../save_model/dqn_Xlines_model_01.pth')
    args = parser.parse_args()

    env = SimpleEnv()  # fixed airways with global perception
    # env = SimpleEnvLocal()  # fixed airways with local perception
    # env = SimpleEnvLocalX()  # random airways with local perception
    print('state dimension:', env.observation_space.shape)
    print('action dimension:', env.action_space.n)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.train:
        train(env, agent, n_episodes=args.episodes, save_path=args.save_path)

    evaluate(env, agent, save_path=args.load_path)


if __name__ == "__main__":
    main()
