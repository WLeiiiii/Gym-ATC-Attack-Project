import datetime
import os
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt

from agents.dqn_agent_simple_env_img import AgentImg
from envs.SimpleATC_env_img import SimpleImgEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path="save_model/dqn_stack_model_01.pth"):
    total_rewards = []
    reward_window = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf
    for i in range(1, n_episodes + 1):
        episode_timestep = 0
        last_ob = env.reset()
        # print(last_ob.shape)
        episode_experience = []  # experience in one episode
        total_reward = 0
        done = False
        while not done and episode_timestep < 5000:
            env.render()
            observation = np.copy(last_ob)
            # plt.imshow(observation[0][0]/255.)
            # print(observation.shape)
            # desired_goal = np.copy(last_ob['desired_goal'])
            # inputs = np.concatenate([observation, desired_goal], axis=-1)
            episode_timestep += 10
            action = agent.act(observation, epsilon)

            for ___ in range(5):
                new_ob, reward, done, _ = env.step(action)
                if not done:
                    break

            new_observation = np.copy(new_ob)
            agent.step(observation, action, reward, new_observation, done)
            # episode_experience.append((observation, action, reward, new_observation, done))
            last_ob = new_ob
            total_reward += reward


        reward_window.append(total_reward)
        total_rewards.append(total_reward)
        epsilon = max(eps_end, decay * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}\tSteps: {}'.format(i, np.mean(reward_window), episode_timestep),
              end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tSteps: {}'.format(i, np.mean(reward_window), episode_timestep),
                  end="")
        if i >= 1000:
            if last_mean_reward < np.mean(reward_window) or i % 100 == 0:
                torch.save(agent.local.state_dict(), save_path)
                print(
                    '\rEpisode {}\tAverage Score: {:.2f}\tPrevious Score: {:.2f}\tSteps: {}'.format(i, np.mean(
                        reward_window),
                                                                                                    last_mean_reward,
                                                                                                    episode_timestep))
                print('Model saved')
                last_mean_reward = np.mean(reward_window)
    torch.save(agent.local.state_dict(), save_path)

    env.close()
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_rewards)), total_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=30000)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--save_path', type=str, default='../save_model/dqn_img_model_02.pth')
    parser.add_argument('--load_path', type=str, default="../save_model/dqn_img_model_01.pth")
    args = parser.parse_args()

    env = SimpleImgEnv()
    agent = AgentImg(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.load_path:
        agent.local.load_state_dict(torch.load(args.load_path))
        print('model loaded successfully from %s' % args.load_path)

    if args.train:
        train(env, agent, n_episodes=args.episodes, save_path=args.save_path)

    # evaluate(env, agent, save_path=args.save_path)
    pass


if __name__ == "__main__":
    main()
