import datetime
import os
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt, animation

from agents.two_dqn_agent_simple_env import Agent2
from envs.SimpleATC_env_five_2_agent import SimpleEnv
from utils.display_plt import display_plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# def display_frames_as_gif(frames):
#     # print(frames[0])
#     patch = plt.imshow(frames[0])
#     plt.axis('off')
#
#     def animate(i):
#         patch.set_data(frames[i])
#     print("生成gif...")
#     anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
#     anim.save('../logs/results/gifs/' + 'result_dqn_' + current_time + '.gif', writer='pillow', fps=30)
#     print("gif已保存！")


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path="save_model/dqn_model_01.pth"):
    total_rewards_j = []
    reward_window_j = deque(maxlen=100)
    total_rewards_p = []
    reward_window_p = deque(maxlen=100)
    epsilon = eps_start
    # epsilon = 0
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
            # print(observation)
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
                torch.save(agent.local_j.state_dict(), save_path)
                print('\rEpisode {}\tAverage Score_j+p: {:.2f}\tPrevious Score_j+p: {:.2f}'.format(i, np.mean(
                    reward_window_j) + np.mean(reward_window_p),
                                                                                                   last_mean_reward))
                print('Model saved')
                last_mean_reward = np.mean(reward_window_j)

    torch.save(agent.local_j.state_dict(), save_path)

    env.close()

    display_plt(total_rewards_j, 'Episode', 'Score', "DQN_j")
    display_plt(total_rewards_p, 'Episode', 'Score', "DQN_p")


# def evaluate(env, agent, save_path):
#     goal_num = 0
#     collision_num = 0
#     wall_num = 0
#     conflict_num = 0
#     total_timestep = 0
#     total_rewards = []
#     results = []
#     # frames = []
#     reward_window = deque(maxlen=300)
#     # env = gym.wrappers.Monitor(env, './videos/' + 'SingleAircraft3env' + current_time)
#     agent.local_j.load_state_dict(torch.load(save_path))
#     for i in range(300):
#         frames = []
#         last_ob = env.reset()
#         done = False
#         total_reward = 0
#         # reward_window = deque(maxlen=100)
#         episode_timestep = 0
#         while not done and episode_timestep < 500:
#             observation = np.copy(last_ob)
#             # print(observation.shape)
#             action = agent.act(observation)
#             # frames.append(env.render())
#             env.render()
#             for ___ in range(2):
#                 new_ob, reward, done, _ = env.step(action)
#                 if done:
#                     # print("{}".format(total_reward))
#                     break
#             # ob, reward, done, _ = env.step(action)
#             last_ob = new_ob
#             total_reward += reward
#
#             episode_timestep += 1
#         # total_rewards.append(total_reward)
#         results = env.terminal_info()
#         total_timestep += episode_timestep
#         goal_num += results[0]
#         conflict_num += results[1]
#         collision_num += results[2]
#         wall_num += results[3]
#         conflict_frq = conflict_num / total_timestep
#         reward_window.append(total_reward)
#         total_rewards.append(np.mean(reward_window))
#         print(total_rewards[-1])
#         # plt.xticks([])
#         # plt.yticks([])
#         # plt.imshow(frames[0])
#         # plt.show()
#
#     print("测试结束")
#     print("----------DQN_without_attack-----------")
#     print("goal_num: {}/300".format(goal_num))
#     print("collision_num: {}/300".format(collision_num))
#     print("wall_num: {}/300".format(wall_num))
#     print("conflict_frq: {}".format(conflict_frq))
#     print("mean_score: {}".format(total_rewards[-1]))
#     print("---------------------------------------")
#     # display_frames_as_gif(frames)
#     env.close()
#     # display_frames_as_gif(frames)
#     fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     plt.plot(np.arange(len(total_rewards)), total_rewards)
#     plt.ylabel('Score')
#     plt.xlabel('Episode')
#     plt.title("DQN_without_attack")
#     plt.show()
#     pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=19999)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=9)
    # parser.add_argument('--load_path', type=str, default="save_model/dqn_model_01.pth")
    parser.add_argument('--save_path', type=str, default='../save_model/dqn_random_goal_model_08.pth')
    args = parser.parse_args()

    env = SimpleEnv()
    print('state dimension:', env.observation_space.shape)
    print('action dimension:', env.action_space.n)

    # env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True

    agent = Agent2(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.train:
        train(env, agent, n_episodes=args.episodes, save_path=args.save_path)

    # evaluate(env, agent, save_path=args.save_path)


if __name__ == "__main__":
    main()
