import os
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import autograd, nn
from torch.distributions.beta import Beta
import torch.nn.functional as F

from utils.display_plt import display_plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class Attack:
    def __init__(self, env, device, agent, load_path, epsilon=0, atk=False, episodes=300, method="F"):
        self.method = method
        self.env = env
        self.epsilon = epsilon
        self.device = device
        self.agent = agent
        self.agent.load_state_dict(torch.load(load_path))
        print('model loaded successfully from %s' % load_path)
        self.agent.eval()
        self.can_attack = atk
        self.frames = []
        self.goal_num = 0
        self.collision_num = 0
        self.conflict_num = 0
        self.total_timestep = 0
        self.max_step_num = 0
        self.attack_frequency = 0
        self.total_rewards = []
        self.attack_frq = []
        self.episodes = episodes
        self.reward_window = deque(maxlen=episodes)
        self.attack_frq_window = deque(maxlen=episodes)
        self.steps_num_mean = []
        self.steps_num_window = deque(maxlen=episodes)

    def run(self):
        for _ in range(self.episodes):
            self.attack_counts = 0
            obs = self.env.reset()
            done = False
            episode_timestep = 0
            total_reward = 0
            observation = np.copy(obs)
            while not done and episode_timestep < 500:
                episode_timestep += 1
                ori_obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
                obs_tensor = self.attack(ori_obs_tensor)
                self.attack_frequency = self.attack_counts / episode_timestep
                with torch.no_grad():
                    action = self.agent.act(obs_tensor)
                # self.frames.append(self.env.render())
                self.env.render()
                for ___ in range(2):
                    obs_next, reward, done, _ = self.env.step(action)
                    if done:
                        break
                total_reward += reward
                observation = obs_next
            results = self.env.terminal_info()
            self.total_timestep += episode_timestep
            self.goal_num += results[0]
            self.conflict_num += results[1]
            self.collision_num += results[2]
            self.max_step_num += results[3]
            self.average_steps = results[4]
            conflict_frq = self.conflict_num / self.total_timestep
            self.reward_window.append(total_reward)
            self.total_rewards.append(np.mean(self.reward_window))
            self.attack_frq_window.append(self.attack_frequency)
            self.attack_frq.append(np.mean(self.attack_frq_window))
            print(self.total_rewards[-1], self.attack_frq[-1])
        self.env.close()
        # display_frames_as_gif(self.frames)
        print("------------Results-------------")
        print("goal_num: {}/{}".format(self.goal_num, self.episodes))
        print("collision_num: {}/{}".format(self.collision_num, self.episodes))
        print("max_step_num: {}/{}".format(self.max_step_num, self.episodes))
        print("steps_mean: {}".format(self.average_steps))
        print("conflict_frq: {:4f}".format(conflict_frq))
        print("mean_score: {:4f}".format(self.total_rewards[-1]))
        print("attack_frq: {:4f}".format(self.attack_frq[-1]))
        print("-------------------------------")
        pass

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.can_attack:
            action = self.agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = self.agent.forward(obs)
            logsoftmax = nn.LogSoftmax(dim=-1)
            prob = logsoftmax(logits)
            if self.method == "F":
                obs = self.fgsm(obs, action, prob)
            else:
                obs = self.gradient_based_attack(obs, action, prob)
            if self.epsilon != 0:
                self.attack_counts += 1
        return obs.data
        pass

    def plot(self, atk_name="Without_attack"):
        display_plt(self.total_rewards, 'Episode', 'Score', atk_name)
        if self.can_attack:
            display_plt(self.attack_frq, 'Episode', 'Frequency', atk_name)
        pass

    def save_data(self, atk_name="Without_attack"):
        header = ["score"]
        df = pd.DataFrame(columns=header, data=self.total_rewards)
        df.to_csv("../save_data/" + atk_name + ".csv")

        result = [[self.goal_num / self.episodes, self.collision_num / self.episodes, self.max_step_num / self.episodes,
                   self.average_steps, self.conflict_num / self.total_timestep, self.total_rewards[-1],
                   self.attack_frq[-1]]]
        headers = ["Success rate", "Collision rate", "Max Steps Reached rate", "Mean Steps", "Conflict rate",
                   "Mean score", "Attack Frequency"]
        df = pd.DataFrame(columns=headers, data=result)
        df.to_csv("../save_data/" + atk_name + "_results.csv")
        pass

    def fgsm(self, obs, action, prob):
        loss = F.nll_loss(prob, action)
        self.agent.zero_grad()
        loss.backward()
        eta = self.epsilon * obs.grad.data.sign()
        obs = Variable(obs.data + eta, requires_grad=True)
        obs.data = torch.clamp(obs.data, 0, 1)
        return obs
        pass

    def gradient_based_attack(self, obs, action, prob):
        obs_adv = obs
        action_index = action.data.cpu().numpy()[-1]
        q_star = prob[-1][action_index]
        loss = -1 * min(prob[-1])
        self.agent.zero_grad()
        loss.backward()
        for _ in range(9):
            eta = Beta(1, 1).sample().data * obs.grad.data.sign()
            obs_i = Variable(obs.data - eta, requires_grad=True)
            obs_i.data = torch.clamp(obs_i.data, 0, 1)
            action_adv = self.agent.act(obs_i)
            action_adv_tensor = torch.from_numpy(action_adv).to(self.device)
            q_adv = prob[-1][action_adv_tensor.data.cpu().numpy()[-1]]
            if q_adv < q_star:
                q_star = q_adv
                obs_adv = obs_i
        return obs_adv


pass
