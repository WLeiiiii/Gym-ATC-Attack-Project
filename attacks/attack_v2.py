import os
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import autograd, nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

from utils.display_plt import display_plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class Attack:
    def __init__(self, env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon=0, atk=False, attack_p=False,
                 episodes=300, method="F"):
        self.method = method
        self.env = env
        self.epsilon = epsilon
        self.device = device
        self.agent_j = agent_j
        self.agent_p = agent_p
        self.attack_p = attack_p
        self.agent_j.load_state_dict(torch.load(load_path_j))
        self.agent_p.load_state_dict(torch.load(load_path_p))
        print('model loaded successfully from {} & {}'.format(load_path_j, load_path_p))
        self.agent_j.eval()
        self.agent_p.eval()
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
        self.attack_frq_window = deque(maxlen=episodes)
        self.total_rewards_j = []
        self.reward_window_j = deque(maxlen=episodes)
        self.total_rewards_p = []
        self.reward_window_p = deque(maxlen=episodes)
        self.steps_num_mean = []
        self.steps_num_window = deque(maxlen=episodes)

    def run(self):
        for _ in range(self.episodes):
            self.attack_counts = 0
            obs = self.env.reset()
            done = False
            episode_timestep = 0
            total_reward = [0, 0]
            observation = np.copy(obs)
            while not done and episode_timestep < 500:
                episode_timestep += 1
                ori_obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
                obs_tensor = self.attack(ori_obs_tensor)

                self.attack_frequency = self.attack_counts / episode_timestep
                with torch.no_grad():
                    action = self.act(obs_tensor)
                # self.frames.append(self.env.render())
                self.env.render()
                for ___ in range(2):
                    obs_next, reward, done, _ = self.env.step(action)
                    if done:
                        break
                total_reward[0] += reward[0]
                total_reward[1] += reward[1]
                observation = obs_next
            results = self.env.terminal_info()
            self.total_timestep += episode_timestep
            self.goal_num += results[0]
            self.conflict_num += results[1]
            self.collision_num += results[2]
            self.max_step_num += results[3]
            self.average_steps = results[4]
            conflict_frq = self.conflict_num / self.total_timestep
            self.reward_window_j.append(total_reward[0])
            self.total_rewards_j.append(np.mean(self.reward_window_j))
            self.reward_window_p.append(total_reward[1])
            self.total_rewards_p.append(np.mean(self.reward_window_p))
            self.total_rewards.append(self.total_rewards_j[-1] + self.total_rewards_p[-1])
            self.attack_frq_window.append(self.attack_frequency)
            self.attack_frq.append(np.mean(self.attack_frq_window))
            print(self.total_rewards[-1], self.attack_frq[-1])
        self.env.close()
        # display_frames_as_gif(self.frames)
        print("------------Results-------------")
        print("goal_num: {}/{}".format(self.goal_num, self.episodes))
        print("collision_num: {}/{}".format(self.collision_num, self.episodes))
        print("max_step_num: {}/{}".format(self.max_step_num, self.episodes))
        print("steps_mean: {}".format(int(self.average_steps)))
        print("conflict_frq: {:4f}".format(conflict_frq))
        print("mean_score: {:4f}".format(self.total_rewards[-1]))
        print("attack_frq: {:4f}".format(self.attack_frq[-1]))
        print("--------------------------------")

    def act(self, state, epsilon=0.):
        self.agent_j.eval()
        self.agent_p.eval()
        q_value_j = self.agent_j(state)
        q_value_p = self.agent_p(state)
        q_value = q_value_j + q_value_p
        self.agent_j.train()  # back to train mode
        self.agent_p.train()  # back to train mode

        if random.random() > epsilon:
            for i in range(1, len(q_value[0])):
                index_max = np.argsort(q_value.cpu().data.numpy())[-1][-i]
                if q_value_p.cpu().data.numpy()[0][index_max] < np.mean(q_value_p.cpu().data.numpy()):
                # if q_value_p.cpu().data.numpy()[0][index_max] < np.median(q_value_p.cpu().data.numpy()):
                    continue
                else:
                    return index_max
        else:
            return random.choice(np.arange(9))
        pass

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.attack_p:
            agent = self.agent_p
        else:
            agent = self.agent_j
        agent.eval()
        if self.can_attack:
            action = self.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = agent.forward(obs)
            logsoftmax = nn.LogSoftmax(dim=-1)
            prob = logsoftmax(logits)
            if self.method == "F":
                obs = self.fgsm(obs, action, prob, agent)
            else:
                obs = self.gradient_based_attack(obs, action, prob, agent)
            if self.epsilon != 0:
                self.attack_counts += 1
        return obs.data

    def plot(self, atk_name="Without_attack"):
        display_plt(self.total_rewards, 'Episode', 'Score', atk_name)
        if self.can_attack:
            display_plt(self.attack_frq, 'Episode', 'Frequency', atk_name)

    def save_data(self, atk_name="Without_attack"):
        header = ["score"]
        df = pd.DataFrame(columns=header, data=self.total_rewards)
        df.to_csv("../save_data_new/" + atk_name + ".csv")

        result = [[self.goal_num / self.episodes, self.collision_num / self.episodes, self.max_step_num / self.episodes,
                   self.average_steps, self.conflict_num / self.total_timestep, self.total_rewards[-1],
                   self.attack_frq[-1]]]
        headers = ["Success rate", "Collision rate", "Max Steps Reached rate", "Mean Steps", "Conflict rate",
                   "Mean score", "Attack Frequency"]
        df = pd.DataFrame(columns=headers, data=result)
        df.to_csv("../save_data_new/" + atk_name + "_results.csv")
        pass

    def fgsm(self, obs, action, prob, agent):
        action = torch.from_numpy(np.array(action)).unsqueeze(0).to(self.device)
        loss = F.nll_loss(prob, action)
        agent.zero_grad()
        loss.backward()
        eta = self.epsilon * obs.grad.data.sign()
        obs = Variable(obs.data + eta, requires_grad=True)
        obs.data = torch.clamp(obs.data, 0, 1)
        return obs

    def gradient_based_attack(self, obs, action, prob, agent):
        obs_adv = obs
        action_index = action
        q_star = prob[-1][action_index]
        prob_copy = prob.clone().detach()

        worst_action_index = torch.argmin(prob[-1])
        prob_w = torch.zeros_like(prob)
        prob_w[-1][worst_action_index] = 1

        loss = -1 * torch.matmul(prob_w[-1], prob[-1])
        agent.zero_grad()
        loss.backward()
        for _ in range(10):
            eta = Beta(1, 1).sample().data * obs.grad.data.sign()
            obs_i = Variable(obs.data - eta, requires_grad=True)
            obs_i.data = torch.clamp(obs_i.data, 0, 1)
            action_adv = agent.act(obs_i)
            action_adv_tensor = torch.from_numpy(action_adv).to(self.device)
            q_adv = prob_copy[-1][action_adv_tensor.data.cpu().numpy()[-1]]
            if q_adv < q_star:
                q_star = q_adv
                obs_adv = obs_i
        return obs_adv
