import random

import torch
from torch import nn

from attacks.attack import Attack, Variable


class UniAttack(Attack):
    def __init__(self, env, device, agent, load_path, epsilon, atk, episodes, frq=1.0):
        super().__init__(env, device, agent, load_path, epsilon, atk, episodes)
        self.frq = frq

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.can_attack:
            action = self.agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = self.agent.forward(obs)
            softmax = nn.Softmax(dim=-1)
            prob = softmax(logits)
            if self.attack_frequency <= self.frq:
                obs = self.fgsm(obs, action, prob)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
