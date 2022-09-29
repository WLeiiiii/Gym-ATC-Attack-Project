import torch
from torch import nn

from attacks.attack_v1 import Attack, Variable


class PhysAttack(Attack):
    def __init__(self, env, device, agent, load_path, epsilon, atk, episodes, method, alpha):
        super().__init__(env, device, agent, load_path, epsilon, atk, episodes, method)
        self.alpha = alpha

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.can_attack:
            action = self.agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = self.agent.forward(obs)
            logsoftmax = nn.LogSoftmax(dim=-1)
            prob = logsoftmax(logits)
            near_dist = self.env.near_intruder_dist()
            goal_dist = self.env.goal_dist()
            if near_dist <= self.alpha or goal_dist <= self.alpha:
                if self.method == "F":
                    obs = self.fgsm(obs, action, prob)
                else:
                    obs = self.gradient_based_attack(obs, action, prob)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
