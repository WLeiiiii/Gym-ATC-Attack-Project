import torch
from torch import nn

from attacks.attack_v1 import Attack, Variable


class UniAttack(Attack):
    def __init__(self, env, device, agent, load_path, epsilon, atk, episodes, method, frq=1.0):
        super().__init__(env, device, agent, load_path, epsilon, atk, episodes, method)
        self.frq = frq

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.can_attack:
            action = self.agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = self.agent.forward(obs)
            logsoftmax = nn.LogSoftmax(dim=-1)
            prob = logsoftmax(logits)
            if self.attack_frequency <= self.frq:
                if self.method == "F":
                    obs = self.fgsm(obs, action, prob)
                else:
                    obs = self.gradient_based_attack(obs, action, prob)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
