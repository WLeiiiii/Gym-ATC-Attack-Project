import torch
from torch import nn

from attacks.attack_v2 import Attack, Variable


class UniAttack(Attack):
    def __init__(self, env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk, attack_p, episodes,
                 method, frq=1.0):
        super().__init__(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk, attack_p, episodes,
                         method)
        self.frq = frq

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.attack_g:
            agent = self.agent_g
        else:
            agent = self.agent_c
        if self.can_attack:
            action = self.act(obs_tensor)
            agent.eval()
            logits = agent.forward(obs)
            logsoftmax = nn.LogSoftmax(dim=-1)
            prob = logsoftmax(logits)
            if torch.rand(1) <= self.frq:
                if self.method == "F":
                    obs = self.fgsm(obs, action, prob, agent)
                else:
                    obs = self.gradient_based_attack(obs, action, prob, agent)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
