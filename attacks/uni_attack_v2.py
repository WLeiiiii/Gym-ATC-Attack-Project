import torch
from torch import nn

from attacks.attack_v2 import Attack, Variable


class UniAttack(Attack):
    def __init__(self, env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk, attack_p, episodes,
                 frq=1.0):
        super().__init__(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk, attack_p, episodes)
        self.frq = frq

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.attack_p:
            agent = self.agent_p
        else:
            agent = self.agent_j
        agent.eval()
        if self.can_attack:
            action = agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = agent.forward(obs)
            softmax = nn.Softmax(dim=-1)
            prob = softmax(logits)
            if self.attack_frequency <= self.frq:
                obs = self.fgsm(obs, action, prob, agent)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
