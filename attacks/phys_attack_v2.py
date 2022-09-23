import torch
from torch import nn

from attacks.attack_v2 import Attack, Variable


class PhysAttack(Attack):
    def __init__(self, env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk, attack_p, episodes,
                 alpha):
        super().__init__(env, device, agent_j, agent_p, load_path_j, load_path_p, epsilon, atk, attack_p, episodes)
        self.alpha = alpha

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
            near_dist = self.env.near_intruder_dist()
            goal_dist = self.env.goal_dist()
            if near_dist <= self.alpha or goal_dist <= self.alpha:
                obs = self.fgsm(obs, action, prob, agent)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
