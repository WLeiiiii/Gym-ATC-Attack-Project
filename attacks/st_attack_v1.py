import numpy as np
import torch
from torch import nn

from attacks.attack_v1 import Attack, Variable


class STAttack(Attack):
    def __init__(self, env, device, agent, load_path,  epsilon, atk, episodes, beta):
        super().__init__(env, device, agent, load_path, epsilon, atk, episodes)
        self.beta = beta

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.can_attack:
            action = self.agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = self.agent.forward(obs)
            softmax = nn.Softmax(dim=-1)
            prob = softmax(logits)
            prob_np = softmax(logits).cpu().detach().numpy()
            prob_a = prob_np[0][0] + prob_np[0][1] + prob_np[0][2]
            prob_b = prob_np[0][3] + prob_np[0][4] + prob_np[0][5]
            prob_c = prob_np[0][6] + prob_np[0][7] + prob_np[0][8]
            prob_heading = [prob_a, prob_b, prob_c]
            max_a = np.amax(prob_heading)
            min_a = np.amin(prob_heading)
            diff = max_a - min_a
            if diff >= self.beta:
                obs = self.fgsm(obs, action, prob)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
        pass

