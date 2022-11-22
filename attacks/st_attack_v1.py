import numpy as np
import torch
from torch import nn

from attacks.attack_v1 import Attack, Variable


class STAttack(Attack):
    def __init__(self, env, device, agent, load_path,  epsilon, atk, episodes, method, beta):
        super().__init__(env, device, agent, load_path, epsilon, atk, episodes, method)
        self.beta = beta

    def attack(self, obs_tensor):
        obs = Variable(obs_tensor.data, requires_grad=True)
        if self.can_attack:
            action = self.agent.act(obs_tensor)
            action = torch.from_numpy(action).to(self.device)
            logits = self.agent.forward(obs)
            softmax = nn.Softmax(dim=-1)
            logsoftmax = nn.LogSoftmax(dim=-1)
            prob = logsoftmax(logits)
            prob_np = softmax(logits).cpu().detach().numpy()
            prob_r = prob_np[0][0] + prob_np[0][1] + prob_np[0][2]
            prob_k = prob_np[0][3] + prob_np[0][4] + prob_np[0][5]
            prob_l = prob_np[0][6] + prob_np[0][7] + prob_np[0][8]
            prob_heading = [prob_r, prob_k, prob_l]
            max_a = np.amax(prob_heading)
            min_a = np.amin(prob_heading)
            diff = max_a - min_a
            if diff >= self.beta:
                if self.method == "F":
                    obs = self.fgsm(obs, action, prob)
                else:
                    obs = self.gradient_based_attack(obs, action, prob)
                if self.epsilon != 0:
                    self.attack_counts += 1
        return obs.data
        pass

