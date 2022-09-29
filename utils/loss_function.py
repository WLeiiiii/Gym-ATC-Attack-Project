import torch
from torch.distributions import Beta


def worst_action_loss():
    n = Beta(torch.tensor([1.]), torch.tensor([1.]))
    return n


print(Beta(1., 1.).sample().data)
print(Beta(torch.tensor([1.]), torch.tensor([1.])).sample())
