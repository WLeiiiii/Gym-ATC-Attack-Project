import numpy as np
import random
from collections import namedtuple, deque

from models.dqn_model_img import DQN

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3  # parameter for soft update
LEARNING_RATE = 1e-4
UPDATE_EVERY = 5


class AgentImg():
    '''AgentImg interacts with the environment'''

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # double network
        self.local = DQN().to(device)
        # move model to either gpu or cpu
        self.target = DQN().to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=LEARNING_RATE)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0:
            # If enough samples in memory, generate batch and learn
            if len(self.memory.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, epsilon=0.):
        '''Choose an action given state using epsilon-greedy'''
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().to(device)
        self.local.eval()  # change model to evaluation mode (turn off dropout)
        with torch.no_grad():  # turn off gradient descent since evaluating
            q_value = self.local(state)
        self.local.train()  # back to train mode

        if random.random() > epsilon:
            return np.argmax(q_value.cpu().data.numpy())  # move q_value to cpu
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''learning from batch'''
        states, actions, rewards, next_states, dones = experiences
        # get the max predicted q value for next state

        q_target_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)

        # detach the variable from the graph using detach()
        q_target = rewards + (gamma * q_target_next * (1 - dones))

        # if episode is done q_target equals to reward only
        q_expected = self.local(states).gather(1, actions)

        # gradient step
        loss = F.mse_loss(q_expected, q_target)

        self.optimizer.zero_grad()  # zero gradient if not pytorch will accmulate
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local, self.target, TAU)

    def soft_update(self, local_net, target_net, tau):
        '''soft update for parameters
        θ_target = tau*θ_local + (1 - tau)*θ_target
        '''
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer():
    '''Buffer stores experience'''

    def __init__(self, action_size, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.action_size = action_size
        self.batch_size = batch_size
        self.experience = namedtuple('experience', field_names=['state', 'action',
                                                                'reward', 'next_state',
                                                                'done'])

    def add(self, state, action, reward, next_state, done):
        '''add new memory'''
        memory = self.experience(state, action, reward, next_state, done)
        self.memory.append(memory)

    def sample(self):
        '''random sample a batch'''
        experiences = random.sample(self.memory, k=self.batch_size)  # return a list of experience

        # restack the list of experience into batch form
        # np.vstack stacks lists verticallly
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)
