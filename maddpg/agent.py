from os import stat
from .model import Actor, Critic
from .utils import OUNoise, ReplayBuffer

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MADDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        index,
        state_size,
        action_size,
        num_agents,
        random_seed,
        device,
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_size=10000,
        batch_size=512,
        gamma=0.99,
        tau=1e-3,
    ):
        self.index = index
        self.device = device
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.state_size = state_size

        # Actor Network with Target
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_opponent = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=actor_lr
        )
        self.actor_opponent_optimizer = torch.optim.Adam(
            self.actor_opponent.parameters(), lr=actor_lr
        )

        # Critic Network with Target
        self.critic_local = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=critic_lr
        )

        # noise process
        self.noise = OUNoise(action_size, random_seed)
        # experience replay
        self.memory = ReplayBuffer(
            action_size, int(buffer_size), batch_size, random_seed, device
        )

    def step(self, state, action, opponent_action, reward, next_state, done):
        self.memory.add(state, action, opponent_action, reward, next_state, done)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().squeeze().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        # clip to action range(-1 ~ 1)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def start_learn(self):
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, opponent_actions, rewards, next_states, dones = experiences

        self.learn_opponent(states, opponent_actions)
        self.learn_critic(states, actions, opponent_actions, next_states, rewards, dones)
        self.learn_actor(states, opponent_actions)
        self.soft_update_target()

    def learn_opponent(self, states, opponent_actions):
        predicts = self.actor_opponent(states)
        loss = F.mse_loss(predicts, opponent_actions)

        self.actor_opponent_optimizer.zero_grad()
        loss.backward()
        self.actor_opponent_optimizer.step()
    
    def learn_critic(self, states, actions, opponent_actions, next_states, rewards, dones):
        actions_next = self.actor_target(next_states)
        actions_opponent_next = self.actor_opponent(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next, actions_opponent_next)
        Q_targets = rewards + (
            self.gamma * Q_targets_next * (1 - dones)
        )
        Q_expected = self.critic_local(states, actions, opponent_actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip grad for stable learning
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def learn_actor(self, states, opponent_actions):
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred, opponent_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update_target(self):
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filename="checkpoint"):
        torch.save(
            self.actor_local.state_dict(),
            f"models/agent-{self.index}/{filename}_actor.pth",
        )
        torch.save(
            self.actor_opponent.state_dict(),
            f"models/agent-{self.index}/{filename}_actor_opponent.pth",
        )
        torch.save(
            self.critic_local.state_dict(),
            f"models/agent-{self.index}/{filename}_critic.pth",
        )
