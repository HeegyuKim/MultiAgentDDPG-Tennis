from os import stat
from .model import Actor, Critic, hidden_init
from .utils import OUNoise, ReplayBuffer

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPGAgent:
    def __init__(
        self,
        index,
        random_seed,
        device,
        params
    ):
        self.index = index
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.batch_size = params['batch_size']
        self.tau = params['tau']
        self.gamma = params['gamma']
        self.device = device
        self.num_agents = params['num_agents']

        actor_lr = params['actor_lr']
        actor_weight_decay = params['actor_weight_decay']
        
        critic_lr = params['critic_lr']
        critic_weight_decay = params['critic_weight_decay']

        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=actor_lr, weight_decay=actor_weight_decay
        )

        # Critic Network with Target
        self.critic_local = Critic(self.state_size, self.action_size, self.num_agents, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.num_agents, random_seed).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=critic_lr, weight_decay=critic_weight_decay
        )

        self.hard_update_target()

        # noise process
        self.noise = OUNoise(self.action_size, random_seed)
        # experience replay
        self.memory = ReplayBuffer(
            params['batch_size'],
            params['buffer_size'],
            random_seed,
            device
            )

    def reset(self):
        self.noise.reset()
    
    def act(self, state, noise=1.0):
        """
        state: tensor [state_size] or [batch_size, state_size]
        return: tensor [1 or batch_size, state_size]
        """
        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action += noise * self.noise.sample()

        # clip to action range(-1 ~ 1)
        return np.clip(action, -1, 1)

    def act_target(self, state):
        """
        state: tensor [state_size] or [batch_size, state_size]
        return: tensor [1 or batch_size, state_size]
        """
        return self.actor_target(state)
        
    def learn(self, opponent):
        if not self.memory.ready():
            return
        experiences = self.memory.sample()
        states, states_opponent, actions, actions_opponent, rewards, next_states, next_states_opponent, dones = experiences


        # update critic
        next_actions = self.actor_target(next_states)
        next_actions_opponent = opponent.actor_target(next_states_opponent)

        Q_target_next = self.critic_target(
            next_states,
            next_states_opponent,
            next_actions,
            next_actions_opponent
        )
        Q_target = rewards + self.gamma * Q_target_next * (1.0 - dones)
        Q_expected = self.critic_local(states, states_opponent, actions, actions_opponent)

        Q_loss= F.mse_loss(Q_target, Q_expected)
        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()


        # update actor
        actions = self.actor_local(states)
        actor_rewards = -self.critic_local(states, states_opponent, actions, actions_opponent).mean()

        self.actor_optimizer.zero_grad()
        actor_rewards.backward()
        self.actor_optimizer.step()

        self.soft_update_target()
        

    def hard_update_target(self):
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        self.soft_update(self.actor_local, self.actor_target, 1.0)

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
            self.critic_local.state_dict(),
            f"models/agent-{self.index}/{filename}_critic.pth",
        )

class MADDPGAgents:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        random_seed,
        device,
        params
    ):
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.num_agents = params['num_agents']
        self.device = device

        self.gamma = params['gamma']
        self.tau = params['tau']
        self.noise = 1.0
        self.noise_decay = params['noise_decay']
        self.agents = [DDPGAgent(i, random_seed, device, params) for i in range(self.num_agents)]


    def step(self, state, action, reward, next_state, done):
        """
        state = [num_agents, state_size]
        action = [num_agents, action_size]
        reward = []


        save in memory
        state = [num_agents, state_size]

        """
        
        for i, agent in enumerate(self.agents):
            me = i
            opp = 1 - i

            agent.memory.add(state[me], state[opp], action[me], action[opp], reward[me], next_state[me], next_state[opp], done[me])

    def act(self, state):
        """
        state: [num_agents, state_per_agent]
        returns: [num_agents, action_per_agent]
        """

        actions = [agent.act(s, self.noise) for s, agent in zip(state, self.agents)]
        self.noise *= self.noise_decay

        return actions
    def act_random(self):

        return np.random.uniform(-1, 1, (self.num_agents, self.action_size))

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def start_learn(self):
        for i, agent in enumerate(self.agents):
            agent.learn(self.agents[1 - i])
    
    def save(self, filename="checkpoint"):
        for agent in self.agents:
            agent.save(filename)
