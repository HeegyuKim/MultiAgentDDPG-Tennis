from collections import deque, namedtuple
import random
import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, batch_size, buffer_size, seed, device):
        """Initialize a ReplayBuffer object."""
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "state_opponent", "action", "action_opponent", "reward", "next_state", "next_state_opponent", "done"])
        self.device = device
        random.seed(seed)
    
    def add(self, state, state_opponent, action, action_opponent, reward, next_state, next_state_opponent, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_opponent, action, action_opponent, reward, next_state, next_state_opponent, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        states_opponent = torch.from_numpy(np.vstack([e.state_opponent for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        actions_opponent = torch.from_numpy(np.vstack([e.action_opponent for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        next_states_opponent = torch.from_numpy(np.vstack([e.next_state_opponent for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, states_opponent, actions, actions_opponent, rewards, next_states, next_states_opponent, dones)

    def ready(self):
        """Return the current size of internal memory."""
        return len(self.memory) >= self.batch_size