\
import numpy as np
import torch
import config

class PPOMemory:
    """Stores transitions for PPO updates."""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def generate_batches(self):
        """Generates batches for PPO training."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Convert lists to tensors before batching
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float).to(config.DEVICE)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long).to(config.DEVICE)
        log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float).to(config.DEVICE)
        values_tensor = torch.tensor(self.values, dtype=torch.float).to(config.DEVICE)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float).to(config.DEVICE)
        dones_tensor = torch.tensor(self.dones, dtype=torch.bool).to(config.DEVICE)

        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards_tensor, values_tensor, dones_tensor)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate returns (target for value function)
        returns = advantages + values_tensor

        return (states_tensor, actions_tensor, log_probs_tensor,
                values_tensor, advantages, returns, batches)

    def _calculate_gae(self, rewards, values, dones):
        """Calculates Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards).to(config.DEVICE)
        last_advantage = 0
        last_value = values[-1] # Use the value of the last state

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + config.GAMMA * last_value * mask - values[t]
            last_advantage = delta + config.GAMMA * config.GAE_LAMBDA * mask * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        return advantages

    def __len__(self):
        return len(self.states)

