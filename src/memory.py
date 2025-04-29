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

    def generate_batches(self, next_value, last_done): # Add parameters
        """Generates batches for PPO training."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Convert lists to tensors before batching - Use np.array first for efficiency
        states_np = np.array(self.states)
        if states_np.ndim == 1:
             states_np = np.stack(states_np)

        states_tensor = torch.tensor(states_np, dtype=torch.float).to(config.DEVICE)
        actions_tensor = torch.tensor(np.array(self.actions), dtype=torch.long).to(config.DEVICE)
        log_probs_tensor = torch.tensor(np.array(self.log_probs), dtype=torch.float).to(config.DEVICE)
        values_tensor = torch.tensor(np.array(self.values), dtype=torch.float).to(config.DEVICE)
        rewards_tensor = torch.tensor(np.array(self.rewards), dtype=torch.float).to(config.DEVICE)
        dones_tensor = torch.tensor(np.array(self.dones), dtype=torch.bool).to(config.DEVICE)


        # Calculate advantages and returns using GAE, passing bootstrap info
        advantages, returns = self._calculate_gae(rewards_tensor, values_tensor, dones_tensor, next_value, last_done)

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8: # Avoid division by zero or near-zero
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            # If std is zero, advantages are constant. If mean is also 0, they are all 0.
            # If mean is non-zero, normalization is tricky. Setting to 0 is a common heuristic.
            advantages = (advantages - advantages.mean()) # Center advantages if std is zero
            # Alternatively, could set advantages to zero: advantages = torch.zeros_like(advantages)


        return (states_tensor, actions_tensor, log_probs_tensor,
                values_tensor, advantages, returns, batches) # Return values_tensor (old values)

    def _calculate_gae(self, rewards, values, dones, next_value, last_done): # Add parameters
        """Calculates Generalized Advantage Estimation (GAE) and returns."""
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(config.DEVICE)
        returns = torch.zeros_like(rewards).to(config.DEVICE)
        last_advantage = 0.0

        # Use the provided next_value for bootstrap if the trajectory didn't end
        # This is V(s_{T+1}) where T is the last step in the collected trajectory
        bootstrap_value = torch.tensor(next_value, dtype=torch.float).to(config.DEVICE) * (1.0 - float(last_done))

        # Calculate advantages and returns backwards
        current_next_value = bootstrap_value # Start with V(s_{T+1})
        for t in reversed(range(n_steps)):
            mask = 1.0 - dones[t].float() # Mask for terminal states *within* the trajectory
            # TD Error (delta)
            # delta = r_t + gamma * V(s_{t+1}) * mask - V(s_t)
            # Note: mask here applies to dones[t], ensuring V(s_{t+1}) is zero if s_t was terminal
            delta = rewards[t] + config.GAMMA * current_next_value * mask - values[t]
            # GAE Advantage
            # A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1}) * mask
            last_advantage = delta + config.GAMMA * config.GAE_LAMBDA * mask * last_advantage
            advantages[t] = last_advantage
            # Return (target for value function) = GAE Advantage + V(s_t)
            returns[t] = advantages[t] + values[t]
            # Update next_value for the previous step (t-1) -> V(s_t) becomes the next_value for step t-1
            current_next_value = values[t]


        return advantages, returns # Return both advantages and returns

    def __len__(self):
        return len(self.states)

