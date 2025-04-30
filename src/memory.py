import numpy as np
import torch

class PPOMemory:
    """Stores transitions for PPO updates."""
    # Accept cfg dictionary
    def __init__(self, batch_size, cfg):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size
        self.cfg = cfg # Store config

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

    # Use cfg for device and GAE params
    def generate_batches(self, next_value, last_done): # Add parameters
        """Generates batches for PPO training."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Convert lists to tensors before batching - Use np.vstack for states
        try:
            # Use vstack for potentially mixed shapes (e.g., during collection)
            # Ensure states are consistently shaped before vstack if necessary
            states_np = np.vstack(self.states).astype(np.float32)
        except ValueError as e:
            print(f"Error stacking states: {e}")
            # Potentially add debugging: print shapes of self.states elements
            # for i, s in enumerate(self.states):
            #     print(f"State {i} shape: {np.array(s).shape}")
            raise e

        states_tensor = torch.tensor(states_np, dtype=torch.float).to(self.cfg['DEVICE'])
        # Convert other lists directly to numpy arrays before tensor conversion
        actions_tensor = torch.tensor(np.array(self.actions), dtype=torch.long).to(self.cfg['DEVICE'])
        log_probs_tensor = torch.tensor(np.array(self.log_probs), dtype=torch.float).to(self.cfg['DEVICE'])
        values_tensor = torch.tensor(np.array(self.values), dtype=torch.float).to(self.cfg['DEVICE'])
        rewards_tensor = torch.tensor(np.array(self.rewards), dtype=torch.float).to(self.cfg['DEVICE'])
        dones_tensor = torch.tensor(np.array(self.dones), dtype=torch.bool).to(self.cfg['DEVICE'])


        # Calculate advantages and returns using GAE, passing bootstrap info
        advantages, returns = self._calculate_gae(rewards_tensor, values_tensor, dones_tensor, next_value, last_done)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Add epsilon for stability

        return (states_tensor, actions_tensor, log_probs_tensor,
                values_tensor, advantages, returns, batches) # Return values_tensor (old values)

    # Use cfg for device and GAE params
    def _calculate_gae(self, rewards, values, dones, next_value, last_done): # Add parameters
        """Calculates Generalized Advantage Estimation (GAE) and returns."""
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.cfg['DEVICE'])
        # returns = torch.zeros_like(rewards).to(self.cfg['DEVICE']) # Removed, calculated below
        last_advantage = 0.0

        # Use the provided next_value for bootstrap if the trajectory didn't end
        # Ensure next_value is treated as a tensor
        if not isinstance(next_value, torch.Tensor):
            next_value_tensor = torch.tensor([next_value], dtype=torch.float).to(self.cfg['DEVICE'])
        else:
            next_value_tensor = next_value.to(self.cfg['DEVICE'])

        last_value = next_value_tensor * (1.0 - float(last_done))

        # Calculate advantages and returns backwards
        # current_next_value = bootstrap_value # Start with V(s_{T+1})
        for t in reversed(range(n_steps)):
            mask = 1.0 - dones[t].float() # Mask for terminal states *within* the trajectory
            # TD Error (delta)
            delta = rewards[t] + self.cfg['GAMMA'] * last_value * mask - values[t] # Use cfg['GAMMA']
            # GAE Advantage
            last_advantage = delta + self.cfg['GAMMA'] * self.cfg['GAE_LAMBDA'] * mask * last_advantage # Use cfg params
            advantages[t] = last_advantage
            # Update last_value for the previous step (t-1)
            last_value = values[t]

        # Calculate returns = advantages + values
        returns = advantages + values # Calculate returns based on final advantages

        return advantages, returns # Return both advantages and returns

    def __len__(self):
        return len(self.states)

