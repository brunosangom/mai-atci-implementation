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
    def generate_batches(self, next_values, last_dones): # Accept arrays
        """Generates batches for PPO training, calculating GAE per actor."""
        n_states = len(self.states) # Should be NUM_ACTORS * PPO_STEPS
        expected_states = self.cfg['NUM_ACTORS'] * self.cfg['PPO_STEPS'] # Use self.cfg
        if n_states != expected_states:
             print(f"Warning: Expected {expected_states} states, but got {n_states}. Check data collection.")
             # Handle potential mismatch if needed, e.g., by truncating or erroring
             # For now, proceed assuming it matches, but the warning is important.

        # --- Convert lists to tensors ---
        try:
            states_np = np.vstack(self.states).astype(np.float32)
        except ValueError as e:
            print(f"Error stacking states: {e}")
            raise e

        states_tensor = torch.tensor(states_np, dtype=torch.float).to(self.cfg['DEVICE']) # Use self.cfg
        actions_tensor = torch.tensor(np.array(self.actions), dtype=torch.long).to(self.cfg['DEVICE']) # Use self.cfg
        log_probs_tensor = torch.tensor(np.array(self.log_probs), dtype=torch.float).to(self.cfg['DEVICE']) # Use self.cfg
        # These will be reshaped for GAE calculation
        rewards_np = np.array(self.rewards, dtype=np.float32)
        values_np = np.array(self.values, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.bool_)

        # --- Calculate GAE per actor ---
        all_advantages = []
        all_returns = []

        # Ensure next_values and last_dones are numpy arrays
        next_values_np = np.array(next_values, dtype=np.float32)
        last_dones_np = np.array(last_dones, dtype=np.bool_)

        # Reshape rewards, values, dones for per-actor processing
        # Assuming data was stored sequentially: actor0_step0, actor1_step0, ..., actorN_step0, actor0_step1, ...
        # We need to transpose to get (actor, step)
        rewards_tensor = torch.tensor(rewards_np, dtype=torch.float).to(self.cfg['DEVICE']).view(self.cfg['PPO_STEPS'], self.cfg['NUM_ACTORS']).transpose(0, 1) # Use self.cfg
        values_tensor = torch.tensor(values_np, dtype=torch.float).to(self.cfg['DEVICE']).view(self.cfg['PPO_STEPS'], self.cfg['NUM_ACTORS']).transpose(0, 1) # Use self.cfg
        dones_tensor = torch.tensor(dones_np, dtype=torch.bool).to(self.cfg['DEVICE']).view(self.cfg['PPO_STEPS'], self.cfg['NUM_ACTORS']).transpose(0, 1) # Use self.cfg
        next_values_tensor = torch.tensor(next_values_np, dtype=torch.float).to(self.cfg['DEVICE']) # Shape (num_actors,) # Use self.cfg
        last_dones_tensor = torch.tensor(last_dones_np, dtype=torch.bool).to(self.cfg['DEVICE']) # Shape (num_actors,) # Use self.cfg


        for i in range(self.cfg['NUM_ACTORS']): # Use self.cfg
            actor_rewards = rewards_tensor[i] # Shape (ppo_steps,)
            actor_values = values_tensor[i]   # Shape (ppo_steps,)
            actor_dones = dones_tensor[i]     # Shape (ppo_steps,)
            actor_next_value = next_values_tensor[i] # Scalar tensor
            actor_last_done = last_dones_tensor[i]   # Scalar tensor

            # Calculate GAE and returns for this actor
            advantages_i, returns_i = self._calculate_gae(
                actor_rewards, actor_values, actor_dones,
                actor_next_value, actor_last_done
            )
            all_advantages.append(advantages_i)
            all_returns.append(returns_i)

        # Concatenate results and flatten back to (N,) shape
        # Need to transpose back before flattening to match original order
        advantages_tensor = torch.stack(all_advantages).transpose(0, 1).flatten()
        returns_tensor = torch.stack(all_returns).transpose(0, 1).flatten()
        # Also flatten the values tensor used for training (V(s_t))
        old_values_tensor = values_tensor.transpose(0, 1).flatten()


        # --- Normalize advantages ---
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # --- Generate batch indices ---
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (states_tensor, actions_tensor, log_probs_tensor,
                old_values_tensor, advantages_tensor, returns_tensor, batches) # Return flattened tensors

    # Use cfg for device and GAE params - Now accepts scalar bootstrap info
    def _calculate_gae(self, rewards, values, dones, next_value, last_done): # Accept scalar tensors
        """Calculates Generalized Advantage Estimation (GAE) and returns for a single trajectory."""
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.cfg['DEVICE']) # Use self.cfg
        last_advantage = 0.0

        # Bootstrap value V(s_T) - use next_value if the episode didn't end at the last step
        # Ensure next_value is treated as a tensor scalar
        last_value = next_value * (1.0 - last_done.float())

        # Calculate advantages and returns backwards
        for t in reversed(range(n_steps)):
            # Mask for terminal states *within* the trajectory slice
            mask = 1.0 - dones[t].float()
            # TD Error (delta)
            delta = rewards[t] + self.cfg['GAMMA'] * last_value * mask - values[t] # Use self.cfg
            # GAE Advantage
            last_advantage = delta + self.cfg['GAMMA'] * self.cfg['GAE_LAMBDA'] * mask * last_advantage # Use self.cfg
            advantages[t] = last_advantage
            # Update last_value for the previous step (t-1)
            last_value = values[t] # V(s_t) becomes the 'next value' for step t-1

        # Calculate returns = advantages + values (for this trajectory)
        returns = advantages + values

        return advantages, returns # Return tensors for this actor's trajectory

    def __len__(self):
        return len(self.states)

