import torch
import numpy as np
from models import ActorCritic
from memory import PPOMemory
from ppo import PPOAlgorithm

class PPOAgent:
    """Agent that interacts with the environment and learns using PPO."""
    def __init__(self, state_dim, action_dim, cfg, is_continuous=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg # Store config
        self.is_continuous = is_continuous

        # Initialize Actor-Critic model
        self.actor_critic = ActorCritic(state_dim, action_dim, cfg['HIDDEN_SIZE'], is_continuous=is_continuous).to(cfg['DEVICE'])

        # Initialize PPO Algorithm
        self.ppo_algorithm = PPOAlgorithm(self.actor_critic, cfg)

        # Initialize Memory buffer
        self.memory = PPOMemory(cfg, is_continuous=is_continuous)

    def select_actions(self, states):
        """Selects actions for a batch of states using the current policy."""
        # states is expected to be a numpy array (num_actors, state_dim)
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.cfg['DEVICE'])
        with torch.no_grad():
            actions, log_probs, values = self.actor_critic.act(states_tensor)
            
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy() # Return batch values

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Stores a transition in the memory buffer."""
        self.memory.store_memory(state, action, log_prob, reward, value, done)

    def learn(self, next_value, last_done):
        """Triggers the PPO update step."""
        self.ppo_algorithm.update(self.memory, next_value, last_done)
        self.memory.clear_memory() # Clear memory after update

    def load_model(self, path):
        """Loads the Actor-Critic model state."""
        print(f"Loading model...")
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.cfg['DEVICE']))
        self.actor_critic.eval() # Set to evaluation mode

    def save_model(self, path):
        """Saves the Actor-Critic model state."""
        print(f"\nSaving model...")
        torch.save(self.actor_critic.state_dict(), path)
