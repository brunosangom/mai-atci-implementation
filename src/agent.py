import torch
import numpy as np
from models import ActorCritic
from memory import PPOMemory
from ppo import PPOAlgorithm

class PPOAgent:
    """Agent that interacts with the environment and learns using PPO."""
    # Accept cfg dictionary
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg # Store config

        # Initialize Actor-Critic model using cfg['HIDDEN_SIZE'] and cfg['DEVICE']
        self.actor_critic = ActorCritic(state_dim, action_dim, cfg['HIDDEN_SIZE']).to(cfg['DEVICE'])

        # Initialize PPO Algorithm handler, passing cfg
        self.ppo_algorithm = PPOAlgorithm(self.actor_critic, cfg)

        # Initialize Memory buffer using cfg['MINI_BATCH_SIZE'], passing cfg
        self.memory = PPOMemory(cfg['MINI_BATCH_SIZE'], cfg)

    # Use cfg['DEVICE']
    def select_actions(self, states):
        """Selects actions for a batch of states using the current policy."""
        # states is expected to be a numpy array (num_actors, state_dim)
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.cfg['DEVICE'])
        with torch.no_grad():
            # act now returns actions, log_probs, values for the batch
            actions, log_probs, values = self.actor_critic.act(states_tensor)
        # Return numpy arrays for interaction with vectorized env
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy() # Return batch values

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Stores a transition in the memory buffer."""
        self.memory.store_memory(state, action, log_prob, reward, value, done)

    # learn method passes info to algorithm, no change needed here
    def learn(self, next_value, last_done): # Add parameters
        """Triggers the PPO update step."""
        # Pass next_value and last_done for GAE calculation
        self.ppo_algorithm.update(self.memory, next_value, last_done)
        self.memory.clear_memory() # Clear memory after update

    # Use cfg['DEVICE'] in load_model
    def load_model(self, path):
        """Loads the Actor-Critic model state."""
        print(f"Loading model...")
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.cfg['DEVICE']))
        self.actor_critic.eval() # Set to evaluation mode

    def save_model(self, path):
        """Saves the Actor-Critic model state."""
        print(f"\nSaving model...")
        torch.save(self.actor_critic.state_dict(), path)
