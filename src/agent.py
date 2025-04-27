\
import torch
import numpy as np
from models import ActorCritic
from memory import PPOMemory
from ppo import PPOAlgorithm
import config

class PPOAgent:
    """Agent that interacts with the environment and learns using PPO."""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize Actor-Critic model
        self.actor_critic = ActorCritic(state_dim, action_dim, config.HIDDEN_SIZE).to(config.DEVICE)

        # Initialize PPO Algorithm handler
        self.ppo_algorithm = PPOAlgorithm(self.actor_critic, config.LEARNING_RATE)

        # Initialize Memory buffer
        self.memory = PPOMemory(config.MINI_BATCH_SIZE)

    def select_action(self, state):
        """Selects an action using the current policy."""
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(config.DEVICE)
        with torch.no_grad():
            action, log_prob = self.actor_critic.act(state_tensor)
            value = self.actor_critic.critic_head(self.actor_critic.shared_layer(state_tensor))
        return action, log_prob.cpu().numpy(), value.cpu().numpy().item() # Return value as well for storage

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Stores a transition in the memory buffer."""
        self.memory.store_memory(state, action, log_prob, reward, value, done)

    def learn(self):
        """Triggers the PPO update step."""
        self.ppo_algorithm.update(self.memory)
        self.memory.clear_memory() # Clear memory after update

    def save_model(self, path):
        """Saves the Actor-Critic model state."""
        print(f"Saving model to {path}...")
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, path):
        """Loads the Actor-Critic model state."""
        print(f"Loading model from {path}...")
        self.actor_critic.load_state_dict(torch.load(path, map_location=config.DEVICE))
        self.actor_critic.eval() # Set to evaluation mode
