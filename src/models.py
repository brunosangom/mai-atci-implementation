import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO.
    Shares layers for feature extraction, then splits into Actor and Critic heads.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: outputs action logits (raw scores)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # REMOVED nn.Softmax(dim=-1) - Categorical distribution expects logits
        )

        # Critic head: outputs state value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Forward pass through the network.
        Args:
            state: The input state.
        Returns:
            action_logits: Logits over actions (from Actor).
            state_value: Estimated value of the state (from Critic).
        """
        shared_features = self.shared_layer(state)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_logits, state_value # Return logits

    def evaluate_actions(self, state, action):
        """
        Evaluate actions using the current policy and value estimate.
        Used during PPO updates.
        Args:
            state: The state batch.
            action: The action batch.
        Returns:
            value: State values.
            log_probs: Log probabilities of the actions taken.
            entropy: Policy entropy.
        """
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits) # Use logits=...
        log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return value, log_probs, entropy

    def act(self, state):
        """
        Select an action based on the current policy.
        Used during interaction with the environment.
        Args:
            state: The current state.
        Returns:
            action: The selected action.
            log_prob: The log probability of the selected action.
            value: The estimated state value (added return)
        """
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits) # Use logits=...
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value # Return value as well
