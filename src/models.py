import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO.
    Handles both discrete (Categorical) and continuous (Normal) action spaces.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, is_continuous=False, shared_features=False):
        super(ActorCritic, self).__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim
        self.shared_features = shared_features

        if shared_features:
            # Shared layer for both actor and critic
            self.shared_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh()
            )

            # Actor head
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim) # Outputs logits for discrete, mu for continuous
            )

            # Critic head: outputs state value
            self.critic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Separate layers for actor and critic
            self.actor_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim) # Outputs logits for discrete, mu for continuous
            )
            self.critic_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # Learnable parameter for log standard deviation (for continuous actions)
        if self.is_continuous:
            # Initialize log_std close to zero for stable initial exploration
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        """
        Forward pass through the network.
        Args:
            state: The input state.
        Returns:
            actor_output: Action logits (discrete) or action mean (continuous).
            state_value: Estimated value of the state (from Critic).
        """
        if self.shared_features:
            shared_features = self.shared_layer(state)
            actor_output = self.actor_head(shared_features)
            state_value = self.critic_head(shared_features)
        else:
            actor_output = self.actor_net(state)
            state_value = self.critic_net(state)
        return actor_output, state_value

    def _get_distribution(self, actor_output):
        """Creates the appropriate distribution based on the action space type."""
        if self.is_continuous:
            mu = actor_output # Output from actor head is the mean
            log_std = self.actor_log_std.expand_as(mu) # Expand log_std to match batch size
            std = torch.exp(log_std)
            dist = Normal(mu, std)
        else:
            logits = actor_output # Output from actor head is logits
            dist = Categorical(logits=logits)
        return dist

    def evaluate_actions(self, state, actions):
        """
        Evaluate actions using the current policy and value estimate.
        Used during PPO updates. Handles both continuous and discrete actions.
        Args:
            state: The state batch.
            actions: The actions batch (long for discrete, float for continuous).
        Returns:
            value: State values.
            log_probs: Log probabilities of the actions taken.
            entropy: Policy entropy.
        """
        actor_output, value = self.forward(state)
        dist = self._get_distribution(actor_output)

        # Ensure action has the correct shape for log_prob calculation, especially for multi-dim continuous
        if self.is_continuous and len(actions.shape) < len(actor_output.shape):
             actions = actions.unsqueeze(-1) # Add dimension if needed

        log_probs = dist.log_prob(actions)
        # Sum log_prob across action dimensions for continuous spaces
        if self.is_continuous:
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        else:
            log_probs = log_probs.unsqueeze(-1) # Ensure shape consistency (batch, 1)

        entropy = dist.entropy()
        # Sum entropy across action dimensions for continuous spaces
        if self.is_continuous:
            entropy = entropy.sum(dim=-1)

        return value, log_probs.squeeze(-1), entropy.mean() # Return value (batch, 1), log_probs (batch,), entropy (scalar)


    def act(self, state):
        """
        Select an action based on the current policy. Handles both action spaces.
        Used during interaction with the environment.
        Args:
            state: The current state (can be a batch).
        Returns:
            action: The selected action(s) (tensor).
            log_prob: The log probability of the selected action(s) (tensor).
            value: The estimated state value(s) (tensor).
        """
        actor_output, value = self.forward(state)
        dist = self._get_distribution(actor_output)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Sum log_prob across action dimensions for continuous spaces
        if self.is_continuous:
            log_prob = log_prob.sum(dim=-1)

        return action, log_prob, value.squeeze(-1) # Return action, log_prob (batch,), value (batch,)

    def get_deterministic_action(self, state):
        """ Get the deterministic action (mean for continuous, argmax for discrete). """
        actor_output, _ = self.forward(state)
        if self.is_continuous:
            action = actor_output # Mean is the deterministic action
        else:
            action = torch.argmax(actor_output, dim=-1)
        return action
