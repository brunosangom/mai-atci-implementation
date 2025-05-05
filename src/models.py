import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# Helper class for running mean and standard deviation
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        # These will be registered as buffers in the main module
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = torch.tensor(epsilon, dtype=torch.float32)

    def update(self, x):
        batch_mean = torch.mean(x, dim=0).to(self.mean.device)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32).to(self.mean.device)

        # Only update variance if batch size is greater than 1
        if batch_count > 1:
            batch_var = torch.var(x, dim=0, unbiased=True).to(self.mean.device)
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            # Only update mean if batch size is 1
            self.update_from_moments(batch_mean, self.var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = batch_count + self.count

        # Update the buffers in-place
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)

    @property
    def std(self):
        # Ensure std calculation is on the correct device
        return torch.sqrt(self.var).to(self.mean.device)

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO with Observation Normalization.
    Handles both discrete (Categorical) and continuous (Normal) action spaces.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, is_continuous=False, shared_features=False, use_obs_norm=True, obs_clip=5.0):
        super(ActorCritic, self).__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim
        self.shared_features = shared_features
        self.use_obs_norm = use_obs_norm
        self.obs_clip = obs_clip

        # Observation Normalization Filter
        if self.use_obs_norm:
            self._obs_rms = RunningMeanStd(shape=state_dim)
            # Register buffers
            self.register_buffer('obs_rms_mean', self._obs_rms.mean)
            self.register_buffer('obs_rms_var', self._obs_rms.var)
            self.register_buffer('obs_rms_count', self._obs_rms.count)
        else:
            self._obs_rms = None # Keep internal reference None

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
            # Initialize final layer weights with normc
            self._init_weights(self.actor_head[-1], gain=0.01)
            self._init_weights(self.critic_head[-1], gain=1.0)

        else:
            # Separate networks for actor and critic
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
            # Initialize final layers similar to TF normc(0.01) for actor and normc(1.0) for critic
            self._init_weights(self.actor_net[-1], gain=0.01)
            self._init_weights(self.critic_net[-1], gain=1.0)


        # Learnable parameter for log standard deviation (for continuous actions)
        if self.is_continuous:
            # Initialize log_std close to zero for stable initial exploration
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def _init_weights(self, layer, gain=1.0):
        """Initialize weights using orthogonal initialization scaled by gain."""
        def normc_init(tensor, gain=1.0):
            with torch.no_grad():
                tensor.normal_(0, 1)
                tensor *= gain / tensor.norm(2, dim=0, keepdim=True)
                
        if isinstance(layer, nn.Linear):
            normc_init(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    @property
    def obs_rms(self):
        """Provides access to the RunningMeanStd object using the registered buffers."""
        if self._obs_rms is not None:
            # Update the internal object's state from the buffers before returning
            self._obs_rms.mean = self.obs_rms_mean
            self._obs_rms.var = self.obs_rms_var
            self._obs_rms.count = self.obs_rms_count
            return self._obs_rms
        return None

    def _normalize_obs(self, obs):
        """Normalize observations using the running mean and std buffers."""
        if self.use_obs_norm and self.obs_rms is not None:
            # Avoid division by zero
            epsilon = 1e-8
            normalized_obs = (obs - self.obs_rms_mean) / (torch.sqrt(self.obs_rms_var) + epsilon)
            # Clip observations
            normalized_obs = torch.clamp(normalized_obs, -self.obs_clip, self.obs_clip)
            return normalized_obs
        return obs

    def update_obs_rms(self, obs_batch):
        """Update the running mean and std buffers from a batch of observations."""
        if self.use_obs_norm and self.obs_rms is not None:
            self.obs_rms.update(obs_batch)

    def forward(self, state):
        """
        Forward pass through the network with observation normalization.
        Args:
            state: The input state.
        Returns:
            actor_output: Action logits (discrete) or action mean (continuous).
            state_value: Estimated value of the state (from Critic).
        """
        # Normalize observations first
        normalized_state = self._normalize_obs(state)

        if self.shared_features:
            shared_features = self.shared_layer(normalized_state)
            actor_output = self.actor_head(shared_features)
            state_value = self.critic_head(shared_features)
        else:
            actor_output = self.actor_net(normalized_state)
            state_value = self.critic_net(normalized_state)
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
        actor_output, value = self.forward(state) # Forward pass uses normalized state internally
        dist = self._get_distribution(actor_output)

        # Ensure action has the correct shape for log_prob calculation
        if self.is_continuous and len(actions.shape) < len(actor_output.shape):
             actions = actions.unsqueeze(-1)

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

        # Return value (batch, 1), log_probs (batch,), entropy (scalar)
        return value, log_probs.squeeze(-1), entropy.mean()


    def act(self, state, update_rms=False):
        """
        Select an action based on the current policy. Handles both action spaces.
        Used during interaction with the environment. Optionally updates RMS.
        Args:
            state: The current state (can be a batch).
            update_rms (bool): Whether to update the running mean/std with this state.
        Returns:
            action: The selected action(s) (tensor).
            log_prob: The log probability of the selected action(s) (tensor).
            value: The estimated state value(s) (tensor).
        """
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
             # Add batch dim if single state, move to model's device
             state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.obs_rms_mean.device if self.use_obs_norm else next(self.parameters()).device)
        else:
             state = state.to(self.obs_rms_mean.device if self.use_obs_norm else next(self.parameters()).device)


        # Optionally update RMS before normalization in forward pass
        if update_rms and self.training and self.use_obs_norm: # Only update RMS during training rollouts if enabled
            self.update_obs_rms(state)

        with torch.no_grad():
            actor_output, value = self.forward(state) # Forward pass uses normalized state internally
            dist = self._get_distribution(actor_output)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Sum log_prob across action dimensions for continuous spaces
            if self.is_continuous:
                log_prob = log_prob.sum(dim=-1)

        # Return action, log_prob (batch,), value (batch,)
        return action, log_prob, value.squeeze(-1)

    def get_deterministic_action(self, state):
        """ Get the deterministic action (mean for continuous, argmax for discrete). """
         # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
             # Add batch dim if single state, move to model's device
             state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.obs_rms_mean.device if self.use_obs_norm else next(self.parameters()).device)
        else:
             state = state.to(self.obs_rms_mean.device if self.use_obs_norm else next(self.parameters()).device)


        with torch.no_grad():
            actor_output, _ = self.forward(state) # Forward pass uses normalized state internally
            if self.is_continuous:
                action = actor_output # Mean is the deterministic action
            else:
                action = torch.argmax(actor_output, dim=-1)
        return action
