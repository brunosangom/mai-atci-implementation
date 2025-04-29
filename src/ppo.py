\
import torch
import torch.optim as optim
import config

class PPOAlgorithm:
    """Proximal Policy Optimization algorithm logic."""
    def __init__(self, actor_critic_model, learning_rate):
        self.actor_critic = actor_critic_model
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

    def update(self, memory, next_value, last_done): # Add parameters
        """Performs PPO update."""
        # Generate batches using memory, providing bootstrap info
        states, actions, old_log_probs, old_values, advantages, returns, batches = memory.generate_batches(next_value, last_done)

        for _ in range(config.PPO_EPOCHS):
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate current policy
                new_values, new_log_probs, entropy = self.actor_critic.evaluate_actions(batch_states, batch_actions)

                # Calculate ratio
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - config.PPO_EPSILON, 1.0 + config.PPO_EPSILON) * batch_advantages

                # Actor loss (Policy loss)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (Value loss)
                critic_loss = (new_values.squeeze() - batch_returns).pow(2).mean()

                # Total loss
                loss = actor_loss + config.CRITIC_DISCOUNT * critic_loss - config.ENTROPY_BETA * entropy

                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
