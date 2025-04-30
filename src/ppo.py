import torch
import torch.optim as optim

class PPOAlgorithm:
    """Proximal Policy Optimization algorithm logic."""
    # Accept cfg dictionary
    def __init__(self, actor_critic_model, cfg):
        self.actor_critic = actor_critic_model
        self.cfg = cfg # Store config
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg['LEARNING_RATE'])

    # Use cfg for update parameters
    def update(self, memory, next_value, last_done): # Add parameters
        """Performs PPO update."""
        # Generate batches using memory, providing bootstrap info
        states, actions, old_log_probs, old_values, advantages, returns, batches = memory.generate_batches(next_value, last_done)

        for _ in range(self.cfg['PPO_EPOCHS']): # Use cfg['PPO_EPOCHS']
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
                # Use cfg['PPO_EPSILON']
                surr2 = torch.clamp(ratio, 1.0 - self.cfg['PPO_EPSILON'], 1.0 + self.cfg['PPO_EPSILON']) * batch_advantages

                # Actor loss (Policy loss)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (Value loss)
                critic_loss = (new_values.squeeze() - batch_returns).pow(2).mean()

                # Total loss
                # Use cfg['CRITIC_DISCOUNT'] and cfg['ENTROPY_BETA']
                loss = actor_loss + self.cfg['CRITIC_DISCOUNT'] * critic_loss - self.cfg['ENTROPY_BETA'] * entropy

                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
