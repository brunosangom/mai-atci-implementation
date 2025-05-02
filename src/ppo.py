import torch
import torch.optim as optim

class PPOAlgorithm:
    """Proximal Policy Optimization algorithm logic."""
    def __init__(self, actor_critic_model, cfg):
        self.actor_critic = actor_critic_model
        self.cfg = cfg
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg['LEARNING_RATE'])

    def update(self, memory, next_values, last_dones):
        """Performs PPO update."""
        # Generate batches using memory, providing bootstrap info
        states, actions, log_probs, advantages, returns, batches = memory.generate_batches(next_values, last_dones)

        for _ in range(self.cfg['PPO_EPOCHS']):
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate current policy
                new_values, new_log_probs, entropy = self.actor_critic.evaluate_actions(batch_states, batch_actions)

                # Calculate ratio
                ratio = (new_log_probs - batch_log_probs).exp()

                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg['PPO_EPSILON'], 1.0 + self.cfg['PPO_EPSILON']) * batch_advantages

                # Actor loss (Policy loss)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (Value loss)
                critic_loss = (new_values - batch_returns).pow(2).mean()

                # Total loss
                loss = actor_loss + self.cfg['CRITIC_DISCOUNT'] * critic_loss - self.cfg['ENTROPY_BETA'] * entropy

                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
