import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class PPOAlgorithm:
    """Proximal Policy Optimization algorithm logic."""
    def __init__(self, actor_critic_model, cfg):
        self.actor_critic = actor_critic_model
        self.cfg = cfg
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg['LEARNING_RATE'])
        # Initialize logs
        self.actor_loss_log = []
        self.critic_loss_log = []
        self.entropy_log = []

    def update(self, memory, next_values, last_dones):
        """Performs PPO update and logs average losses for the update cycle."""
        # Generate batches using memory, providing bootstrap info
        states, actions, log_probs, advantages, returns, batches = memory.generate_batches(next_values, last_dones)

        # Store losses for this update cycle (across all epochs and batches)
        update_actor_losses = []
        update_critic_losses = []
        update_entropies = []

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
                discounted_critic_loss = self.cfg['CRITIC_DISCOUNT'] * critic_loss
                entropy_loss = self.cfg['ENTROPY_BETA'] * entropy

                # Store batch losses (before weighting)
                update_actor_losses.append(actor_loss.item())
                update_critic_losses.append(discounted_critic_loss.item())
                update_entropies.append(entropy_loss.item())

                # Total loss
                loss = actor_loss + discounted_critic_loss - entropy_loss

                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Log the average losses for this update cycle
        if update_actor_losses:
            self.actor_loss_log.append(np.mean(update_actor_losses))
        if update_critic_losses:
            self.critic_loss_log.append(np.mean(update_critic_losses)) 
        if update_entropies:
            self.entropy_log.append(np.mean(update_entropies))

    def save_loss_logs(self, results_dir):
        """Saves the logged actor loss, discounted critic loss, and entropy to a CSV and generates a plot."""
        # Check if all logs have data
        if not self.actor_loss_log or not self.critic_loss_log or not self.entropy_log:
            print("No loss/entropy data logged, skipping saving logs.")
            return

        num_updates = len(self.actor_loss_log)
        # Ensure all logs have the same length, otherwise something went wrong
        if not (len(self.critic_loss_log) == num_updates and len(self.entropy_log) == num_updates):
             print(f"Warning: Loss/entropy log lengths mismatch. Actor: {num_updates}, Critic: {len(self.critic_loss_log)}, Entropy: {len(self.entropy_log)}. Skipping log saving.")
             return

        log_df = pd.DataFrame({
            'update_cycle': range(num_updates),
            'actor_loss': self.actor_loss_log,
            'critic_loss': self.critic_loss_log, # Updated column name
            'entropy': self.entropy_log # Added entropy column
        })

        csv_path = os.path.join(results_dir, "loss_log.csv")
        plot_path = os.path.join(results_dir, "loss_plot.png")

        try:
            # Save log to CSV
            log_df.to_csv(csv_path, index=False)
            print(f"Loss log saved to {csv_path}")

            # Generate and save plot
            plt.figure(figsize=(12, 7)) # Adjusted figure size
            plt.plot(log_df['update_cycle'], log_df['actor_loss'], label='Actor Loss', marker='.')
            plt.plot(log_df['update_cycle'], log_df['critic_loss'], label='Critic Loss', marker='.') # Updated label
            plt.plot(log_df['update_cycle'], log_df['entropy'], label='Entropy', marker='.') # Added entropy plot
            plt.title(f"PPO Loss & Entropy Curves") # Updated title
            plt.xlabel("Update Cycle")
            plt.ylabel("Value") # Changed y-axis label
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close() # Close the plot to free memory
            print(f"Loss plot saved to {plot_path}")

        except Exception as e:
            print(f"Error saving loss log or plot: {e}")
