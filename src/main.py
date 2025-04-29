import gymnasium as gym
import numpy as np
import torch
import config
from agent import PPOAgent
import os
from datetime import datetime

def train():
    """Main training loop."""
    print(f"Using device: {config.DEVICE}")
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    total_steps = 0
    episode = 0
    best_reward = -np.inf

    # Create a directory for saving models if it doesn't exist
    model_dir = "ppo_models"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(model_dir, f"ppo_{config.ENV_NAME}_{timestamp}.pth")


    while total_steps < config.MAX_FRAMES:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        current_episode_steps = 0

        while not done and not truncated:
            # Collect experience (PPO_STEPS)
            for step in range(config.PPO_STEPS):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)

                agent.store_transition(state, action, log_prob, reward, value, done or truncated)
                state = next_state
                episode_reward += reward
                total_steps += 1
                current_episode_steps += 1

                if done or truncated:
                    break # End inner loop if episode finishes before PPO_STEPS

            # Calculate the value of the state after the last step for GAE bootstrap
            next_value = 0.0 # Default to 0 if the episode ended
            last_done_or_truncated = done or truncated
            if not last_done_or_truncated:
                # If the episode didn't end, estimate value of the final next_state
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(config.DEVICE)
                with torch.no_grad():
                    # Get value from critic head
                    shared_features = agent.actor_critic.shared_layer(state_tensor)
                    next_value = agent.actor_critic.critic_head(shared_features).cpu().numpy().item()

            # Only learn if enough steps were collected
            if len(agent.memory) > 0:
                 agent.learn(next_value, last_done_or_truncated) # Pass bootstrap info

            if done or truncated:
                 break # Break outer loop as well

        episode += 1
        print(f"Episode: {episode}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

        # Optional: Evaluate agent periodically
        if episode % config.TEST_EPOCHS == 0:
            test_reward = evaluate_agent(env, agent)
            print(f"----------------------------------------")
            print(f"Evaluation after {episode} episodes: Average Reward: {test_reward:.2f}")
            print(f"----------------------------------------")
            if test_reward > best_reward:
                best_reward = test_reward
                agent.save_model(model_save_path) # Save best model
            if test_reward >= config.TARGET_REWARD:
                print(f"Target reward reached! Stopping training.")
                break

    env.close()
    print("Training finished.")
    # Save the final model
    final_model_path = os.path.join(model_dir, f"ppo_{config.ENV_NAME}_final_{timestamp}.pth")
    agent.save_model(final_model_path)


def evaluate_agent(env, agent, n_episodes=10):
    """Evaluates the agent's performance over n_episodes."""
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            # In evaluation, we typically take the greedy action (highest probability)
            # but for simplicity here, we still sample. For deterministic eval, modify agent.select_action
            action, _, _ = agent.select_action(state) # We don't need log_prob or value for eval
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    train()
