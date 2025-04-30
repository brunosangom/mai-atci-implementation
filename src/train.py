import gymnasium as gym
import torch
import numpy as np
import os
from datetime import datetime
from agent import PPOAgent
import imageio
# CONFIG is already imported in main.py and passed to train_agent

def evaluate_agent(env_name, agent, eval_episodes, device):
    """Evaluates the agent over a number of episodes."""
    env = gym.make(env_name)
    total_rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                # Only need action for evaluation, ignore log_prob and value
                action_logits, _ = agent.actor_critic(state_tensor)
                action = torch.argmax(action_logits, dim=-1).item() # Choose best action deterministically

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    env.close()
    return np.mean(total_rewards)

def generate_renders(env_name, agent, num_renders, save_dir, device):
    """Generates GIF renders of the agent playing."""
    print(f"Generating GIF renders...")
    os.makedirs(save_dir, exist_ok=True)
    # Create a temporary env with rgb_array rendering
    render_env = gym.make(env_name, render_mode="rgb_array")

    for i in range(num_renders):
        frames = []
        state, _ = render_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Capture frame before taking action
            frames.append(render_env.render())

            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                action_logits, _ = agent.actor_critic(state_tensor)
                action = torch.argmax(action_logits, dim=-1).item() # Deterministic action

            next_state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        # Save the collected frames as a GIF
        gif_path = os.path.join(save_dir, f"render_episode_{i}.gif")
        imageio.mimsave(gif_path, frames, fps=30) # Adjust fps as needed
        print(f"  Render {i+1}/{num_renders} finished. Reward: {episode_reward:.2f}")

    render_env.close() # Close the base env
    print("GIF Renders generated.")

def train_agent(cfg):
    """Trains the PPO agent."""
    print(f"Using device: {cfg['DEVICE']}")
    print(f"Training on environment: {cfg['ENV_NAME']}")
    print("Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    # --- Initialization ---
    env = gym.make(cfg['ENV_NAME'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, cfg)

    # Create directories for saving models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("ppo_models", f"ppo_{cfg['ENV_NAME']}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved in: {model_dir}")


    total_steps = 0
    episode = 0
    best_eval_reward = -np.inf

    # --- Training Loop ---
    state, _ = env.reset()

    while total_steps < cfg['NUM_STEPS']:
        # --- Data Collection Phase ---
        for step in range(cfg['PPO_STEPS']):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, log_prob, reward, value, done)
            state = next_state
            total_steps += 1

            if done:
                episode += 1
                state, _ = env.reset()

            # Break if total steps reached during collection
            if total_steps >= cfg['NUM_STEPS']:
                break

        # --- Learning Phase ---
        # Bootstrap value estimation for the last state if not done
        last_done = done
        if not last_done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(cfg['DEVICE'])
            with torch.no_grad():
                 _, next_value = agent.actor_critic(state_tensor)
                 next_value = next_value.cpu().numpy().item() # Get scalar value
        else:
            next_value = 0.0 # Terminal state value is 0

        agent.learn(next_value, last_done) # Pass bootstrap info

        # --- Evaluation Phase ---
        # Evaluate the agent periodically (e.g., after each learning cycle)
        if episode > 0: # Start evaluating after at least one learning cycle
            avg_eval_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['EVAL_EPISODES'], cfg['DEVICE'])
            print(f"Evaluation after {total_steps} steps: Average Reward = {avg_eval_reward:.2f}")

            # Save the model if it's the best so far
            if avg_eval_reward > best_eval_reward and total_steps >= cfg['NUM_STEPS'] * 0.3:
                best_eval_reward = avg_eval_reward
                save_path = os.path.join(model_dir, f"ppo_{cfg['ENV_NAME']}_{timestamp}.pth")
                agent.save_model(save_path)

                # Generate renders if requested
                if cfg.get('RENDER', False):
                    render_save_dir = os.path.join(model_dir, "renders", f"{avg_eval_reward:.2f}")
                    generate_renders(cfg['ENV_NAME'], agent, 3, render_save_dir, cfg['DEVICE'])
            
            if best_eval_reward >= cfg['GOAL_REWARD']:
                print(f"Goal reward of {cfg['GOAL_REWARD']} reached! Stopping training.")
                break


    # --- Final Evaluation & Saving ---
    print("\nTraining finished.")
    # Load the best model for final evaluation
    best_model_path = os.path.join(model_dir, f"ppo_{cfg['ENV_NAME']}_{timestamp}.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation.")

        best_agent = PPOAgent(state_dim, action_dim, cfg)
        best_agent.load_model(best_model_path)
        final_avg_reward = evaluate_agent(cfg['ENV_NAME'], best_agent, cfg['TEST_EPISODES'], cfg['DEVICE'])
        print(f"Final Evaluation (Best Model): Average Reward = {final_avg_reward:.2f}")
    else:
        print("No best model was saved during training. Evaluating current agent state.")
        final_avg_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['TEST_EPISODES'], cfg['DEVICE'])
        print(f"Final Evaluation (Current Agent): Average Reward = {final_avg_reward:.2f}")


    env.close()
    print("Training complete.")

