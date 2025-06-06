import gymnasium as gym
import torch
import numpy as np
import os
from datetime import datetime
from .agent import PPOAgent
from gymnasium.wrappers import RecordVideo, ClipAction
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_agent(env_name, agent, eval_episodes, device, is_continuous):
    """Evaluates the agent over a number of episodes using a single environment."""
    env = gym.make(env_name)
    if is_continuous:
        env = ClipAction(env) # Clip actions to the environment's bounds

    total_rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                # Get deterministic action from the agent's model
                action_tensor = agent.actor_critic.get_deterministic_action(state_tensor)
                action = action_tensor.cpu().numpy()
                action = action[0] if is_continuous else action.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    env.close()
    return np.mean(total_rewards)

def generate_renders(env_name, agent, num_renders, save_dir, device, is_continuous):
    """Generates video renders of the agent playing using RecordVideo."""
    print(f"Generating video renders...")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_renders):
        # Create a new RecordVideo wrapped env for each render
        temp_video_folder = os.path.join(save_dir, f"temp_render_{i}")
        render_env_base = gym.make(env_name, render_mode="rgb_array")
        if is_continuous:
            render_env_base = ClipAction(render_env_base) # Clip actions

        render_env = RecordVideo(
            render_env_base,
            video_folder=temp_video_folder,
            episode_trigger=lambda x: True, # Record every episode
            name_prefix=f"rl-video-render-{i}" # Unique prefix for this render
        )

        state, _ = render_env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                # Get deterministic action
                action_tensor = agent.actor_critic.get_deterministic_action(state_tensor)
                action = action_tensor.cpu().numpy()
                action = action[0] if is_continuous else action.item()

            next_state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        # Closing the env triggers the video saving by RecordVideo
        render_env.close()

        # --- Rename the generated video file ---
        video_files = glob.glob(os.path.join(temp_video_folder, "*.mp4"))
        if video_files:
            source_path = video_files[0]
            # Define the desired destination path with the reward included in the filename
            destination_path = os.path.join(save_dir, f"episode_{i}-reward_{episode_reward:.2f}.mp4")
            # Rename the file
            os.rename(source_path, destination_path)
            # Remove the temporary directory
            os.rmdir(temp_video_folder)
            print(f"  Render {i+1}/{num_renders} finished. Reward: {episode_reward:.2f}")
        else:
             print(f"  Render {i+1}/{num_renders} finished. Reward: {episode_reward:.2f}. Error: Could not find saved video file in {temp_video_folder}.")


    print("Video Renders generated.\n")

def record_final_evaluation(agent_id, cfg, agent, results_dir_base="results"):
    """
    Performs final evaluation, generates renders, and logs results to a CSV file.
    """
    print(f"\n--- Final Evaluation and Recording for {agent_id} ---")
    device = cfg['DEVICE']
    is_continuous = cfg['is_continuous']

    # 1. Perform final evaluation
    final_avg_reward = evaluate_agent(
        cfg['ENV_NAME'],
        agent,
        cfg['TEST_EPISODES'],
        device,
        is_continuous
    )
    print(f"Final Evaluation: Average Reward over {cfg['TEST_EPISODES']} episodes = {final_avg_reward:.2f}")

    # 2. Generate renders if requested
    if cfg.get('RENDER', False):
        results_dir_current_run = os.path.join(results_dir_base, agent_id)
        render_save_dir = os.path.join(results_dir_current_run, "renders", "final")
        print(f"Generating final evaluation renders in {render_save_dir}...")
        generate_renders(
            cfg['ENV_NAME'],
            agent,
            num_renders=3,
            save_dir=render_save_dir,
            device=device,
            is_continuous=is_continuous
        )
    else:
        print("Skipping final evaluation rendering as per configuration.")

    # 3. Prepare data for CSV
    excluded_keys_for_csv = ['ENV_NAME', 'GOAL_REWARD', 'RENDER', 'DEVICE']
    
    csv_row_data = {'agent_id': agent_id, 'final_average_reward': final_avg_reward}

    for key, value in cfg.items():
        if key not in excluded_keys_for_csv:
            if isinstance(value, (list, dict)): # Serialize complex types to JSON string
                csv_row_data[key] = json.dumps(value)
            else:
                csv_row_data[key] = value
    
    # 4. Append to evaluation_results.csv
    csv_path = os.path.join(results_dir_base, "evaluation_results.csv")
    print(f"Logging final results to {csv_path}")

    try:
        new_data_df = pd.DataFrame([csv_row_data])
        if os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)
            results_df = pd.concat([results_df, new_data_df], ignore_index=True)
        else:
            results_df = new_data_df
        
        results_df.to_csv(csv_path, index=False)
        print(f"Successfully logged final results to {csv_path}")
    except Exception as e:
        print(f"Error saving final evaluation results to CSV: {e}")

def train_agent(cfg):
    """Trains the PPO agent using parallel environments."""
    print(f"Using device: {cfg['DEVICE']}")
    print(f"Training on environment: {cfg['ENV_NAME']} with {cfg['NUM_ACTORS']} actors")
    print("Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    # --- Initialization ---
    # Create a temporary single env to get space info
    temp_env = gym.make(cfg['ENV_NAME'])
    is_continuous = isinstance(temp_env.action_space, gym.spaces.Box)
    state_dim = temp_env.observation_space.shape[0]

    if is_continuous:
        action_dim = temp_env.action_space.shape[0]
        action_low = temp_env.action_space.low
        action_high = temp_env.action_space.high
        print(f"Detected Continuous action space (dim={action_dim}, low={action_low}, high={action_high})")
    else:
        action_dim = temp_env.action_space.n
        action_low, action_high = None, None # Not applicable for discrete
        print(f"Detected Discrete action space (n={action_dim})")
    temp_env.close() # Close the temporary environment

    # Create a function to generate individual environments
    def make_env(env_id):
        def _init():
            env = gym.make(env_id)
            # Apply ClipAction wrapper for continuous environments within the vector env
            if is_continuous:
                env = ClipAction(env)
            return env
        return _init

    # Create the vectorized environment to allow parallel actors
    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg['ENV_NAME']) for _ in range(cfg['NUM_ACTORS'])]
    )

    agent = PPOAgent(state_dim, action_dim, cfg, is_continuous=is_continuous)

    # Create directories for saving models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_id_str = f"ppo_{cfg['ENV_NAME']}_{timestamp}"
    model_dir = os.path.join("ppo_models", agent_id_str)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models and config will be saved in: {model_dir}")

    # Create results directory for this run
    results_dir = os.path.join("results", agent_id_str)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Evaluation results, plots and renders will be saved in: {results_dir}")

    # Save the configuration dictionary as a JSON file
    config_path_1 = os.path.join(model_dir, "config.json")
    config_path_2 = os.path.join(results_dir, "config.json")
    cfg_to_save = cfg.copy() # This will be passed to record_final_evaluation
    try:
        # Add action space info to config before saving
        cfg_to_save['is_continuous'] = is_continuous
        cfg_to_save['action_dim'] = int(action_dim)
        cfg_to_save['state_dim'] = state_dim

        with open(config_path_1, 'w') as f:
            json.dump(cfg_to_save, f, indent=4)
        print(f"Configuration saved to {config_path_1}")
        with open(config_path_2, 'w') as f:
            json.dump(cfg_to_save, f, indent=4)
        print(f"Configuration saved to {config_path_2}")
    except Exception as e:
        print(f"Error saving configuration to {config_path_1}: {e}")

    total_steps = 0
    global_episode_count = 0 # Track total episodes across all actors
    episode_rewards_log = [] # Store rewards for each episode
    update_cycle_count = 0 # Track evaluation/update cycles
    best_eval_reward = -np.inf

    training_log = [] # Initialize list to store training progress
    evaluation_log = [] # Initialize list to store evaluation results

    # --- Training Loop ---
    # Reset all environments and get initial states
    states, _ = envs.reset()

    while total_steps < cfg['NUM_STEPS']:
        # --- Data Collection Phase ---
        cycle_rewards = []
        episode_rewards = np.zeros(cfg['NUM_ACTORS'])
        # Collect PPO_STEPS * NUM_ACTORS transitions in total per cycle
        for step in range(cfg['PPO_STEPS']):
            # Select actions for the batch of states
            actions, log_probs, values = agent.select_actions(states)

            # Step the parallel environments
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds) # Combine termination conditions
            episode_rewards += rewards

            # Store transitions for each actor
            for i in range(cfg['NUM_ACTORS']):
                # Extract individual data points
                state = states[i]
                action = actions[i]
                log_prob = log_probs[i]
                reward = rewards[i]
                value = values[i]
                done = dones[i]

                agent.store_transition(state, action, log_prob, reward, value, done)
                cycle_rewards.append(reward)

                # If an episode finished in one of the environments
                if done:
                    episode_rewards_log.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    global_episode_count += 1
                    # Resetting the env is handled automatically by AsyncVectorEnv

            # Update states for the next iteration
            states = next_states
            total_steps += cfg['NUM_ACTORS'] # Increment by the number of actors

        training_log.append({'steps': total_steps, 'average_reward':np.mean(cycle_rewards)}) # Log the average reward for this cycle
        print(f"Training step {total_steps} / {cfg['NUM_STEPS']} (Episodes: {global_episode_count}): Average Reward for this cycle = {np.mean(cycle_rewards):.2f}")

        # --- Learning Phase ---
        # Bootstrap value estimation for the last states from each actor
        with torch.no_grad():
            # states are the final states after the collection loop
            states_tensor = torch.tensor(states, dtype=torch.float).to(cfg['DEVICE'])
            # Get values for the last states reached by each actor
            _, next_values_tensor = agent.actor_critic(states_tensor)
            # Ensure next_values is a flat numpy array (NUM_ACTORS,)
            next_values = next_values_tensor.squeeze().cpu().numpy()
            # Ensure dones is a numpy boolean array (NUM_ACTORS,)
            last_dones = dones.astype(bool)


        # Pass the actual next values and done flags for each actor to learn
        agent.learn(next_values, last_dones)

        # --- Evaluation Phase ---
        if cfg['EVAL_EPISODES'] > 0:
            # Use the single-env evaluation function
            avg_eval_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['EVAL_EPISODES'], cfg['DEVICE'], is_continuous)
            print(f"Evaluation after {total_steps} steps ({global_episode_count} episodes): Average Reward = {avg_eval_reward:.2f}")

            # Log the evaluation result
            evaluation_log.append({'steps': total_steps, 'update_cycle': update_cycle_count, 'average_reward': avg_eval_reward})
            update_cycle_count += 1

            # Save the model if it's the best so far
            if avg_eval_reward > best_eval_reward and (total_steps >= cfg['NUM_STEPS'] * 0.1 or avg_eval_reward >= cfg['GOAL_REWARD']):
                best_eval_reward = avg_eval_reward
                save_path = os.path.join(model_dir, f"{agent_id_str}.pth")
                agent.save_model(save_path)

                # Generate renders if requested
                if cfg.get('RENDER', False):
                    render_save_dir = os.path.join(results_dir, "renders", f"{avg_eval_reward:.3f}")
                    generate_renders(cfg['ENV_NAME'], agent, 3, render_save_dir, cfg['DEVICE'], is_continuous)
            
            # Check goal reward
            if best_eval_reward >= cfg['GOAL_REWARD']:
                print(f"Goal reward of {cfg['GOAL_REWARD']} reached! Stopping training.")
                break


    # --- Final Evaluation & Saving ---
    print("\nTraining complete.")
    print(f"Average reward over the last 100 training episodes: {np.mean(episode_rewards_log[-100:]):.2f}")

    # --- Save Logs and Plots ---
    if episode_rewards_log:
        rewards_log_df = pd.DataFrame({
            'episode': range(1, len(episode_rewards_log) + 1),
            'reward': episode_rewards_log
        })
        csv_path = os.path.join(results_dir, "rewards_log.csv")
        plot_path = os.path.join(results_dir, "rewards_plot.png")

        try:
            # Save log to CSV
            rewards_log_df.to_csv(csv_path, index=False)
            print(f"Rewards log saved to {csv_path}")

            # Generate and save plot
            plt.figure(figsize=(10, 6))
            plt.plot(rewards_log_df['episode'], rewards_log_df['reward'], marker=None)
            plt.title(f"PPO Rewards over episodes - {cfg['ENV_NAME']}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close() # Close the plot to free memory
            print(f"Rewards plot saved to {plot_path}")

        except Exception as e:
            print(f"Error saving rewards log or plot: {e}")


    if training_log:
        log_df = pd.DataFrame(training_log)
        csv_path = os.path.join(results_dir, "training_log.csv")
        plot_path = os.path.join(results_dir, "training_plot.png")

        try:
            # Save log to CSV
            log_df.to_csv(csv_path, index=False)
            print(f"Training log saved to {csv_path}")

            # Generate and save plot
            plt.figure(figsize=(10, 6))
            plt.plot(log_df['steps'], log_df['average_reward'], marker=None)
            plt.title(f"PPO Training Performance - {cfg['ENV_NAME']}")
            plt.xlabel("Steps")
            plt.ylabel("Average Training Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close() # Close the plot to free memory
            print(f"Training plot saved to {plot_path}")

        except Exception as e:
            print(f"Error saving training log or plot: {e}")
    
    if evaluation_log:
        log_df = pd.DataFrame(evaluation_log)
        csv_path = os.path.join(results_dir, "evaluation_log.csv")
        plot_path = os.path.join(results_dir, "evaluation_plot.png")

        try:
            # Save log to CSV
            log_df.to_csv(csv_path, index=False)
            print(f"Evaluation log saved to {csv_path}")

            # Generate and save plot
            plt.figure(figsize=(10, 6))
            plt.plot(log_df['update_cycle'], log_df['average_reward'], marker=None)
            plt.title(f"PPO Evaluation Performance - {cfg['ENV_NAME']}")
            plt.xlabel("Update Cycle")
            plt.ylabel("Average Evaluation Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close() # Close the plot to free memory
            print(f"Evaluation plot saved to {plot_path}")

        except Exception as e:
            print(f"Error saving evaluation log or plot: {e}")

    # --- Save Loss Logs ---
    try:
        agent.ppo_algorithm.save_loss_logs(results_dir)
    except Exception as e:
        print(f"Error calling save_loss_logs: {e}")

    # --- Final Recording using the new function ---
    final_model_path = os.path.join(model_dir, f"{agent_id_str}.pth")

    # Ensure a model is saved at final_model_path.
    # If best_eval_reward condition saved a model, it's already there.
    # Otherwise, save the current agent's state.
    if not os.path.exists(final_model_path):
        print(f"No 'best' model was saved during training. Saving current agent state.")
        agent.save_model(final_model_path)

    # Create a new agent instance for loading the saved model to ensure clean state if needed
    final_agent = PPOAgent(state_dim, action_dim, cfg, is_continuous=is_continuous)
    try:
        final_agent.load_model(final_model_path)
    except Exception as e:
        print(f"Error loading model from {final_model_path}: {e}. Using agent from memory for evaluation.")
        final_agent = agent # Fallback

    # Call the new function for final evaluation, rendering, and CSV logging
    record_final_evaluation(
        agent_id=agent_id_str,
        cfg=cfg_to_save,
        agent=final_agent
    )

    envs.close()

